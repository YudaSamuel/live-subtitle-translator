"""
Live Subtitle Translator
------------------------
Listens to your PC's audio output (speakers / headphones), detects speech,
identifies its language automatically and shows a translated subtitle at the
bottom of the screen (Turkish by default).

  Audio (WASAPI loopback) -> Silero VAD -> faster-whisper (GPU) -> NLLB-200 (GPU) -> Overlay

Everything runs locally; no data leaves your machine.
"""

from __future__ import annotations

import json
import os
import queue
import re
import site
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


# --------------------------------------------------------------------------- #
#  Make the pip-installed CUDA DLLs (nvidia-cublas / nvidia-cudnn) loadable
# --------------------------------------------------------------------------- #
def _register_cuda_dlls() -> None:
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            p = Path(sp) / sub
            if p.is_dir():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()

import numpy as np  # noqa: E402

from languages import nllb_to_whisper, to_nllb  # noqa: E402

TARGET_SR = 16000


def log(*args) -> None:
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  1) Audio capture — loopback of the default output device
# --------------------------------------------------------------------------- #
class LoopbackCapture:
    def __init__(self, out_queue: "queue.Queue[np.ndarray]", device_name: str | None = None):
        import pyaudiowpatch as pyaudio

        self.pa = pyaudio.PyAudio()
        self.out_queue = out_queue
        self.device = self._find_device(device_name)
        self.rate = int(self.device["defaultSampleRate"])
        self.channels = int(self.device["maxInputChannels"])
        self.stream = None

    def _find_device(self, name: str | None) -> dict:
        if name:
            for dev in self.pa.get_loopback_device_info_generator():
                if name.lower() in dev["name"].lower():
                    return dev
            log(f"Device '{name}' not found, using the default output.")
        return self.pa.get_default_wasapi_loopback()

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        audio = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels).mean(axis=1)
        self.out_queue.put(resample(audio, self.rate, TARGET_SR))
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        import pyaudiowpatch as pyaudio

        log(f"Audio source: {self.device['name']}  ({self.rate} Hz, {self.channels} channels)")
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device["index"],
            frames_per_buffer=int(self.rate * 0.05),
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    ratio = src_sr / dst_sr
    if ratio.is_integer():
        # Simple low-pass (box average) + decimation
        n = int(ratio)
        usable = len(audio) - len(audio) % n
        return audio[:usable].reshape(-1, n).mean(axis=1).astype(np.float32)
    n_out = int(round(len(audio) / ratio))
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# --------------------------------------------------------------------------- #
#  2) Segmentation — find sentence boundaries with Silero VAD
# --------------------------------------------------------------------------- #
class Segmenter(threading.Thread):
    def __init__(self, cfg: dict, audio_q: queue.Queue, seg_q: queue.Queue, stop: threading.Event,
                 asr_busy: threading.Event | None = None):
        super().__init__(daemon=True, name="segmenter")
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self._get_ts = get_speech_timestamps
        self.vad_opts = VadOptions(
            threshold=cfg["vad_threshold"],
            min_silence_duration_ms=200,
            speech_pad_ms=60,
            min_speech_duration_ms=200,
        )
        self.min_silence = int(cfg["min_silence_ms"] / 1000 * TARGET_SR)
        self.max_len = int(cfg["max_segment_seconds"] * TARGET_SR)
        self.min_len = int(cfg["min_segment_seconds"] * TARGET_SR)
        self.pad = int(0.2 * TARGET_SR)
        self.audio_q, self.seg_q, self.stop_ev = audio_q, seg_q, stop
        # Live (partial) subtitles: while someone is talking, periodically translate what's been said so far
        self.partial_interval = cfg.get("partial_interval_seconds", 0.8)
        self.partial_min = int(cfg.get("partial_min_seconds", 1.0) * TARGET_SR)
        self.asr_busy = asr_busy or threading.Event()  # set while the Transcriber is working

    def _emit(self, audio: np.ndarray) -> None:
        if len(audio) >= self.min_len:
            if os.environ.get("SUBTITLE_DEBUG"):
                log(f"   [final segment sent, {len(audio) / TARGET_SR:.2f}s of audio]")
            self.seg_q.put(("final", audio))

    def _maybe_partial(self, audio: np.ndarray) -> None:
        if (
            self.partial_interval > 0
            and len(audio) >= self.partial_min
            and time.monotonic() - self.last_partial >= self.partial_interval
            and not self.asr_busy.is_set()
            and self.seg_q.empty()
        ):
            self.last_partial = time.monotonic()
            self.seg_q.put(("partial", audio.copy()))

    def run(self) -> None:
        buf = np.zeros(0, dtype=np.float32)
        last_check = time.monotonic()
        self.last_partial = 0.0
        while not self.stop_ev.is_set():
            try:
                buf = np.concatenate([buf, self.audio_q.get(timeout=0.05)])
            except queue.Empty:
                pass
            if time.monotonic() - last_check < 0.1 or len(buf) < TARGET_SR // 2:
                continue
            last_check = time.monotonic()

            # No need to run VAD on near-silence
            if np.abs(buf).max() < 1e-3:
                buf = buf[-TARGET_SR:]
                continue

            speech = self._get_ts(buf, self.vad_opts)
            if not speech:
                buf = buf[-TARGET_SR:]  # keep 1 s so we don't miss the start of speech
                continue

            start = max(0, speech[0]["start"] - self.pad)
            end_of_speech = speech[-1]["end"]

            if len(buf) - end_of_speech >= self.min_silence:
                # Sentence finished
                self._emit(buf[start:min(len(buf), end_of_speech + self.pad)])
                buf = buf[end_of_speech:]
            elif len(buf) - start >= self.max_len:
                # Very long speech: cut at a gap between utterances if possible
                cut = speech[-2]["end"] if len(speech) > 1 else len(buf)
                self._emit(buf[start:cut])
                buf = buf[cut:]
            else:
                trailing_silence = len(buf) - end_of_speech
                if start > 0:
                    buf = buf[start:]
                # Only produce a partial while speech is actually ongoing; once the speaker
                # has gone quiet, wait for the final segment instead of delaying it
                if trailing_silence < int(0.15 * TARGET_SR):
                    self._maybe_partial(buf)


# --------------------------------------------------------------------------- #
#  3) Speech recognition — faster-whisper with automatic language detection
# --------------------------------------------------------------------------- #
# Known Whisper hallucinations on silence/music (YouTube subtitle credits etc.)
HALLUCINATIONS = re.compile(
    r"(thank(s| you) for watching|(please|and) subscribe|subtitles by|amara\.org|altyazı|"
    r"untertitel|sous-titres|sottotitoli|subtítulos|legendas|napisy|"
    r"субтитр|dimatorzok|продолжение следует|спасибо за просмотр|"
    r"ご視聴ありがとう|字幕|자막)",
    re.IGNORECASE,
)


class Transcriber(threading.Thread):
    def __init__(self, cfg: dict, seg_q: queue.Queue, text_q: queue.Queue, stop: threading.Event,
                 busy: threading.Event | None = None):
        super().__init__(daemon=True, name="transcriber")
        from faster_whisper import WhisperModel
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens

        self.busy = busy or threading.Event()
        self.beam = cfg.get("whisper_beam_size", 3)
        self.last_lang: str | None = None
        self.allowed = set(cfg.get("allowed_languages") or [])
        self.min_lang_prob = cfg.get("min_language_probability", 0.3)

        self._pad_or_trim = pad_or_trim
        self._compression_ratio = get_compression_ratio
        self._tok_cache: dict = {}
        self._tokenizer = lambda lang: self._tok_cache.setdefault(
            lang, Tokenizer(self.model.hf_tokenizer, True, task="transcribe", language=lang)
        )
        self._suppress = lambda tok: get_suppressed_tokens(tok, [-1])

        log(f"Loading Whisper model: {cfg['whisper_model']} ({cfg['whisper_device']}) ...")
        local = ROOT / cfg["whisper_model"]
        self.model = WhisperModel(
            str(local) if local.is_dir() else cfg["whisper_model"],
            device=cfg["whisper_device"],
            compute_type=cfg["whisper_compute_type"],
            download_root=str(MODELS_DIR / "whisper"),
        )
        self.force_lang = cfg.get("force_source_language")
        self.seg_q, self.text_q, self.stop_ev = seg_q, text_q, stop
        # Warm-up so the first real call isn't slow
        list(self.model.transcribe(np.zeros(TARGET_SR, dtype=np.float32), language="en")[0])
        log("Whisper ready.")

    def run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                kind, audio = self.seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # If something newer is already waiting, skip this stale partial
            if kind == "partial" and not self.seg_q.empty():
                continue
            self.busy.set()
            try:
                self._process(kind, audio)
            finally:
                self.busy.clear()

    def _choose_language(self, encoder_output) -> tuple[str, float]:
        """Detect the language, keeping it stable on short/noisy segments."""
        if self.force_lang:
            return self.force_lang, 1.0
        results = self.model.model.detect_language(encoder_output)[0]
        probs = {tok[2:-2]: p for tok, p in results}
        if self.allowed:
            # If the user restricted the possible languages, only pick among those
            probs = {k: v for k, v in probs.items() if k in self.allowed} or probs
        top = max(probs, key=probs.get)
        # Stickiness: stay with the previous language if it's still plausible
        # and the new one isn't a confident switch
        if (
            self.last_lang
            and top != self.last_lang
            and probs[top] < 0.85
            and probs.get(self.last_lang, 0) >= 0.15
        ):
            top = self.last_lang
        return top, probs[top]

    def _process(self, kind: str, audio: np.ndarray) -> None:
        partial = kind == "partial"
        t0 = time.perf_counter()
        # Single encoder pass reused for both language detection and decoding.
        # (faster-whisper's transcribe() runs the encoder a second time to detect the language.)
        m = self.model
        feats = self._pad_or_trim(m.feature_extractor(audio[: 30 * TARGET_SR]))
        enc = m.encode(feats)
        lang, lang_prob = self._choose_language(enc)
        # Very uncertain language usually means this isn't speech (music / sound effects)
        if lang_prob < self.min_lang_prob:
            return
        tok = self._tokenizer(lang)
        result = m.model.generate(
            enc,
            [list(tok.sot_sequence) + [tok.no_timestamps]],
            beam_size=1 if partial else self.beam,
            max_length=m.max_length,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=True,
            suppress_tokens=self._suppress(tok),
        )[0]
        tokens = [t for t in result.sequences_ids[0] if t < tok.eot]
        text = tok.decode(tokens).strip()
        avg_logprob = result.scores[0] * len(tokens) / (len(tokens) + 1) if tokens else -9
        if result.no_speech_prob > 0.6 and avg_logprob < -0.8:
            return  # no speech
        if self._compression_ratio(text) > 2.4:
            return  # repetitive hallucination
        if not text or not re.search(r"\w", text) or HALLUCINATIONS.search(text):
            return
        dt = time.perf_counter() - t0
        if not partial and lang_prob >= 0.5:
            self.last_lang = lang
        tag = f"partial {lang}" if partial else f"{lang} {lang_prob:.0%}"
        log(f"[{tag}] ({dt:.2f}s) {text}")
        self.text_q.put((kind, lang, text))


# --------------------------------------------------------------------------- #
#  4) Translation — NLLB-200 (CTranslate2)
# --------------------------------------------------------------------------- #
class Translator:
    def __init__(self, cfg: dict):
        import ctranslate2
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        repo = cfg["translator_repo"]
        log(f"Loading translation model: {repo} ({cfg['translator_device']}) ...")
        local = MODELS_DIR / repo.split("/")[-1]
        if (local / "model.bin").is_file() and (local / "tokenizer.json").is_file():
            path = str(local)
        else:
            path = snapshot_download(
                repo,
                local_dir=str(local),
                allow_patterns=["*.bin", "*.json", "*.txt"],
            )
        # HF `tokenizers` is used instead of `sentencepiece`, because Windows Smart App
        # Control can block sentencepiece's native DLL.
        self.tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.model = ctranslate2.Translator(
            path,
            device=cfg["translator_device"],
            compute_type=cfg.get("translator_compute_type", "int8")
            if cfg["translator_device"] == "cuda" else "int8",
            intra_threads=cfg["translator_threads"],
        )
        self.target = cfg["target_language"]
        self.beam = cfg.get("translator_beam_size", 4)
        log("Translation model ready.")

    def translate(self, text: str, src_whisper_lang: str, beam: int | None = None) -> str:
        src = to_nllb(src_whisper_lang)
        if src == self.target:
            return text
        # NLLB works best per sentence: split into sentences and translate as a batch
        sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
        batch = [
            [src] + self.tok.encode(s, add_special_tokens=False).tokens + ["</s>"]
            for s in sentences
        ]
        results = self.model.translate_batch(
            batch,
            target_prefix=[[self.target]] * len(batch),
            beam_size=beam or self.beam,
            max_decoding_length=200,
            repetition_penalty=1.1,
        )
        out = []
        for r in results:
            ids = [self.tok.token_to_id(t) for t in r.hypotheses[0] if t != self.target]
            text_out = self.tok.decode([i for i in ids if i is not None], skip_special_tokens=True)
            # NLLB sometimes prepends a dialogue dash ("- Thank you.")
            out.append(re.sub(r"^\s*[-–—]\s*", "", text_out))
        return " ".join(out).strip()


class TranslatorWorker(threading.Thread):
    def __init__(self, translator: Translator, text_q: queue.Queue, on_result, stop: threading.Event):
        super().__init__(daemon=True, name="translator")
        self.tr, self.text_q, self.on_result, self.stop_ev = translator, text_q, on_result, stop
        self.last_key: tuple | None = None
        self.last_tr = ""

    def run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                kind, lang, text = self.text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            # Don't waste time on a stale partial when newer text is already waiting
            if kind == "partial" and not self.text_q.empty():
                continue
            t0 = time.perf_counter()
            # Partials: beam=1 for speed. Finals: wider beam for quality.
            partial = kind == "partial"
            key = (lang, text.strip().lower(), partial)
            if key == self.last_key:
                tr = self.last_tr  # same text as last time: don't translate again
            else:
                try:
                    tr = self.tr.translate(text, lang, 1 if partial else None)
                except Exception as e:  # a translation error must not stop the subtitles
                    log("Translation error:", e)
                    continue
                self.last_key, self.last_tr = key, tr
            log(f"   -> ({time.perf_counter() - t0:.2f}s) {tr}")
            self.on_result(lang, text, tr, kind == "final")


# --------------------------------------------------------------------------- #
#  5) Subtitle window — transparent, always on top, click-through
# --------------------------------------------------------------------------- #
def run_overlay(cfg: dict, start_pipeline) -> int:
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QColor, QFont
    from PyQt6.QtWidgets import (
        QApplication, QGraphicsDropShadowEffect, QLabel, QMenu, QStyle,
        QSystemTrayIcon, QVBoxLayout, QWidget,
    )

    target_lang = nllb_to_whisper(cfg["target_language"])

    class Bridge(QObject):
        subtitle = pyqtSignal(str, str, str, bool)
        status = pyqtSignal(str)

    class Overlay(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

            alpha = int(255 * cfg["background_opacity"])
            self.label = QLabel(self)
            self.label.setWordWrap(True)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setTextFormat(Qt.TextFormat.RichText)
            self.label.setFont(QFont(cfg["font_family"], cfg["font_size"], QFont.Weight.DemiBold))
            self.label.setStyleSheet(
                f"color: {cfg['text_color']};"
                f"background-color: rgba(0,0,0,{alpha});"
                "border-radius: 10px; padding: 10px 22px;"
            )
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(6)
            shadow.setOffset(0, 0)
            shadow.setColor(QColor(cfg["outline_color"]))
            self.label.setGraphicsEffect(shadow)

            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.label)

            self.hide_timer = QTimer(self)
            self.hide_timer.setSingleShot(True)
            self.hide_timer.timeout.connect(self.hide)
            self.show_original = cfg["show_original"]

        def show_text(self, html: str, timeout_s: float) -> None:
            screen = QApplication.primaryScreen().geometry()
            max_w = int(screen.width() * cfg["max_width_ratio"])
            self.label.setText(html)
            # Compute the correct height of a word-wrapped label for the chosen width
            w = min(self.label.sizeHint().width(), max_w)
            h = self.label.heightForWidth(w)
            if h <= 0:
                h = self.label.sizeHint().height()
            self.label.setFixedSize(w, h)
            self.setGeometry(
                screen.x() + (screen.width() - w) // 2,
                screen.y() + screen.height() - h - cfg["bottom_margin_px"],
                w, h,
            )
            self.show()
            self.raise_()
            self.hide_timer.start(int(timeout_s * 1000))

        def on_subtitle(self, lang: str, original: str, translated: str, final: bool = True) -> None:
            import html as h

            body = h.escape(translated)
            if not final:
                # Still speaking: partial text, slightly dimmer, ending with "…"
                body = f"<span style='color:{cfg['partial_text_color']}'>{body.rstrip('.')}…</span>"
            if cfg["show_language_tag"] and lang != target_lang:
                body = f"<span style='font-size:{cfg['font_size'] // 2}pt; color:#bbb'>[{lang.upper()}]</span> " + body
            if self.show_original and lang != target_lang:
                body += (
                    f"<br><span style='font-size:{int(cfg['font_size'] * 0.6)}pt; color:#ccc'>"
                    f"{h.escape(original)}</span>"
                )
            # Keep long sentences on screen longer
            timeout = max(cfg["subtitle_timeout_seconds"], len(translated) / 15)
            self.show_text(body, timeout)

        def on_status(self, msg: str) -> None:
            self.show_text(f"<span style='font-size:{cfg['font_size'] // 2 + 4}pt'>{msg}</span>", 4)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()
    bridge = Bridge()
    bridge.subtitle.connect(overlay.on_subtitle)
    bridge.status.connect(overlay.on_status)

    # System tray menu
    tray = QSystemTrayIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume), app)
    tray.setToolTip("Live Subtitle Translator")
    menu = QMenu()
    act_orig = QAction("Show original text", menu, checkable=True)
    act_orig.setChecked(overlay.show_original)
    act_orig.toggled.connect(lambda v: setattr(overlay, "show_original", v))
    paused = threading.Event()
    act_pause = QAction("Pause", menu, checkable=True)
    act_pause.toggled.connect(lambda v: paused.set() if v else paused.clear())
    act_quit = QAction("Quit", menu)
    act_quit.triggered.connect(app.quit)
    menu.addAction(act_orig)
    menu.addAction(act_pause)
    menu.addSeparator()
    menu.addAction(act_quit)
    tray.setContextMenu(menu)
    tray.show()

    def on_result(lang, original, translated, final=True):
        if not paused.is_set():
            bridge.subtitle.emit(lang, original, translated, final)

    # Periodically hand control back to Python so Ctrl+C in the console works
    import signal

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QTimer()
    tick.start(300)
    tick.timeout.connect(lambda: None)

    bridge.status.emit("Loading models…")
    stop_pipeline = start_pipeline(on_result, bridge.status.emit, app.quit)
    code = app.exec()
    stop_pipeline()
    return code


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config()
    stop = threading.Event()
    audio_q: queue.Queue = queue.Queue()
    seg_q: queue.Queue = queue.Queue()
    text_q: queue.Queue = queue.Queue()
    holder: dict = {}

    def start_pipeline(on_result, on_status, on_fatal):
        def boot():
            try:
                translator = Translator(cfg)
                asr_busy = threading.Event()
                transcriber = Transcriber(cfg, seg_q, text_q, stop, asr_busy)
                segmenter = Segmenter(cfg, audio_q, seg_q, stop, asr_busy)
                capture = LoopbackCapture(audio_q, cfg.get("output_device_name"))
                holder["capture"] = capture
                TranslatorWorker(translator, text_q, on_result, stop).start()
                transcriber.start()
                segmenter.start()
                capture.start()
                log("Ready! Translating speech into subtitles. (Quit with Ctrl+C or the tray menu)")
                on_status("Subtitles ready ✓")
            except Exception as e:
                import traceback

                traceback.print_exc()
                log("Startup error:", e)
                on_status(f"Error: {e}")
                time.sleep(5)
                on_fatal()

        threading.Thread(target=boot, daemon=True, name="boot").start()

        def stop_all():
            stop.set()
            if "capture" in holder:
                holder["capture"].stop()

        return stop_all

    return run_overlay(cfg, start_pipeline)


if __name__ == "__main__":
    sys.exit(main())
