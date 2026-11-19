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


USER_SETTINGS = ROOT / "user_settings.json"


def load_config() -> dict:
    """config.json, overridden by user_settings.json (what the user set from the tray menu)."""
    with open(ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        with open(USER_SETTINGS, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def save_user_settings(**values) -> None:
    try:
        with open(USER_SETTINGS, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.update(values)
    with open(USER_SETTINGS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
def run_overlay(cfg: dict, start_pipeline, on_ui_ready=None) -> int:
    from PyQt6.QtCore import QObject, QPoint, QRect, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QColor, QCursor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import (
        QApplication, QGraphicsDropShadowEffect, QLabel, QMenu, QStyle,
        QSystemTrayIcon, QWidget,
    )

    target_lang = nllb_to_whisper(cfg["target_language"])

    class Bridge(QObject):
        subtitle = pyqtSignal(str, str, str, bool)
        status = pyqtSignal(str)

    class AreaEditor(QWidget):
        """A box the user drags/resizes to choose where subtitles appear.

        Drag inside to move, drag an edge or corner to resize, scroll to change the
        font size. Enter or double-click saves; Esc cancels.
        """

        EDGE = 12  # px from the border that counts as a resize handle
        MIN_W, MIN_H = 200, 60

        def __init__(self, rect: QRect, font_size: int, on_done):
            super().__init__()
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setMouseTracking(True)
            self.setGeometry(rect)
            self.font_size = font_size
            self.on_done = on_done
            self._drag_edges: tuple | None = None
            self._press_global = QPoint()
            self._press_geom = QRect()

        def _edges_at(self, pos: QPoint) -> tuple:
            e = self.EDGE
            return (
                pos.x() <= e,                      # left
                pos.y() <= e,                      # top
                pos.x() >= self.width() - e,       # right
                pos.y() >= self.height() - e,      # bottom
            )

        def _cursor_for(self, edges: tuple):
            left, top, right, bottom = edges
            if (left and top) or (right and bottom):
                return Qt.CursorShape.SizeFDiagCursor
            if (right and top) or (left and bottom):
                return Qt.CursorShape.SizeBDiagCursor
            if left or right:
                return Qt.CursorShape.SizeHorCursor
            if top or bottom:
                return Qt.CursorShape.SizeVerCursor
            return Qt.CursorShape.SizeAllCursor

        def paintEvent(self, _):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = self.rect().adjusted(1, 1, -2, -2)
            p.fillRect(r, QColor(0, 0, 0, 110))
            pen = QPen(QColor(255, 210, 0), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(r)
            # Sample subtitle at the chosen font size, bottom-aligned like real subtitles
            p.setPen(QColor(cfg["text_color"]))
            p.setFont(QFont(cfg["font_family"], self.font_size, QFont.Weight.DemiBold))
            p.drawText(
                r.adjusted(10, 6, -10, -8),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom | Qt.TextFlag.TextWordWrap,
                "Örnek altyazı / Sample subtitle",
            )
            p.setPen(QColor(255, 210, 0))
            p.setFont(QFont(cfg["font_family"], 10))
            p.drawText(
                r.adjusted(10, 6, -10, -6),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                f"Drag to move · drag edges to resize · scroll: font size ({self.font_size}) · "
                "Enter / double-click: save · Esc: cancel",
            )

        def mousePressEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self._drag_edges = self._edges_at(ev.position().toPoint())
                self._press_global = ev.globalPosition().toPoint()
                self._press_geom = QRect(self.geometry())

        def mouseMoveEvent(self, ev):
            if self._drag_edges is None:
                self.setCursor(QCursor(self._cursor_for(self._edges_at(ev.position().toPoint()))))
                return
            d = ev.globalPosition().toPoint() - self._press_global
            g = QRect(self._press_geom)
            left, top, right, bottom = self._drag_edges
            if not any(self._drag_edges):
                g.translate(d)
            else:
                if left:
                    g.setLeft(min(g.left() + d.x(), g.right() - self.MIN_W))
                if right:
                    g.setRight(max(g.right() + d.x(), g.left() + self.MIN_W))
                if top:
                    g.setTop(min(g.top() + d.y(), g.bottom() - self.MIN_H))
                if bottom:
                    g.setBottom(max(g.bottom() + d.y(), g.top() + self.MIN_H))
            # Keep the box on the desktop (across all monitors)
            desk = QRect()
            for s in QApplication.screens():
                desk = desk.united(s.geometry())
            g.setWidth(min(g.width(), desk.width()))
            g.setHeight(min(g.height(), desk.height()))
            g.moveLeft(max(desk.left(), min(g.left(), desk.right() + 1 - g.width())))
            g.moveTop(max(desk.top(), min(g.top(), desk.bottom() + 1 - g.height())))
            self.setGeometry(g)

        def mouseReleaseEvent(self, _):
            self._drag_edges = None

        def mouseDoubleClickEvent(self, _):
            self._finish(save=True)

        def wheelEvent(self, ev):
            step = 1 if ev.angleDelta().y() > 0 else -1
            self.font_size = max(10, min(72, self.font_size + step))
            self.update()

        def keyPressEvent(self, ev):
            if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish(save=True)
            elif ev.key() == Qt.Key.Key_Escape:
                self._finish(save=False)

        def _finish(self, save: bool):
            self.close()
            self.on_done(QRect(self.geometry()) if save else None, self.font_size)

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
            self.font_size = cfg["font_size"]
            self.label.setFont(QFont(cfg["font_family"], self.font_size, QFont.Weight.DemiBold))
            # User-chosen subtitle area [x, y, w, h] in screen coordinates (None = default)
            a = cfg.get("subtitle_area")
            self.area = QRect(*a) if a and len(a) == 4 else None
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

            # No layout on purpose: a layout keeps the window's minimum size from the
            # previous (longer) subtitle, which left a big empty box on screen.
            self.hide_timer = QTimer(self)
            self.hide_timer.setSingleShot(True)
            self.hide_timer.timeout.connect(self.hide)
            self.show_original = cfg["show_original"]

        def default_area(self) -> QRect:
            """Bottom-centre band of the primary screen (used until the user picks one)."""
            screen = QApplication.primaryScreen().geometry()
            w = int(screen.width() * cfg["max_width_ratio"])
            h = max(120, self.font_size * 5)
            return QRect(
                screen.x() + (screen.width() - w) // 2,
                screen.y() + screen.height() - h - cfg["bottom_margin_px"],
                w, h,
            )

        def current_area(self) -> QRect:
            if not self.area:
                return self.default_area()
            # If the saved area is no longer on any screen (monitor unplugged,
            # resolution changed), fall back to the default instead of hiding subtitles
            if QApplication.screenAt(self.area.center()) is None:
                return self.default_area()
            return QRect(self.area)

        def set_font_size(self, size: int) -> None:
            self.font_size = size
            self.label.setFont(QFont(cfg["font_family"], size, QFont.Weight.DemiBold))

        def show_text(self, html: str, timeout_s: float) -> None:
            area = self.current_area()
            self.label.setText(html)
            # Compute the correct height of a word-wrapped label for the area's width
            self.label.setMinimumSize(0, 0)
            self.label.setMaximumSize(16777215, 16777215)
            # Use the text's natural single-line width, capped at the area width
            # (a word-wrapped QLabel's sizeHint would pick a narrow "pleasant" width)
            self.label.setWordWrap(False)
            natural_w = self.label.sizeHint().width()
            self.label.setWordWrap(True)
            w = min(natural_w, area.width())
            h = self.label.heightForWidth(w)
            if h <= 0:
                h = self.label.sizeHint().height()
            self.label.setGeometry(0, 0, w, h)
            # Size the window exactly to the text: centred horizontally, anchored to the
            # bottom of the area (text grows upwards)
            self.setFixedSize(w, h)
            self.move(area.x() + (area.width() - w) // 2, area.bottom() + 1 - h)
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
                body = f"<span style='font-size:{self.font_size // 2}pt; color:#bbb'>[{lang.upper()}]</span> " + body
            if self.show_original and lang != target_lang:
                body += (
                    f"<br><span style='font-size:{int(self.font_size * 0.6)}pt; color:#ccc'>"
                    f"{h.escape(original)}</span>"
                )
            # Keep long sentences on screen longer
            timeout = max(cfg["subtitle_timeout_seconds"], len(translated) / 15)
            self.show_text(body, timeout)

        def on_status(self, msg: str) -> None:
            self.show_text(f"<span style='font-size:{self.font_size // 2 + 4}pt'>{msg}</span>", 4)

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

    # Choosing where subtitles appear
    editor_ref: dict = {}

    def on_area_chosen(rect, font_size):
        editor_ref.pop("w", None)
        if rect is None:  # cancelled: keep everything as it was
            return
        overlay.set_font_size(font_size)
        overlay.area = rect
        save_user_settings(
            subtitle_area=[rect.x(), rect.y(), rect.width(), rect.height()],
            font_size=font_size,
        )
        log(f"Subtitle area saved: {rect.x()},{rect.y()} {rect.width()}x{rect.height()}, font {font_size}")
        overlay.on_status("Position saved ✓")

    def open_area_editor():
        if "w" in editor_ref:
            editor_ref["w"].activateWindow()
            return
        overlay.hide()
        ed = AreaEditor(overlay.current_area(), overlay.font_size, on_area_chosen)
        editor_ref["w"] = ed
        ed.show()
        ed.raise_()
        ed.activateWindow()
        ed.setFocus()

    def reset_area():
        overlay.area = None
        save_user_settings(subtitle_area=None)
        overlay.on_status("Position reset to default")

    act_area = QAction("Position subtitles…", menu)
    act_area.triggered.connect(open_area_editor)
    act_reset = QAction("Reset position", menu)
    act_reset.triggered.connect(reset_area)
    menu.addAction(act_area)
    menu.addAction(act_reset)
    menu.addSeparator()
    menu.addAction(act_orig)
    menu.addAction(act_pause)
    menu.addSeparator()
    menu.addAction(act_quit)
    tray.setContextMenu(menu)
    # Double-clicking the tray icon also opens the position editor
    tray.activated.connect(
        lambda reason: open_area_editor()
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
    )
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
    if on_ui_ready:  # used by tests to drive the UI
        on_ui_ready(dict(app=app, overlay=overlay, editor=editor_ref,
                         open_area_editor=open_area_editor, reset_area=reset_area,
                         on_result=on_result))
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
