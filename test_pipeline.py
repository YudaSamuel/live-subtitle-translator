"""Bir WAV dosyasıyla tanıma + çeviri hattını test eder:  python test_pipeline.py dosya.wav"""
import queue
import sys
import threading
import time
import wave

import numpy as np

import altyazi

path = sys.argv[1]
with wave.open(path) as w:
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype={2: np.int16, 4: np.int32}[sw])
audio = raw.astype(np.float32) / np.iinfo(raw.dtype).max
if ch > 1:
    audio = audio.reshape(-1, ch).mean(axis=1)
audio = altyazi.resample(audio, sr, altyazi.TARGET_SR)

cfg = altyazi.load_config()
stop = threading.Event()
audio_q, seg_q, text_q = queue.Queue(), queue.Queue(), queue.Queue()
translator = altyazi.Translator(cfg)
busy = threading.Event()
transcriber = altyazi.Transcriber(cfg, seg_q, text_q, stop, busy)
segmenter = altyazi.Segmenter(cfg, audio_q, seg_q, stop, busy)

results = []
t0 = 0.0


def on_result(lang, orig, tr, final=True):
    results.append((time.perf_counter() - t0, final, lang, orig, tr))


altyazi.TranslatorWorker(translator, text_q, on_result, stop).start()
transcriber.start()
segmenter.start()

# Sesi gerçek zamanlıymış gibi 50 ms'lik parçalarla besle, sonuna 1.5 sn sessizlik ekle
audio = np.concatenate([audio, np.zeros(int(1.5 * altyazi.TARGET_SR), dtype=np.float32)])
chunk = int(0.05 * altyazi.TARGET_SR)
time.sleep(1)
t0 = time.perf_counter()
for i in range(0, len(audio), chunk):
    audio_q.put(audio[i:i + chunk])
    time.sleep(0.05)
time.sleep(4)
stop.set()
loud = np.nonzero(np.abs(audio) > 0.02)[0]
speech_end = loud[-1] / altyazi.TARGET_SR if len(loud) else 0
print(f"\n=== SONUÇ ===  (konuşmanın bittiği an: {speech_end:.2f}s)")
for t, final, lang, orig, tr in results:
    kind = "KESIN" if final else "ara  "
    print(f"{t:6.2f}s {kind} [{lang}] TR: {tr}")
finals = [r for r in results if r[1]]
if finals:
    print(f"\nKonuşma bitişinden son kesin altyazıya: {finals[-1][0] - speech_end:.2f} sn")
if not results:
    print("Hiç altyazı üretilmedi!")
    sys.exit(1)
