# Live Subtitle Translator

Real-time, fully local subtitles for anything playing on your PC — games, videos, streams, calls.
It listens to your audio output, **detects the spoken language automatically**, and shows a
translated subtitle at the bottom of the screen. Turkish is the default target; any of the
200 NLLB languages works.

No cloud APIs, no accounts, and no audio leaves your machine.

```
Audio output (WASAPI loopback)
   → Silero VAD           finds where speech starts and ends
   → faster-whisper       large-v3-turbo on GPU: language detection + transcription
   → NLLB-200 1.3B        CTranslate2 on GPU: translation into your language
   → Overlay              transparent, always-on-top, click-through subtitle window
```

## Features
- **Any source language → your language.** Whisper detects 99 languages; NLLB-200 translates between 200.
- **Live subtitles while someone is still talking.** A partial translation updates about every 0.8 s and is replaced by a higher-quality final one when the sentence ends.
- **Low latency.** On an RTX 4060 the full translation of a sentence appears about 0.3 s after the speaker stops; the final version arrives about 1.2 s after.
- **Language stickiness.** Short or noisy clips don't make the detected language jump around. You can also limit detection to a few languages.
- **Hallucination filtering.** Drops Whisper's well-known "Thanks for watching" and subtitle-credit lines on silence or music.
- **Small footprint.** About 3 GB of VRAM and about 1 GB of RAM.
- **Tray menu.** Show the original text, pause, or quit.

## Requirements
- Windows 10 or 11
- An NVIDIA GPU with about 3 GB of free VRAM. CPU-only works but is slower; see `translator_device` and `whisper_device`.
- Python 3.12
- About 3.5 GB of disk space for the models

## Installation
```powershell
git clone https://github.com/YudaSamuel/live-subtitle-translator.git
cd live-subtitle-translator
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\download_models.ps1   # ~3 GB, resumable
```

## Usage
Double-click `start.bat`, or run:
```powershell
.\.venv\Scripts\python.exe subtitle_translator.py
```
Loading the models takes about 10 s. When "Subtitles ready ✓" appears, play something.
To quit, right-click the tray icon and choose **Quit**, or press Ctrl+C in the console.

> **Games:** run them in **Borderless / Windowed** mode. Exclusive fullscreen draws over any overlay window.

## Configuration (`config.json`)
| Key | Default | Description |
|---|---|---|
| `target_language` | `"tur_Latn"` | Subtitle language as an [NLLB code](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200), e.g. `"eng_Latn"`, `"deu_Latn"`, `"spa_Latn"` |
| `allowed_languages` | `[]` | Restrict detection to these Whisper codes, e.g. `["en", "ru"]`. This greatly reduces wrong-language errors. |
| `force_source_language` | `null` | Always assume this source language, e.g. `"en"` |
| `translator_device` | `"cuda"` | `"cpu"` frees about 1.4 GB of VRAM at the cost of 1–2 s of extra latency |
| `whisper_beam_size` / `translator_beam_size` | `3` / `4` | Beam sizes for the final pass. Partials always use 1. |
| `partial_interval_seconds` | `0.8` | How often live partial subtitles update; `0` disables them |
| `min_silence_ms` | `350` | Silence that ends a sentence. Raise it if sentences get split. |
| `output_device_name` | `null` | Capture a specific output device (substring match) instead of the default |
| `font_size`, `bottom_margin_px`, `background_opacity`, `max_width_ratio`, `subtitle_timeout_seconds` | | Appearance |
| `show_original`, `show_language_tag` | `false` / `true` | Also show the source text and its language |

## Testing
`test_pipeline.py` feeds a WAV file through the pipeline in real time and reports the latency:
```powershell
.\.venv\Scripts\python.exe test_pipeline.py sample.wav
```
Set `SUBTITLE_DEBUG=1` to log when each final segment is sent.

## Troubleshooting
- **`DLL load failed ... blocked by Application Control policy`:** Windows Smart App Control blocks some unsigned native modules. This project avoids `sentencepiece` for that reason. If another module is blocked, check Windows Security → App & browser control.
- **No subtitles:** make sure audio is playing through the **default** output device, or set `output_device_name`.
- **Wrong language detected:** set `allowed_languages` or `force_source_language`.
- **Slang, profanity and names are translated poorly:** this is a limitation of NLLB, which is trained mostly on formal text.

## Models and licenses
The code is released under the [MIT License](LICENSE). The models are downloaded separately and have their own licenses:
- [faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo), converted from OpenAI Whisper (MIT)
- [NLLB-200 distilled 1.3B (CTranslate2 int8)](https://huggingface.co/JustFrederik/nllb-200-distilled-1.3B-ct2-int8), from Meta's NLLB-200 (**CC-BY-NC 4.0, non-commercial use only**)
- Silero VAD, bundled with faster-whisper (MIT)
