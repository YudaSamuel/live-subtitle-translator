# Oyun Altyazı Çevirmeni

Bilgisayarın ses çıkışını dinleyip konuşmaları **otomatik dil algılamayla Türkçe altyazıya** çeviren, tamamen yerel çalışan bir Windows uygulaması. İnternete veri gönderilmez.

```
Ses (WASAPI loopback) → Silero VAD → faster-whisper large-v3-turbo (GPU) → NLLB-200 1.3B (GPU) → Şeffaf altyazı penceresi
```

- Herhangi bir dildeki konuşmayı algılar (Whisper, 99 dil) ve Türkçeye çevirir (NLLB-200)
- Konuşma sürerken canlı ara altyazı, cümle bitince kesin çeviri (~0,3 sn / ~1,2 sn)
- Her zaman üstte, tıklamaları geçiren şeffaf altyazı penceresi; sistem tepsisi menüsü

## Gereksinimler
- Windows 10/11, NVIDIA ekran kartı (~3 GB boş VRAM)
- Python 3.12
- Oyun **Borderless / Windowed** modda olmalı (tam ekran exclusive modda overlay görünmez)

## Kurulum
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\modelleri_indir.ps1   # ~3 GB model
```

## Kullanım
`baslat.bat` dosyasına çift tıklayın. Kapatmak için tepsi simgesi → **Çıkış**.

## Ayarlar (`config.json`)
| Ayar | Açıklama |
|---|---|
| `allowed_languages` | Olası dilleri sınırlar, ör. `["en", "ru"]` (yanlış dil algılamayı azaltır) |
| `force_source_language` | Dili sabitler, ör. `"en"` |
| `translator_device` | `"cuda"` (hızlı) veya `"cpu"` (~1,4 GB VRAM tasarrufu, daha yavaş) |
| `partial_interval_seconds` | Canlı ara altyazı sıklığı; `0` kapatır |
| `min_silence_ms` | Cümle sonu için beklenen sessizlik |
| `font_size`, `bottom_margin_px`, `background_opacity` | Görünüm |

## Test
```powershell
.\.venv\Scripts\python.exe test_pipeline.py ornek.wav
```
