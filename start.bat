@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Live Subtitle Translator
".venv\Scripts\python.exe" subtitle_translator.py
pause
