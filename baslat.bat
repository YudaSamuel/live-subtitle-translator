@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Oyun Altyazi Cevirmeni
".venv\Scripts\python.exe" altyazi.py
pause
