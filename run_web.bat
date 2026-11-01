@echo off
chcp 65001 >nul
title Transcript to Text - Web UI Server
cd /d "%~dp0"

echo ======================================================================
echo           TRANSCRIPT TO TEXT - GIAO DIEN WEB TRINH DUYET
echo            100%% Open Source - Faster-Whisper Q4 Quantized
echo ======================================================================
echo.
echo [*] Dang khoi dong Web UI tai dia chi: http://127.0.0.1:8000
echo [*] Trinh duyet mac dinh se tu dong mo trong 2 giay...
echo [*] Nhan Ctrl + C de dung server.
echo.

:: Mo trinh duyet sau 2 giay
start "" http://127.0.0.1:8000

:: Chay server FastAPI
python -m transcript_to_text.cli web --host 127.0.0.1 --port 8000

pause
