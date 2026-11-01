@echo off
chcp 65001 >nul
title Transcript to Text - Desktop Studio
cd /d "%~dp0"

echo ======================================================================
echo           TRANSCRIPT TO TEXT - UNG DUNG DESKTOP STUDIO
echo              Giao Dien Native CustomTkinter - Dark Mode
echo ======================================================================
echo.
echo [*] Dang khoi dong giao dien Desktop App...
echo.

if not "%~1"=="" (
    start "" python -m transcript_to_text.app_gui "%~1"
) else (
    start "" python -m transcript_to_text.app_gui
)

exit
