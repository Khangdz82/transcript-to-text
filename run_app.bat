@echo off
chcp 65001 >nul
title Transcript to Text - App Launcher
cd /d "%~dp0"

echo ======================================================================
echo           TRANSCRIPT TO TEXT - PHAN MEM BOC TACH TRANSCRIPT
echo            100%% Open Source - Faster-Whisper Q4 Quantized
echo ======================================================================
echo.

:: Neu co file keo tha truc tiep vao run_app.bat
if not "%~1"=="" (
    echo [*] Phat hien file keo tha: "%~1"
    echo [*] Dang chay boc tach voi model large-v3-turbo (Q4 INT8)...
    echo.
    python -m transcript_to_text.cli run "%~1" -m large-v3-turbo -f all
    goto end
)

:: Menu huong dan
echo Huong dan:
echo   - Ban co the keo tha truc tiep file Video/Audio vao bieu tuong run_app.bat
echo   - Hoac nhap duong dan file / link YouTube ben duoi:
echo.

set /p INPUT=">> Nhap duong dan file hoac link YouTube: "
if "%INPUT%"=="" (
    echo [!] Ban chua nhap duong dan. Thoat chuong trinh.
    goto end
)

echo.
echo Chon Model AI (luon chay che do Q4 INT8 tiet kiem RAM/VRAM):
echo   [1] large-v3-turbo (Khuyen dung: Sieu chuan SOTA, nhanh, ~810MB, RAM 3-4GB) [Mac dinh]
echo   [2] small          (Chuan tot, can bang, ~250MB, RAM 2GB)
echo   [3] medium         (Rat chuan, phuc tap, ~770MB, RAM 4GB)
echo   [4] base           (Nhe, nhanh, hoi thoai co ban, ~75MB, RAM 1GB)
echo   [5] large-v3       (Chinh xac toi da, da ngon ngu, ~1.5GB, RAM 6-8GB)
echo   [6] tiny           (Sieu toc, test nhanh, ~40MB, RAM 500MB)
echo.
set /p M_CHOICE=">> Chon model [1-6] (Nhan Enter de chon 1): "

set MODEL=large-v3-turbo
if "%M_CHOICE%"=="2" set MODEL=small
if "%M_CHOICE%"=="3" set MODEL=medium
if "%M_CHOICE%"=="4" set MODEL=base
if "%M_CHOICE%"=="5" set MODEL=large-v3
if "%M_CHOICE%"=="6" set MODEL=tiny

echo.
echo [*] Dang bat dau boc tach voi model: %MODEL% (Q4 INT8)...
echo.
python -m transcript_to_text.cli run "%INPUT%" -m %MODEL% -f all

:end
echo.
echo ======================================================================
echo Hoan tat! Nhan phim bat ky de thoat...
pause >nul
