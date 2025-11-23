@echo off
REM ============================================================
REM Cryptocurrency Predictor - Windows Quick Start
REM ============================================================
REM This script runs the full pipeline with one click
REM ============================================================

echo.
echo ============================================================
echo   Cryptocurrency Price Predictor - Windows Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected
echo.
echo [INFO] Starting Cryptocurrency Prediction Pipeline...
echo.

REM Set UTF-8 encoding for better output
chcp 65001 >nul 2>&1

REM Run the pipeline
python run_full_pipeline.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline failed. See error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Pipeline completed successfully!
pause
