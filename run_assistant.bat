@echo off
REM ML Research Assistant Launcher Script
REM This script activates the virtual environment and runs the ML Research Assistant

echo 🔬 ML Research Assistant
echo =======================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if API key is set
if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  OpenAI API key not set!
    echo Please set your API key: set OPENAI_API_KEY=your-key-here
    echo Or run: $env:OPENAI_API_KEY='your-key-here' in PowerShell
    echo.
)

REM Run the ML Research Assistant with passed arguments
python ml_research_assistant.py %*

REM Keep window open if run without arguments
if "%~1"=="" (
    echo.
    echo Press any key to exit...
    pause >nul
)
