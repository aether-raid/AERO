@echo off
echo.
echo ===============================================
echo  ML Researcher LangGraph - Backend Launcher
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if required packages are installed (FastAPI, Uvicorn, TreeQuest)
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, treequest" >nul 2>&1
if errorlevel 1 (
    echo ❌ Required packages not found (installing/updating...)
    echo Installing frontend dependencies...
    pip install -r requirements_frontend.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies ready
echo.

REM Start the backend
echo 🚀 Starting ML Researcher LangGraph Backend...
echo ⏰ This may take a moment to initialize...
echo.
echo 🌐 Once started, the frontend will be available at:
echo    http://localhost:8000
echo.
echo 📖 API documentation will be at:
echo    http://localhost:8000/docs
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo.
echo ===============================================
echo.

python -u start_backend.py

echo.
echo 👋 Backend stopped
pause
