# ML Researcher LangGraph - Frontend Launcher (PowerShell)
# Usage: .\start_frontend.ps1

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " ML Researcher LangGraph - Frontend Launcher" -ForegroundColor Cyan  
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check Python installation
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow
if (-not (Test-Command "python")) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$pythonVersion = python --version
Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "ml_researcher_langgraph.py")) {
    Write-Host "⚠️  Warning: ml_researcher_langgraph.py not found in current directory" -ForegroundColor Yellow
    Write-Host "Make sure you're running this script from the project root directory" -ForegroundColor Yellow
    Write-Host ""
}

# Check dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import fastapi, uvicorn" 2>$null
    Write-Host "✅ FastAPI and Uvicorn found" -ForegroundColor Green
} catch {
    Write-Host "❌ Required packages not found" -ForegroundColor Red
    Write-Host "📥 Installing frontend dependencies..." -ForegroundColor Yellow
    
    if (Test-Path "requirements_frontend.txt") {
        pip install -r requirements_frontend.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Host "❌ requirements_frontend.txt not found" -ForegroundColor Red
        Write-Host "Installing basic dependencies..." -ForegroundColor Yellow
        pip install fastapi uvicorn
    }
}

Write-Host ""

# Check if ML Researcher is available
Write-Host "🔍 Checking ML Researcher LangGraph..." -ForegroundColor Yellow
try {
    python -c "import ml_researcher_langgraph" 2>$null
    Write-Host "✅ ML Researcher LangGraph module found" -ForegroundColor Green
} catch {
    Write-Host "⚠️  ML Researcher LangGraph module not found" -ForegroundColor Yellow
    Write-Host "The backend will attempt to import it when started" -ForegroundColor Yellow
}

Write-Host ""

# Display startup information
Write-Host "🚀 Starting ML Researcher LangGraph Backend..." -ForegroundColor Green
Write-Host "⏰ This may take a moment to initialize..." -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 Once started, access the application at:" -ForegroundColor Cyan
Write-Host "   • Frontend:     http://localhost:8000" -ForegroundColor White
Write-Host "   • API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • Alternative:  Open frontend.html in your browser" -ForegroundColor White
Write-Host ""
Write-Host "📋 Available endpoints:" -ForegroundColor Cyan
Write-Host "   • POST /analyze     - Analyze research task" -ForegroundColor White
Write-Host "   • GET  /health      - Backend health check" -ForegroundColor White
Write-Host "   • GET  /history     - Analysis history" -ForegroundColor White
Write-Host ""
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Start the backend
try {
    python -u start_backend.py
} catch {
    Write-Host ""
    Write-Host "❌ Failed to start backend" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "👋 Backend stopped" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
