# ML Research Assistant Launcher Script (PowerShell)
# This script activates the virtual environment and runs the ML Research Assistant

Write-Host "🔬 ML Research Assistant" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Check if API key is set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "⚠️  OpenAI API key not set!" -ForegroundColor Yellow
    Write-Host "Please set your API key: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Yellow
    Write-Host ""
}

# Run the ML Research Assistant with passed arguments
if ($args.Count -eq 0) {
    # No arguments, run interactively
    python ml_research_assistant.py --interactive
} else {
    # Pass all arguments to the script
    python ml_research_assistant.py @args
}
