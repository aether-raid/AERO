#!/usr/bin/env python3
"""
ML Researcher LangGraph - Backend Startup Script
===============================================

Simple script to start the FastAPI backend server.
"""

import uvicorn
import sys
import os

def main():
    print("🚀 Starting ML Researcher LangGraph Backend...")
    print("=" * 50)
    
    # Check if required dependencies are installed
    try:
        import fastapi
        import ml_researcher_langgraph
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("pip install -r requirements_frontend.txt")
        sys.exit(1)
    
    # Start the server
    print("🌐 Backend will be available at:")
    print("   • Local:    http://localhost:8000")
    print("   • Network:  http://0.0.0.0:8000")
    print("\n📖 API Documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc:      http://localhost:8000/redoc")
    print("\n🎯 Frontend:")
    print("   • Open frontend.html in your browser")
    print("   • Or visit http://localhost:8000 for embedded frontend")
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start uvicorn server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
