#!/usr/bin/env python3
"""
ML Researcher LangGraph - Web Frontend Backend
==============================================

FastAPI backend that provides REST API endpoints for the ML Researcher LangGraph system.
Supports both research planning and model suggestion workflows with real-time streaming.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our ML Researcher LangGraph system
from ml_researcher_langgraph import MLResearcherLangGraph

# Initialize FastAPI app
app = FastAPI(
    title="ML Researcher LangGraph API",
    description="AI-powered research assistant with model suggestions and research planning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML Researcher instance
ml_researcher = None

# Pydantic models for API
class ResearchRequest(BaseModel):
    prompt: str
    stream: bool = False

class AnalysisResponse(BaseModel):
    workflow_type: str
    router_decision: Dict[str, Any]
    results: Dict[str, Any]
    timestamp: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    version: str
    workflows_available: List[str]

# In-memory storage for analysis results (in production, use a database)
analysis_history: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the ML Researcher on startup."""
    global ml_researcher
    try:
        ml_researcher = MLResearcherLangGraph()
        print("✅ ML Researcher LangGraph initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ML Researcher: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend HTML page."""
    return HTMLResponse(content=get_frontend_html(), status_code=200)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        workflows_available=["model_suggestion", "research_planning", "direct_llm"]
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_research_task(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Analyze a research task using the LangGraph system."""
    if not ml_researcher:
        raise HTTPException(status_code=500, detail="ML Researcher not initialized")
    
    start_time = datetime.now()
    
    try:
        # Run the analysis
        results = await ml_researcher.analyze_research_task(request.prompt)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create response
        response = AnalysisResponse(
            workflow_type=results.get("workflow_type", "unknown"),
            router_decision=results.get("router_decision", {}),
            results=results,
            timestamp=start_time.isoformat(),
            processing_time=processing_time
        )
        
        # Store in history
        analysis_record = {
            "id": len(analysis_history) + 1,
            "prompt": request.prompt,
            "response": response.dict(),
            "timestamp": start_time.isoformat()
        }
        analysis_history.append(analysis_record)
        
        # Save to file in background
        background_tasks.add_task(save_analysis_to_file, analysis_record)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/stream")
async def analyze_research_task_stream(request: ResearchRequest):
    """Stream the analysis results in real-time."""
    if not ml_researcher:
        raise HTTPException(status_code=500, detail="ML Researcher not initialized")
    
    async def generate_stream():
        try:
            # This is a simplified streaming approach
            # In a full implementation, you'd modify the LangGraph to yield intermediate results
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...'})}\n\n"
            
            # Run the analysis
            results = await ml_researcher.analyze_research_task(request.prompt)
            
            # Stream the final results
            yield f"data: {json.dumps({'type': 'result', 'data': results})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/history")
async def get_analysis_history():
    """Get the analysis history."""
    return {"history": analysis_history[-20:]}  # Return last 20 analyses

@app.get("/history/{analysis_id}")
async def get_analysis_by_id(analysis_id: int):
    """Get a specific analysis by ID."""
    for analysis in analysis_history:
        if analysis["id"] == analysis_id:
            return analysis
    raise HTTPException(status_code=404, detail="Analysis not found")

async def save_analysis_to_file(analysis_record: Dict[str, Any]):
    """Save analysis to file in background."""
    try:
        # Create Past_analysis directory if it doesn't exist
        os.makedirs("Past_analysis", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.fromisoformat(analysis_record["timestamp"]).strftime("%Y%m%d_%H%M%S")
        workflow_type = analysis_record["response"]["workflow_type"]
        filename = f"Past_analysis/ml_research_analysis_{workflow_type}_web_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_record, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"❌ Failed to save analysis to file: {str(e)}")

def get_frontend_html() -> str:
    """Return the frontend HTML content."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Researcher LangGraph</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-shadow {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .workflow-badge {
            @apply inline-flex items-center px-3 py-1 rounded-full text-sm font-medium;
        }
        .model-suggestion { @apply bg-green-100 text-green-800; }
        .research-planning { @apply bg-purple-100 text-purple-800; }
        .direct-llm { @apply bg-blue-100 text-blue-800; }
        .unknown { @apply bg-gray-100 text-gray-800; }
        
        .result-animation {
            animation: slideInUp 0.5s ease-out;
        }
        
        @keyframes slideInUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .spinner {
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-50">
    <div id="app">
        <!-- Header -->
        <header class="gradient-bg text-white shadow-lg">
            <div class="container mx-auto px-4 py-6">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                        <i class="fas fa-microscope text-3xl"></i>
                        <div>
                            <h1 class="text-3xl font-bold">ML Researcher LangGraph</h1>
                            <p class="text-purple-200">AI-powered research assistant with workflow orchestration</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-4">
                        <span class="workflow-badge bg-white text-purple-700" v-if="healthStatus">
                            <i class="fas fa-check-circle mr-1"></i>
                            {{ healthStatus.status }}
                        </span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="container mx-auto px-4 py-8">
            <!-- Input Section -->
            <div class="card-shadow bg-white rounded-lg p-6 mb-8">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">
                    <i class="fas fa-search mr-2"></i>
                    Research Task Analysis
                </h2>
                
                <div class="space-y-4">
                    <textarea
                        v-model="prompt"
                        @keydown.ctrl.enter="analyzeTask"
                        placeholder="Enter your machine learning research task or question here...

Examples:
• What's the best model for image classification with limited data?
• Generate a research plan for adversarial robustness in deep learning
• How to improve transformer efficiency for mobile deployment?"
                        class="w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                        :disabled="isAnalyzing"
                    ></textarea>
                    
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-4">
                            <label class="flex items-center">
                                <input type="checkbox" v-model="streamMode" class="mr-2">
                                <span class="text-sm text-gray-600">Stream results</span>
                            </label>
                        </div>
                        
                        <button
                            @click="analyzeTask"
                            :disabled="!prompt.trim() || isAnalyzing"
                            class="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                        >
                            <i class="fas fa-spinner spinner" v-if="isAnalyzing"></i>
                            <i class="fas fa-play" v-else></i>
                            <span>{{ isAnalyzing ? 'Analyzing...' : 'Analyze Task' }}</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Results Section -->
            <div v-if="analysisResult" class="result-animation card-shadow bg-white rounded-lg p-6 mb-8">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-2xl font-bold text-gray-800">
                        <i class="fas fa-chart-bar mr-2"></i>
                        Analysis Results
                    </h2>
                    <div class="flex items-center space-x-2">
                        <span :class="'workflow-badge ' + getWorkflowClass(analysisResult.workflow_type)">
                            <i :class="getWorkflowIcon(analysisResult.workflow_type) + ' mr-1'"></i>
                            {{ analysisResult.workflow_type.replace('_', ' ').toUpperCase() }}
                        </span>
                        <span class="text-sm text-gray-500">
                            {{ formatProcessingTime(analysisResult.processing_time) }}
                        </span>
                    </div>
                </div>

                <!-- Router Decision -->
                <div class="bg-gray-50 rounded-lg p-4 mb-6">
                    <h3 class="font-semibold text-gray-800 mb-2">
                        <i class="fas fa-route mr-2"></i>
                        Router Decision
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <span class="text-sm text-gray-600">Confidence:</span>
                            <div class="font-mono text-lg">{{ (analysisResult.router_decision.confidence * 100).toFixed(1) }}%</div>
                        </div>
                        <div class="md:col-span-2">
                            <span class="text-sm text-gray-600">Reasoning:</span>
                            <div class="text-gray-800">{{ analysisResult.router_decision.reasoning }}</div>
                        </div>
                    </div>
                </div>

                <!-- Results Content -->
                <div class="space-y-6">
                    <!-- Model Suggestions -->
                    <div v-if="analysisResult.workflow_type === 'model_suggestion' && analysisResult.results.model_suggestions">
                        <h3 class="font-semibold text-gray-800 mb-3">
                            <i class="fas fa-robot mr-2"></i>
                            Model Recommendations
                        </h3>
                        <div class="bg-green-50 rounded-lg p-4">
                            <pre class="whitespace-pre-wrap text-sm">{{ formatModelSuggestions(analysisResult.results.model_suggestions) }}</pre>
                        </div>
                    </div>

                    <!-- Research Plan -->
                    <div v-if="analysisResult.workflow_type === 'research_planning' && analysisResult.results.research_plan">
                        <h3 class="font-semibold text-gray-800 mb-3">
                            <i class="fas fa-clipboard-list mr-2"></i>
                            Research Plan
                        </h3>
                        <div class="bg-purple-50 rounded-lg p-4">
                            <pre class="whitespace-pre-wrap text-sm">{{ formatResearchPlan(analysisResult.results.research_plan) }}</pre>
                        </div>
                    </div>

                    <!-- Quality Metrics -->
                    <div v-if="analysisResult.results.critique_results">
                        <h3 class="font-semibold text-gray-800 mb-3">
                            <i class="fas fa-star mr-2"></i>
                            Quality Assessment
                        </h3>
                        <div class="bg-blue-50 rounded-lg p-4">
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-blue-600">
                                        {{ (analysisResult.results.critique_results.total_score || 0).toFixed(1) }}
                                    </div>
                                    <div class="text-sm text-gray-600">Overall Score</div>
                                </div>
                            </div>
                            <pre class="whitespace-pre-wrap text-sm">{{ formatCritiqueResults(analysisResult.results.critique_results) }}</pre>
                        </div>
                    </div>
                </div>

                <!-- Raw JSON (Collapsible) -->
                <div class="mt-6">
                    <button
                        @click="showRawJson = !showRawJson"
                        class="flex items-center text-gray-600 hover:text-gray-800"
                    >
                        <i :class="showRawJson ? 'fas fa-chevron-down' : 'fas fa-chevron-right'" class="mr-2"></i>
                        Raw JSON Results
                    </button>
                    <div v-if="showRawJson" class="mt-2 bg-gray-100 rounded-lg p-4 overflow-auto">
                        <pre class="text-xs">{{ JSON.stringify(analysisResult.results, null, 2) }}</pre>
                    </div>
                </div>
            </div>

            <!-- Analysis History -->
            <div class="card-shadow bg-white rounded-lg p-6">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-2xl font-bold text-gray-800">
                        <i class="fas fa-history mr-2"></i>
                        Analysis History
                    </h2>
                    <button
                        @click="loadHistory"
                        class="px-4 py-2 text-purple-600 border border-purple-600 rounded-lg hover:bg-purple-50"
                    >
                        <i class="fas fa-refresh mr-1"></i>
                        Refresh
                    </button>
                </div>

                <div v-if="history.length === 0" class="text-center text-gray-500 py-8">
                    <i class="fas fa-inbox text-4xl mb-2"></i>
                    <p>No analysis history yet. Start by analyzing a research task!</p>
                </div>

                <div v-else class="space-y-3">
                    <div
                        v-for="item in history.slice().reverse()"
                        :key="item.id"
                        class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                        @click="loadAnalysis(item)"
                    >
                        <div class="flex items-center justify-between">
                            <div class="flex-1">
                                <div class="font-medium text-gray-800 truncate">{{ item.prompt }}</div>
                                <div class="flex items-center space-x-2 mt-1">
                                    <span :class="'workflow-badge ' + getWorkflowClass(item.response.workflow_type)">
                                        {{ item.response.workflow_type.replace('_', ' ') }}
                                    </span>
                                    <span class="text-sm text-gray-500">
                                        {{ formatTimestamp(item.timestamp) }}
                                    </span>
                                </div>
                            </div>
                            <i class="fas fa-chevron-right text-gray-400"></i>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer class="bg-gray-800 text-white py-8 mt-12">
            <div class="container mx-auto px-4 text-center">
                <p>&copy; 2025 ML Researcher LangGraph. Powered by LangGraph and FastAPI.</p>
                <div class="mt-2 space-x-4">
                    <span class="text-sm text-gray-400">
                        <i class="fas fa-code mr-1"></i>
                        Built with Vue.js, FastAPI, and LangGraph
                    </span>
                </div>
            </div>
        </footer>
    </div>

    <script>
        const { createApp } = Vue;

        createApp({
            data() {
                return {
                    prompt: '',
                    streamMode: false,
                    isAnalyzing: false,
                    analysisResult: null,
                    history: [],
                    healthStatus: null,
                    showRawJson: false
                }
            },
            async mounted() {
                await this.checkHealth();
                await this.loadHistory();
            },
            methods: {
                async checkHealth() {
                    try {
                        const response = await axios.get('/health');
                        this.healthStatus = response.data;
                    } catch (error) {
                        console.error('Health check failed:', error);
                    }
                },

                async analyzeTask() {
                    if (!this.prompt.trim() || this.isAnalyzing) return;

                    this.isAnalyzing = true;
                    this.analysisResult = null;

                    try {
                        const response = await axios.post('/analyze', {
                            prompt: this.prompt,
                            stream: this.streamMode
                        });

                        this.analysisResult = response.data;
                        await this.loadHistory(); // Refresh history
                    } catch (error) {
                        console.error('Analysis failed:', error);
                        alert('Analysis failed: ' + (error.response?.data?.detail || error.message));
                    } finally {
                        this.isAnalyzing = false;
                    }
                },

                async loadHistory() {
                    try {
                        const response = await axios.get('/history');
                        this.history = response.data.history;
                    } catch (error) {
                        console.error('Failed to load history:', error);
                    }
                },

                loadAnalysis(item) {
                    this.analysisResult = item.response;
                    this.prompt = item.prompt;
                    // Scroll to results
                    document.querySelector('.result-animation')?.scrollIntoView({ behavior: 'smooth' });
                },

                getWorkflowClass(workflowType) {
                    const classes = {
                        'model_suggestion': 'model-suggestion',
                        'research_planning': 'research-planning',
                        'direct_llm': 'direct-llm'
                    };
                    return classes[workflowType] || 'unknown';
                },

                getWorkflowIcon(workflowType) {
                    const icons = {
                        'model_suggestion': 'fas fa-robot',
                        'research_planning': 'fas fa-clipboard-list',
                        'direct_llm': 'fas fa-comments'
                    };
                    return icons[workflowType] || 'fas fa-question';
                },

                formatProcessingTime(seconds) {
                    return `${seconds.toFixed(2)}s`;
                },

                formatTimestamp(timestamp) {
                    return new Date(timestamp).toLocaleString();
                },

                formatModelSuggestions(suggestions) {
                    if (typeof suggestions === 'string') return suggestions;
                    return JSON.stringify(suggestions, null, 2);
                },

                formatResearchPlan(plan) {
                    if (typeof plan === 'string') return plan;
                    return JSON.stringify(plan, null, 2);
                },

                formatCritiqueResults(critique) {
                    if (typeof critique === 'string') return critique;
                    return JSON.stringify(critique, null, 2);
                }
            }
        }).mount('#app');
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
