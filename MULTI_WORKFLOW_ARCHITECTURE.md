# ML Research Tool - Multi-Workflow LangGraph Architecture

## 🎯 Overview

The new multi-workflow architecture transforms the ML Research Tool into an intelligent routing system with specialized workflows for different types of research tasks. Instead of a single linear pipeline, the system now uses:

1. **🤖 Router Agent** - Analyzes user prompts and decides which workflow to execute
2. **📊 Model Suggestion Workflow** - For architecture and model recommendations
3. **📋 Research Planning Workflow** - For identifying open problems and research planning

## 🏗️ Architecture Diagram

```
User Prompt
     ↓
┌─────────────┐
│ Router Agent│ ← Analyzes intent and routes request
└─────────────┘
     ↓
   Decision
     ↓
┌─────────────┬─────────────┐
│   Workflow 1 │  Workflow 2  │
│             │             │
│ MODEL       │ RESEARCH    │
│ SUGGESTION  │ PLANNING    │
│             │             │
│ 1. Extract  │ 1. Domain   │
│    Properties│    Analysis │
│ 2. Decompose│ 2. Identify │
│    Task     │    Problems │
│ 3. Generate │ 3. Create   │
│    Query    │    Plan     │
│ 4. Search   │             │
│    ArXiv    │             │
│ 5. Suggest  │             │
│    Models   │             │
└─────────────┴─────────────┘
     ↓             ↓
   Results       Results
```

## 🚦 Router Agent Details

### Purpose
Intelligently analyzes user prompts to determine the most appropriate workflow.

### Decision Logic
The router examines the user's request for specific indicators:

**Model Suggestion Triggers:**
- "What model should I use for X?"
- "Recommend architectures for Y task"
- "Best approaches for Z problem"
- "Which algorithm is suitable for..."
- Model comparison and selection requests
- Technical implementation guidance

**Research Planning Triggers:**
- "What are open problems in X domain?"
- "Research opportunities in Y field"
- "Future directions for Z"
- "Create a research plan for..."
- "What should I research in X area?"
- Academic research planning requests

### Router State
```python
class RouterState(TypedDict):
    messages: List[BaseMessage]
    original_prompt: str
    routing_decision: str           # "model_suggestion" or "research_planning"
    routing_confidence: float       # 0.0-1.0 confidence score
    routing_reasoning: str          # Explanation of decision
    errors: List[str]
```

## 📊 Workflow 1: Model Suggestion Pipeline

### Purpose
Provides comprehensive model and architecture recommendations based on task analysis and current research.

### Steps
1. **Extract Properties** - Identifies ML categories that apply to the task
2. **Decompose Task** - Breaks down technical requirements
3. **Generate Search Query** - Creates optimized arXiv search terms
4. **Search ArXiv** - Finds and processes relevant research papers
5. **Suggest Models** - Recommends specific architectures with justification

### State Structure
```python
class ModelSuggestionState(BaseState):
    detected_categories: List[Dict[str, Any]]    # ML categories with confidence
    detailed_analysis: Dict[str, Any]            # Technical breakdown
    arxiv_search_query: str                      # Formatted search string
    arxiv_results: Dict[str, Any]                # Papers and rankings
    model_suggestions: Dict[str, Any]            # Final recommendations
```

### Key Features
- **21 ML Categories** analyzed with confidence scoring
- **Concurrent paper processing** (up to 5 papers simultaneously)
- **LLM-based relevance scoring** for found papers
- **Evidence-based recommendations** linking suggestions to research

### Example Use Cases
- "What's the best architecture for real-time object detection?"
- "Recommend models for time series forecasting with limited data"
- "Which transformer variant should I use for document classification?"

## 📋 Workflow 2: Research Planning Pipeline

### Purpose
Identifies open research problems and creates comprehensive academic research plans.

### Steps
1. **Analyze Domain** - Comprehensive analysis of the research field
2. **Identify Problems** - Finds specific open problems and gaps
3. **Create Research Plan** - Generates actionable 24-month research roadmap

### State Structure
```python
class ResearchPlanningState(BaseState):
    domain_analysis: Dict[str, Any]              # Field overview and trends
    open_problems: Dict[str, Any]                # Identified research gaps
    research_plan: Dict[str, Any]                # Comprehensive roadmap
```

### Domain Analysis Components
- **Current State** - Recent breakthroughs and limitations
- **Technical Landscape** - Methods, datasets, metrics
- **Research Trends** - Hot topics and emerging areas
- **Knowledge Gaps** - Unsolved problems
- **Future Outlook** - Promising directions

### Research Plan Structure
- **Phase 1: Foundation** (Months 1-3) - Literature review
- **Phase 2: Problem Formulation** (Months 4-6) - Research questions
- **Phase 3: Core Research** (Months 7-18) - Primary research activities
- **Phase 4: Validation & Dissemination** (Months 19-24) - Publication

### Example Use Cases
- "What are the open problems in federated learning?"
- "Create a research plan for explainable AI in healthcare"
- "What should I research in computer vision for robotics?"

## 🔄 Data Flow Comparison

### Old Single Workflow
```
User Input → Properties → Task Analysis → ArXiv Search → Models → Problems → Plan → Results
```

### New Multi-Workflow
```
User Input → Router → [Workflow 1: Models] OR [Workflow 2: Research Plan] → Results
```

## 🆚 Workflow Comparison

| Aspect | Model Suggestion | Research Planning |
|--------|------------------|-------------------|
| **Focus** | Practical implementation | Academic research |
| **Output** | Model recommendations | Research roadmap |
| **Duration** | Quick analysis | Comprehensive planning |
| **Depth** | Technical specifics | Strategic overview |
| **Target User** | Practitioners, Engineers | Researchers, PhD students |

## 💡 Benefits of Multi-Workflow Architecture

### 1. **Specialized Processing**
- Each workflow is optimized for its specific purpose
- More relevant and focused results
- Faster execution for simple model selection queries

### 2. **Intelligent Routing**
- Automatic detection of user intent
- No need for users to specify workflow type
- Confidence scoring for routing decisions

### 3. **Modular Design**
- Easy to modify or extend individual workflows
- Can add new workflows without affecting existing ones
- Better error isolation and handling

### 4. **Improved User Experience**
- More relevant results based on actual intent
- Clearer output structure
- Workflow-specific file naming and organization

### 5. **Scalability**
- Can process different types of requests more efficiently
- Workflows can be optimized independently
- Future workflows can be added easily

## 🔧 Technical Implementation

### Graph Structure
Each workflow is implemented as a separate LangGraph StateGraph:

```python
# Router Graph
router_graph = StateGraph(RouterState)
router_graph.add_node("route_request", route_request_node)

# Model Suggestion Graph  
model_graph = StateGraph(ModelSuggestionState)
model_graph.add_node("extract_properties", extract_properties_node)
model_graph.add_node("decompose_task", decompose_task_node)
# ... etc

# Research Planning Graph
research_graph = StateGraph(ResearchPlanningState)
research_graph.add_node("analyze_domain", analyze_domain_node)
research_graph.add_node("identify_problems", identify_problems_node)
# ... etc
```

### Execution Flow
```python
async def analyze_research_task(self, prompt: str):
    # Step 1: Route the request
    router_result = await self.router_graph.ainvoke(router_state)
    
    # Step 2: Execute appropriate workflow
    if router_result["routing_decision"] == "model_suggestion":
        result = await self.model_suggestion_graph.ainvoke(model_state)
    else:
        result = await self.research_planning_graph.ainvoke(research_state)
    
    return result
```

## 📁 File Organization

Results are now saved with workflow-specific naming:
- `ml_research_analysis_model_suggestion_langgraph_TIMESTAMP.json`
- `ml_research_analysis_research_planning_langgraph_TIMESTAMP.json`

## 🧪 Testing

The new architecture includes comprehensive tests for:
- Router decision accuracy
- Model suggestion workflow completeness
- Research planning workflow execution
- Graph compilation and structure validation

Run tests with:
```bash
python test_langgraph.py
```

## 🚀 Usage Examples

### Model Suggestion Query
```
Input: "What's the best neural network for sentiment analysis of social media posts?"

Router Decision: MODEL_SUGGESTION (confidence: 0.95)
Output: Detailed model recommendations with specific architectures, 
        training considerations, and implementation guidance
```

### Research Planning Query
```
Input: "What are the current open problems in natural language processing?"

Router Decision: RESEARCH_PLANNING (confidence: 0.92)
Output: Domain analysis, specific research gaps, and 24-month research plan
```

## 🔮 Future Enhancements

The multi-workflow architecture enables easy addition of new specialized workflows:

- **📊 Dataset Workflow** - For dataset recommendation and analysis
- **🔍 Evaluation Workflow** - For metric selection and evaluation strategies
- **🛠️ Implementation Workflow** - For code generation and deployment guidance
- **📚 Literature Review Workflow** - For systematic literature analysis

## 🏁 Conclusion

The multi-workflow architecture transforms the ML Research Tool from a single-purpose analyzer into an intelligent research assistant that adapts to user needs. By automatically routing requests to specialized workflows, the system provides more relevant, focused, and actionable results while maintaining the comprehensive analysis capabilities of the original tool.

This design represents a significant evolution toward more sophisticated AI research assistance, enabling both practical implementation guidance and strategic research planning within a single unified system.
