#!/usr/bin/env python3
"""
Mermaid Visualization Generator for ML Researcher LangGraph
This script generates mermaid diagrams for all workflows without modifying the main file.
Note: This is a standalone generator - no need to import the main ML Researcher file.
"""

import os

def create_diagrams_directory():
    """Create diagrams directory if it doesn't exist."""
    if not os.path.exists('diagrams'):
        os.makedirs('diagrams')
        print("📁 Created diagrams/ directory")

def generate_router_mermaid():
    """Generate mermaid diagram for the router workflow."""
    mermaid_code = """graph TD
    A[👤 User Input<br/>Research Query] --> B[🤖 Router Agent<br/>Analyze Query Intent]
    
    B --> C{🎯 Decision Logic<br/>Query Type Analysis}
    
    C -->|Model/Architecture Query| D[📊 Model Suggestion<br/>Workflow]
    C -->|Research Planning Query| E[🔬 Research Planning<br/>Workflow]
    
    D --> F[📚 arXiv Search]
    F --> G[📄 Process PDFs]
    G --> H[🧠 Domain Analysis]
    H --> I[💡 Generate Suggestions]
    I --> J[📋 Model Recommendations]
    
    E --> K[🎯 Generate Problem]
    K --> L[✅ Validate Problem]
    L --> M{🤔 Accept Problem?}
    M -->|✅ Accept| N[📥 Collect Problem]
    M -->|❌ Reject| K
    N --> O{📊 Need More?<br/>< 3 Problems}
    O -->|Yes| K
    O -->|No| P[📋 Create Research Plan]
    
    %% Styling
    classDef userInput fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef router fill:#FF6B6B,stroke:#DC143C,stroke-width:2px
    classDef decision fill:#FECA57,stroke:#F39C12,stroke-width:2px
    classDef modelWorkflow fill:#4ECDC4,stroke:#17A2B8,stroke-width:2px
    classDef researchWorkflow fill:#45B7D1,stroke:#007BFF,stroke-width:2px
    classDef output fill:#DDA0DD,stroke:#8A2BE2,stroke-width:2px
    classDef validation fill:#96CEB4,stroke:#28A745,stroke-width:2px
    
    class A userInput
    class B router
    class C,M,O decision
    class D,F,G,H,I modelWorkflow
    class E,K,L,N,P researchWorkflow
    class J,P output
    class L,N validation"""
    
    return mermaid_code

def generate_model_suggestion_mermaid():
    """Generate detailed mermaid diagram for model suggestion workflow."""
    mermaid_code = """graph TD
    A[🔍 Model Suggestion Start] --> B[📚 Search arXiv Papers<br/>Query Processing]
    
    B --> C[📄 Download & Process PDFs<br/>Extract Content]
    C --> D[🧹 Clean & Structure Data<br/>Remove Artifacts]
    D --> E[🧠 Domain Analysis<br/>Current State Assessment]
    E --> F[💡 Generate Model Suggestions<br/>Architecture Recommendations]
    F --> G[📊 Format Output<br/>Structured Results]
    
    %% Sub-processes
    B --> B1[🔑 Extract Keywords]
    B --> B2[🔍 arXiv API Query]
    B --> B3[📋 Filter Results]
    
    C --> C1[⬇️ Download PDFs]
    C --> C2[📖 Extract Text]
    C --> C3[🧹 Clean Content]
    
    E --> E1[📈 Analyze Trends]
    E --> E2[🎯 Identify Challenges]
    E --> E3[💪 Assess Strengths]
    
    F --> F1[🏗️ Architecture Options]
    F --> F2[⚖️ Performance Comparisons]
    F --> F3[🛠️ Implementation Guidance]
    
    G --> H[✅ Final Recommendations<br/>Complete Analysis]
    
    %% Styling
    classDef process fill:#4ECDC4,stroke:#17A2B8,stroke-width:2px
    classDef subprocess fill:#A8E6CF,stroke:#28A745,stroke-width:2px
    classDef output fill:#DDA0DD,stroke:#8A2BE2,stroke-width:2px
    
    class A,B,C,D,E,F,G process
    class B1,B2,B3,C1,C2,C3,E1,E2,E3,F1,F2,F3 subprocess
    class H output"""
    
    return mermaid_code

def generate_research_planning_mermaid():
    """Generate detailed mermaid diagram for research planning workflow."""
    mermaid_code = """graph TD
    A[🔬 Research Planning Start] --> B[🎯 Generate Problem Statement<br/>Iteration: 1]
    
    B --> C[🔍 Validate Problem<br/>LLM Analysis]
    C --> D{🤔 Problem Assessment<br/>Solved vs Open}
    
    D -->|✅ Open/Novel<br/>Accept| E[📥 Collect Problem<br/>Add to Validated List]
    D -->|❌ Solved/Well-studied<br/>Reject| F[🔄 Generate New Problem<br/>Iteration: N+1]
    
    E --> G{📊 Collection Check<br/>Have Enough Problems?}
    G -->|❌ Need More<br/>< 3 Problems| F
    G -->|✅ Sufficient<br/>≥ 3 Problems| H[📋 Create Research Plan<br/>Comprehensive Analysis]
    
    F --> I{⏰ Iteration Limit<br/>< 10 Attempts?}
    I -->|✅ Continue| B
    I -->|❌ Max Reached| H
    
    H --> J[✅ Final Research Plan<br/>Validated Problems + Strategy]
    
    %% Validation Details
    C --> C1[📚 Literature Check]
    C --> C2[🧠 Solution Analysis]  
    C --> C3[🎯 Gap Assessment]
    
    %% Problem Generation Details
    B --> B1[💭 Domain Exploration]
    B --> B2[🔍 Challenge Identification]
    B --> B3[❓ Question Formulation]
    
    %% Research Plan Details
    H --> H1[📋 Problem Prioritization]
    H --> H2[⏰ Timeline Creation]
    H --> H3[💰 Resource Planning]
    H --> H4[📊 Success Metrics]
    
    %% Styling
    classDef start fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef process fill:#45B7D1,stroke:#007BFF,stroke-width:2px
    classDef validation fill:#96CEB4,stroke:#28A745,stroke-width:2px
    classDef decision fill:#FECA57,stroke:#F39C12,stroke-width:2px
    classDef collection fill:#DDA0DD,stroke:#8A2BE2,stroke-width:2px
    classDef output fill:#90EE90,stroke:#32CD32,stroke-width:3px
    classDef subprocess fill:#F0F8FF,stroke:#4682B4,stroke-width:1px
    classDef loop fill:#FFB6C1,stroke:#DC143C,stroke-width:2px
    
    class A start
    class B,F process
    class C validation
    class D,G,I decision
    class E collection
    class H,J output
    class B1,B2,B3,C1,C2,C3,H1,H2,H3,H4 subprocess
    class F loop"""
    
    return mermaid_code

def generate_complete_system_mermaid():
    """Generate high-level overview of the complete system."""
    mermaid_code = """graph TB
    subgraph "🌟 ML Researcher LangGraph System"
        A[👤 User Query<br/>Research Question] --> B[🤖 Intelligent Router<br/>Semantic Analysis]
        
        B --> C{🎯 Workflow Decision<br/>Intent Classification}
        
        C -->|🔍 Model Query| D[📊 Model Suggestion Pipeline]
        C -->|🔬 Research Query| E[🧪 Research Planning Pipeline]
    end
    
    subgraph "📊 Model Suggestion Workflow"
        D --> D1[📚 Literature Search<br/>arXiv Papers]
        D1 --> D2[🧠 Domain Analysis<br/>Current State]
        D2 --> D3[💡 Model Recommendations<br/>Architecture Guidance]
        D3 --> D4[📋 Structured Output<br/>Implementation Guide]
    end
    
    subgraph "🧪 Research Planning Workflow (Iterative)"
        E --> E1[🎯 Problem Generation<br/>Novel Research Questions]
        E1 --> E2[✅ Problem Validation<br/>Novelty Assessment]
        E2 --> E3{🤔 Accept Problem?<br/>Open vs Solved}
        E3 -->|✅ Accept| E4[📥 Problem Collection<br/>Build Validated Set]
        E3 -->|❌ Reject| E1
        E4 --> E5{📊 Sufficient Problems?<br/>Target: 3+ Problems}
        E5 -->|❌ Need More| E1
        E5 -->|✅ Ready| E6[📋 Research Plan Creation<br/>Comprehensive Strategy]
    end
    
    subgraph "🎯 System Features"
        F1[🔄 Intelligent Routing<br/>Confidence Scoring]
        F2[📚 Real-time Literature<br/>Analysis & Search]
        F3[🔍 Iterative Validation<br/>Problem Novelty Check]
        F4[📊 Structured Outputs<br/>JSON Results]
        F5[⚡ Error Handling<br/>Graceful Recovery]
        F6[🧠 LLM Integration<br/>Advanced Analysis]
    end
    
    D4 --> G[📤 Final Results<br/>Model Recommendations]
    E6 --> H[📤 Final Results<br/>Research Plan]
    
    %% Styling
    classDef userInput fill:#FFE4B5,stroke:#FF8C00,stroke-width:3px
    classDef router fill:#FF6B6B,stroke:#DC143C,stroke-width:2px
    classDef decision fill:#FECA57,stroke:#F39C12,stroke-width:2px
    classDef modelFlow fill:#4ECDC4,stroke:#17A2B8,stroke-width:2px
    classDef researchFlow fill:#45B7D1,stroke:#007BFF,stroke-width:2px
    classDef output fill:#90EE90,stroke:#32CD32,stroke-width:3px
    classDef features fill:#E6E6FA,stroke:#9370DB,stroke-width:1px
    classDef validation fill:#96CEB4,stroke:#28A745,stroke-width:2px
    
    class A userInput
    class B router
    class C,E3,E5 decision
    class D,D1,D2,D3,D4 modelFlow
    class E,E1,E2,E4,E6 researchFlow
    class E2,E4 validation
    class G,H output
    class F1,F2,F3,F4,F5,F6 features"""
    
    return mermaid_code

def generate_state_flow_mermaid():
    """Generate state flow and data structure diagram."""
    mermaid_code = """graph TD
    subgraph "📊 State Management System"
        A[🔄 RouterState<br/>Workflow Decision] --> B{🎯 Route Decision}
        B -->|Model Query| C[📊 ModelSuggestionState<br/>Literature Processing]
        B -->|Research Query| D[🧪 ResearchPlanningState<br/>Iterative Validation]
    end
    
    subgraph "📊 ModelSuggestionState Flow"
        C --> C1[📚 arXiv Results<br/>Papers & Content]
        C1 --> C2[🧠 Domain Analysis<br/>Research Landscape]
        C2 --> C3[💡 Model Suggestions<br/>Recommendations]
        C3 --> C4[📋 Final Output<br/>Structured Results]
    end
    
    subgraph "🧪 ResearchPlanningState Flow"
        D --> D1[🎯 Generated Problems<br/>All Attempts]
        D1 --> D2[✅ Current Problem<br/>Being Validated]
        D2 --> D3[🔍 Validation Results<br/>Accept/Reject Decision]
        D3 --> D4[📥 Validated Problems<br/>Accepted Collection]
        D4 --> D5[📊 Iteration Counter<br/>Progress Tracking]
        D5 --> D6[📋 Research Plan<br/>Final Strategy]
    end
    
    subgraph "🔍 Problem Validation Structure"
        E[🎯 Problem Statement] --> E1[📝 Description<br/>Detailed Context]
        E1 --> E2[❓ Research Question<br/>Specific Query]
        E2 --> E3[🏷️ Keywords<br/>Domain Tags]
        E3 --> E4[✅ Validation Result<br/>Status Assessment]
        
        E4 --> E5[📊 Status Classification<br/>solved|open|partial]
        E4 --> E6[🎯 Confidence Score<br/>0.0 - 1.0]
        E4 --> E7[💭 Reasoning<br/>Analysis Details]
        E4 --> E8[🔍 Existing Solutions<br/>Known Approaches]
        E4 --> E9[🎯 Research Gaps<br/>Open Opportunities]
        E4 --> E10[✅ Recommendation<br/>accept|reject]
    end
    
    %% Styling
    classDef state fill:#E6F3FF,stroke:#0066CC,stroke-width:2px
    classDef modelState fill:#E6F7FF,stroke:#00A0B0,stroke-width:2px
    classDef researchState fill:#E6F0FF,stroke:#4169E1,stroke-width:2px
    classDef validation fill:#F0FFF0,stroke:#228B22,stroke-width:2px
    classDef decision fill:#FFF8DC,stroke:#DAA520,stroke-width:2px
    classDef structure fill:#FFF0F5,stroke:#C71585,stroke-width:2px
    
    class A,B state
    class C,C1,C2,C3,C4 modelState
    class D,D1,D2,D3,D4,D5,D6 researchState
    class E,E1,E2,E3,E4 validation
    class B decision
    class E5,E6,E7,E8,E9,E10 structure"""
    
    return mermaid_code

def generate_conditional_logic_mermaid():
    """Generate conditional logic and decision points diagram."""
    mermaid_code = """graph TD
    subgraph "🔀 Router Logic Decision Tree"
        A[📝 User Query Analysis] --> B{🎯 Intent Detection<br/>Semantic Analysis}
        B -->|Model Keywords| C[🔍 MODEL_SUGGESTION<br/>Triggers Detected]
        B -->|Research Keywords| D[🔬 RESEARCH_PLANNING<br/>Triggers Detected]
        
        C --> C1[📊 Confidence: High<br/>Architecture Query]
        D --> D1[📊 Confidence: High<br/>Problem Discovery]
    end
    
    subgraph "🔄 Research Planning Conditional Logic"
        E[🎯 Problem Generation] --> F[✅ Problem Validation]
        F --> G{🤔 Validation Decision<br/>_check_completion()}
        
        G -->|recommendation: accept| H[📥 Collect Problem<br/>Add to Validated List]
        G -->|recommendation: reject| I[🔄 Continue Generation<br/>Loop Back]
        
        H --> J{📊 Collection Decision<br/>_should_continue_generating()}
        
        J -->|validated_problems < 3| K[🔄 Generate More<br/>Need Additional Problems]
        J -->|validated_problems >= 3| L[✅ Finalize Plan<br/>Sufficient Problems]
        J -->|iteration_count >= 10| M[⏰ Force Finalize<br/>Max Iterations Reached]
        
        I --> E
        K --> E
        L --> N[📋 Create Research Plan]
        M --> N
    end
    
    subgraph "🎯 Validation Assessment Logic"
        O[🔍 Problem Analysis] --> P{📊 Status Classification}
        
        P -->|Status: solved| Q[❌ REJECT<br/>Well Established]
        P -->|Status: well_studied| R[❌ REJECT<br/>Extensively Researched]
        P -->|Status: partially_solved| S{🎯 Confidence Check<br/>> 0.7?}
        P -->|Status: open| T[✅ ACCEPT<br/>Novel Opportunity]
        
        S -->|Yes| U[✅ ACCEPT<br/>Significant Gaps]
        S -->|No| V[❌ REJECT<br/>Low Confidence]
    end
    
    subgraph "⚙️ System Control Flow"
        W[🚀 System Start] --> X[🤖 Initialize Router]
        X --> Y[📝 Process User Query]
        Y --> Z{🎯 Route Decision}
        Z -->|Model| AA[📊 Execute Model Workflow]
        Z -->|Research| BB[🔬 Execute Research Workflow]
        AA --> CC[📤 Return Model Results]
        BB --> DD[📤 Return Research Results]
        CC --> EE[💾 Save & Display Results]
        DD --> EE
    end
    
    %% Styling
    classDef router fill:#FFE4E1,stroke:#DC143C,stroke-width:2px
    classDef research fill:#E0F6FF,stroke:#0080FF,stroke-width:2px
    classDef validation fill:#F0FFF0,stroke:#32CD32,stroke-width:2px
    classDef control fill:#F5F5DC,stroke:#8B4513,stroke-width:2px
    classDef decision fill:#FFFACD,stroke:#DAA520,stroke-width:2px
    classDef accept fill:#98FB98,stroke:#228B22,stroke-width:2px
    classDef reject fill:#FFB6C1,stroke:#DC143C,stroke-width:2px
    classDef process fill:#E6E6FA,stroke:#9370DB,stroke-width:2px
    
    class A,B,C,D,C1,D1 router
    class E,F,H,I,J,K,L,M,N research
    class O,P,S validation
    class W,X,Y,Z,AA,BB,CC,DD,EE control
    class B,G,J,P,S,Z decision
    class T,U accept
    class Q,R,V reject
    class A,E,F,H,O,AA,BB,CC,DD,N process"""
    
    return mermaid_code

def save_mermaid_files():
    """Generate and save all mermaid diagram files."""
    create_diagrams_directory()
    
    diagrams = {
        'router_workflow.mmd': generate_router_mermaid(),
        'model_suggestion_workflow.mmd': generate_model_suggestion_mermaid(),
        'research_planning_workflow.mmd': generate_research_planning_mermaid(),
        'complete_system_overview.mmd': generate_complete_system_mermaid(),
        'state_flow_diagram.mmd': generate_state_flow_mermaid(),
        'conditional_logic_diagram.mmd': generate_conditional_logic_mermaid()
    }
    
    print("🎨 Generating Mermaid Visualization Files...")
    print("=" * 50)
    
    for filename, content in diagrams.items():
        filepath = os.path.join('diagrams', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Generated: {filepath}")
    
    print("\n📊 Generated 6 mermaid diagram files!")
    return list(diagrams.keys())

def generate_html_viewer():
    """Generate standalone HTML file for viewing all diagrams."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Researcher LangGraph - Workflow Visualizations</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            border-left: 4px solid #007bff;
            padding-left: 15px;
        }
        .diagram {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .description {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }
        .mermaid {
            text-align: center;
        }
        .navigation {
            background: #333;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .navigation a {
            color: #fff;
            text-decoration: none;
            margin: 0 15px;
            padding: 5px 10px;
            border-radius: 3px;
            transition: background 0.3s;
        }
        .navigation a:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 ML Researcher LangGraph - Workflow Visualizations</h1>
        
        <div class="navigation">
            <strong>Quick Navigation:</strong>
            <a href="#overview">System Overview</a>
            <a href="#router">Router Logic</a>
            <a href="#model">Model Workflow</a>
            <a href="#research">Research Workflow</a>
            <a href="#states">State Flow</a>
            <a href="#logic">Conditional Logic</a>
        </div>

        <div id="overview" class="diagram">
            <h2>🌟 Complete System Overview</h2>
            <div class="description">
                <strong>Purpose:</strong> High-level view of the entire ML Researcher system showing both workflows and their relationships.
            </div>
            <div class="mermaid">
""" + generate_complete_system_mermaid() + """
            </div>
        </div>

        <div id="router" class="diagram">
            <h2>🔀 Router & Workflow Decision</h2>
            <div class="description">
                <strong>Purpose:</strong> Shows how user queries are analyzed and routed to the appropriate workflow.
            </div>
            <div class="mermaid">
""" + generate_router_mermaid() + """
            </div>
        </div>

        <div id="model" class="diagram">
            <h2>📊 Model Suggestion Workflow</h2>
            <div class="description">
                <strong>Purpose:</strong> Detailed view of the model recommendation pipeline including literature search and analysis.
            </div>
            <div class="mermaid">
""" + generate_model_suggestion_mermaid() + """
            </div>
        </div>

        <div id="research" class="diagram">
            <h2>🧪 Research Planning Workflow (Iterative)</h2>
            <div class="description">
                <strong>Purpose:</strong> Shows the iterative problem generation and validation loop for research planning.
            </div>
            <div class="mermaid">
""" + generate_research_planning_mermaid() + """
            </div>
        </div>

        <div id="states" class="diagram">
            <h2>📊 State Flow & Data Structures</h2>
            <div class="description">
                <strong>Purpose:</strong> Illustrates how data flows through different state objects and validation structures.
            </div>
            <div class="mermaid">
""" + generate_state_flow_mermaid() + """
            </div>
        </div>

        <div id="logic" class="diagram">
            <h2>⚙️ Conditional Logic & Decision Points</h2>
            <div class="description">
                <strong>Purpose:</strong> Detailed view of all decision points and conditional logic in the system.
            </div>
            <div class="mermaid">
""" + generate_conditional_logic_mermaid() + """
            </div>
        </div>

        <div style="margin-top: 50px; padding: 20px; background: #f0f0f0; border-radius: 5px;">
            <h3>📝 How to Use These Diagrams:</h3>
            <ul>
                <li><strong>Copy to Mermaid.live:</strong> Copy any diagram code and paste into <a href="https://mermaid.live" target="_blank">mermaid.live</a></li>
                <li><strong>GitHub Integration:</strong> Use in README.md files with ```mermaid code blocks</li>
                <li><strong>Documentation:</strong> Embed in project documentation and presentations</li>
                <li><strong>Export:</strong> Save as PNG/SVG from mermaid.live for presentations</li>
            </ul>
        </div>
    </div>

    <script>
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
    </script>
</body>
</html>"""
    
    filepath = os.path.join('diagrams', 'workflow_viewer.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Generated: {filepath}")
    return filepath

def generate_readme_integration():
    """Generate README sections with embedded mermaid diagrams."""
    readme_content = """# ML Researcher LangGraph - Workflow Documentation

## 🌟 System Architecture Overview

```mermaid
""" + generate_complete_system_mermaid() + """
```

## 🔀 Router & Workflow Decision Logic

The system intelligently routes user queries to the appropriate workflow:

```mermaid
""" + generate_router_mermaid() + """
```

## 🧪 Research Planning Workflow (Iterative)

The research planning workflow uses an iterative validation approach:

```mermaid
""" + generate_research_planning_mermaid() + """
```

## 📊 Model Suggestion Workflow

For model recommendation queries:

```mermaid
""" + generate_model_suggestion_mermaid() + """
```

## 📊 State Management & Data Flow

```mermaid
""" + generate_state_flow_mermaid() + """
```

## ⚙️ Conditional Logic & Decision Points

```mermaid
""" + generate_conditional_logic_mermaid() + """
```

## 🎯 Key Features

- **🔀 Intelligent Routing**: Semantic analysis determines appropriate workflow
- **🧪 Iterative Validation**: Problems are validated for novelty before inclusion
- **📚 Real-time Literature Search**: Current arXiv papers inform recommendations
- **📊 Structured Outputs**: JSON results with confidence scoring
- **⚡ Error Handling**: Graceful recovery and fallback mechanisms

## 📝 Usage Examples

### Model Recommendation Query
```
"What model should I use for time series forecasting?"
→ Routes to Model Suggestion Workflow
→ Searches literature, analyzes domain, provides recommendations
```

### Research Planning Query  
```
"What are open problems in federated learning?"
→ Routes to Research Planning Workflow  
→ Generates problems, validates novelty, creates research plan
```
"""
    
    filepath = os.path.join('diagrams', 'README_integration.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Generated: {filepath}")
    return filepath

def create_usage_instructions():
    """Create detailed usage instructions."""
    instructions = """# 🎨 How to Use the Mermaid Visualizations

## 📁 Generated Files

Your `diagrams/` folder now contains:

1. **router_workflow.mmd** - Router and workflow decision logic
2. **model_suggestion_workflow.mmd** - Model recommendation pipeline  
3. **research_planning_workflow.mmd** - Iterative research planning
4. **complete_system_overview.mmd** - High-level system architecture
5. **state_flow_diagram.mmd** - Data structures and state management
6. **conditional_logic_diagram.mmd** - Decision points and logic
7. **workflow_viewer.html** - Standalone HTML viewer
8. **README_integration.md** - Ready-to-use README sections

## 🌐 Viewing Options

### Option 1: Mermaid Live Editor (Recommended)
1. Go to https://mermaid.live
2. Copy content from any `.mmd` file
3. Paste into the editor
4. Instantly see the visual diagram
5. Export as PNG/SVG if needed

### Option 2: Standalone HTML Viewer
1. Open `diagrams/workflow_viewer.html` in your browser
2. View all diagrams in one page
3. Use navigation to jump between sections

### Option 3: GitHub Integration
1. Copy content from `README_integration.md`
2. Add to your project's README.md
3. GitHub automatically renders mermaid diagrams

### Option 4: VS Code (with Extension)
1. Install "Mermaid Markdown Syntax Highlighting" extension
2. Open any `.mmd` file in VS Code
3. Use preview pane to see rendered diagram

## 🎯 Recommended Workflow

1. **Quick viewing**: Use mermaid.live for immediate visualization
2. **Documentation**: Copy sections from README_integration.md
3. **Presentations**: Export PNG/SVG from mermaid.live
4. **Development**: Use VS Code extension for editing

## 📝 Example: Using with Mermaid Live

1. Open https://mermaid.live
2. Copy this content from `complete_system_overview.mmd`:

```
graph TB
    subgraph "🌟 ML Researcher LangGraph System"
        A[👤 User Query<br/>Research Question] --> B[🤖 Intelligent Router<br/>Semantic Analysis]
        // ... rest of the diagram
    end
```

3. Paste into mermaid.live
4. See beautiful rendered diagram!
5. Export as needed

## 🔧 Customization

All `.mmd` files are text-based and fully editable:
- Modify colors by changing `classDef` statements
- Add/remove nodes by editing the graph structure  
- Update labels by changing text in brackets
- Adjust styling with CSS-like syntax

## 💡 Tips

- **Performance**: Large diagrams may take a moment to render
- **Mobile**: Some diagrams are better viewed on desktop
- **Export**: Use SVG for scalable graphics, PNG for presentations
- **Sharing**: Mermaid.live provides shareable URLs for diagrams
"""
    
    filepath = os.path.join('diagrams', 'USAGE_INSTRUCTIONS.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"✅ Generated: {filepath}")
    return filepath

def main():
    """Generate all mermaid visualizations."""
    print("🎨 ML Researcher LangGraph - Mermaid Visualization Generator")
    print("=" * 60)
    print("Generating comprehensive workflow diagrams...")
    print("Zero changes to your main ml_researcher_langgraph.py file!\n")
    
    try:
        # Generate all mermaid files
        mermaid_files = save_mermaid_files()
        
        # Generate HTML viewer
        html_file = generate_html_viewer()
        
        # Generate README integration
        readme_file = generate_readme_integration()
        
        # Generate usage instructions
        instructions_file = create_usage_instructions()
        
        print("\n🎉 SUCCESS! All visualization files generated!")
        print("=" * 60)
        print("\n📁 Files created in diagrams/ folder:")
        print("├── 📊 Mermaid Diagram Files (.mmd)")
        for filename in mermaid_files:
            print(f"│   ├── {filename}")
        print("├── 🌐 workflow_viewer.html (Standalone viewer)")
        print("├── 📝 README_integration.md (Copy to your README)")
        print("└── 📋 USAGE_INSTRUCTIONS.md (How to use)")
        
        print("\n🚀 QUICK START:")
        print("1. Open https://mermaid.live")
        print("2. Copy content from any .mmd file")
        print("3. Paste and see your workflow diagram!")
        print("\nOR")
        print("1. Open diagrams/workflow_viewer.html in your browser")
        print("2. View all diagrams in one place!")
        
        print("\n📚 For GitHub README:")
        print("Copy sections from README_integration.md to add")
        print("beautiful diagrams directly to your project documentation!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
