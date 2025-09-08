# ML Researcher LangGraph - Workflow Documentation

## 🌟 System Architecture Overview

```mermaid
graph TB
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
    class F1,F2,F3,F4,F5,F6 features
```

## 🔀 Router & Workflow Decision Logic

The system intelligently routes user queries to the appropriate workflow:

```mermaid
graph TD
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
    class L,N validation
```

## 🧪 Research Planning Workflow (Iterative)

The research planning workflow uses an iterative validation approach:

```mermaid
graph TD
    A[🔬 Research Planning Start] --> B[🎯 Generate Problem Statement<br/>Iteration: 1]
    
    B --> C[🌐 Validate Problem<br/>Web Search + LLM Analysis]
    C --> D{🤔 Problem Assessment<br/>Solved vs Open}
    
    D -->|✅ Open/Novel<br/>Accept| E[📥 Collect Problem<br/>Add to Validated List]
    D -->|❌ Solved/Well-studied<br/>Reject| F[🔄 Generate New Problem<br/>Iteration: N+1]
    
    E --> G{📊 Collection Check<br/>Have Enough Problems?}
    G -->|❌ Need More<br/>< 3 Problems| F
    G -->|✅ Sufficient<br/>≥ 3 Problems| H[� Select Problem<br/>User Choice Interface]
    
    F --> I{⏰ Iteration Limit<br/>< 10 Attempts?}
    I -->|✅ Continue| B
    I -->|❌ Max Reached| H
    
    H --> J[📋 Create Research Plan<br/>Comprehensive Analysis]
    
    %% NEW: Critique and Refinement Loop
    J --> K[🔍 Critique Plan<br/>AI Quality Assessment]
    
    K --> L{📊 Quality Check<br/>Major Issues?}
    
    L -->|🎉 No Major Issues<br/>High Quality| M[✅ Finalize Plan<br/>Quality Approved]
    L -->|🔄 Has Major Issues<br/>Refinable| N[🔧 Refinement Context<br/>Add Critique Feedback]
    L -->|⚠️ Problem Issues<br/>Fundamental Problems| H
    L -->|🔄 Deep Issues<br/>Start Over| B
    
    N --> O{🔄 Refinement Check<br/>< 3 Attempts?}
    O -->|✅ Continue Refining| J
    O -->|❌ Max Refinements| P[✅ Accept Best Version<br/>Force Finalize]
    
    %% Validation Details
    C --> C1[📚 Literature Check]
    C --> C2[🌐 Web Search Results]  
    C --> C3[🎯 Gap Assessment]
    
    %% Problem Generation Details
    B --> B1[💭 Domain Exploration]
    B --> B2[🔍 Challenge Identification]
    B --> B3[❓ Question Formulation]
    
    %% Critique Details
    K --> K1[📊 Novelty Assessment]
    K --> K2[🔧 Feasibility Check]
    K --> K3[📝 Methodology Review]
    K --> K4[⏰ Timeline Validation]
    
    %% Research Plan Details
    J --> J1[📋 Problem Analysis]
    J --> J2[⏰ Phase Planning]
    J --> J3[💰 Resource Estimation]
    J --> J4[📊 Success Metrics]
    
    %% Styling
    classDef start fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef process fill:#45B7D1,stroke:#007BFF,stroke-width:2px
    classDef validation fill:#96CEB4,stroke:#28A745,stroke-width:2px
    classDef decision fill:#FECA57,stroke:#F39C12,stroke-width:2px
    classDef collection fill:#DDA0DD,stroke:#8A2BE2,stroke-width:2px
    classDef userAction fill:#FF6B6B,stroke:#DC143C,stroke-width:2px
    classDef critique fill:#9370DB,stroke:#663399,stroke-width:2px
    classDef refinement fill:#FFA07A,stroke:#FF4500,stroke-width:2px
    classDef output fill:#90EE90,stroke:#32CD32,stroke-width:3px
    classDef subprocess fill:#F0F8FF,stroke:#4682B4,stroke-width:1px
    classDef loop fill:#FFB6C1,stroke:#DC143C,stroke-width:2px
    
    class A start
    class B,F process
    class C validation
    class D,G,I,L,O decision
    class E collection
    class H userAction
    class J,M,P output
    class K critique
    class N refinement
    class B1,B2,B3,C1,C2,C3,J1,J2,J3,J4,K1,K2,K3,K4 subprocess
    class F,N loop
```

## 📊 Model Suggestion Workflow

For model recommendation queries:

```mermaid
graph TD
    A[🎯 Analyze Properties & Task<br/>Extract Requirements] --> B[� Generate Search Query<br/>Create arXiv Query]
    
    B --> C[� Search arXiv<br/>Find Relevant Papers]
    
    C --> D[� Suggest Models<br/>Generate Recommendations]
    
    D --> E[🔍 Critique Response<br/>AI Quality Assessment]
    
    E --> F{� Quality Check<br/>Revision Needed?}
    
    F -->|✅ Accept<br/>High Quality| G[✅ Finalize Suggestions<br/>Quality Approved]
    F -->|🔄 Revise<br/>Has Issues| H[🔧 Add Revision Context<br/>Include Critique Feedback]
    F -->|⏱️ Max Iterations<br/>4 Attempts Reached| I[✅ Accept Best Version<br/>Force Finalize]
    
    H --> J{🔄 Iteration Check<br/>< 4 Attempts?}
    J -->|✅ Continue| D
    J -->|❌ Max Reached| I
    
    %% Evidence Processing Details
    C --> C1[� Paper Processing]
    C --> C2[🔍 Semantic Search]
    C --> C3[� Evidence Extraction]
    
    %% Task Analysis Details  
    A --> A1[🎯 Requirement Analysis]
    A --> A2[📂 Category Detection]
    A --> A3[🔧 Constraint Identification]
    
    %% Model Suggestion Details
    D --> D1[🧠 Model Selection]
    D --> D2[� Justification Generation]
    D --> D3[⚙️ Implementation Guidance]
    
    %% Critique Details
    E --> E1[� Relevance Assessment]
    E --> E2[🔧 Technical Accuracy]
    E --> E3[📚 Evidence Utilization]
    E --> E4[� Completeness Check]
    
    %% Revision Context Details
    H --> H1[📋 Previous Response]
    H --> H2[⚠️ Critique Issues]
    H --> H3[💡 Improvement Suggestions]
    H --> H4[✅ Cumulative Memory]
    
    %% Quality Tracking
    E -.->|Track Issues| K[📈 Cumulative Issues<br/>Fixed/Recurring/Persistent]
    H -.->|Max 4 Iterations| L[⏹️ Iteration Limit]
    
    %% Final Outputs
    G --> M[📄 Final Recommendations<br/>With Quality Score]
    I --> N[📄 Best Attempt<br/>With Revision History]
    
    %% Styling
    classDef analysis fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef search fill:#96CEB4,stroke:#28A745,stroke-width:2px
    classDef generation fill:#4ECDC4,stroke:#17A2B8,stroke-width:2px
    classDef critique fill:#9370DB,stroke:#663399,stroke-width:2px
    classDef decision fill:#FECA57,stroke:#F39C12,stroke-width:2px
    classDef revision fill:#FFA07A,stroke:#FF4500,stroke-width:2px
    classDef success fill:#90EE90,stroke:#32CD32,stroke-width:2px
    classDef subprocess fill:#F0F8FF,stroke:#4682B4,stroke-width:1px
    classDef tracking fill:#FFE4E1,stroke:#CD5C5C,stroke-width:1px
    
    class A analysis
    class B,C search
    class D generation
    class E critique
    class F,J decision
    class H revision
    class G,I,M,N success
    class A1,A2,A3,C1,C2,C3,D1,D2,D3,E1,E2,E3,E4,H1,H2,H3,H4 subprocess
    class K,L tracking
```

## 📊 State Management & Data Flow

```mermaid
graph TD
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
    class E5,E6,E7,E8,E9,E10 structure
```

## ⚙️ Conditional Logic & Decision Points

```mermaid
graph TD
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
    class A,E,F,H,O,AA,BB,CC,DD,N process
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
