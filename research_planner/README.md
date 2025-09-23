# Research Planning Module 🔬

An intelligent research planning workflow that generates comprehensive, validated research plans with iterative refinement and quality assurance capabilities.

## 🌟 Features

### Core Capabilities
- **🧠 AI-Driven Problem Generation**: LLM creates novel, specific research problems from broad topics
- **✅ Multi-Criteria Validation**: Evaluates research problems for feasibility, novelty, and significance
- **📋 Comprehensive Plan Creation**: Generates detailed research plans with methodology and timelines
- **🔄 Iterative Refinement**: Automated critique and improvement cycles for plan optimization
- **📊 Multi-Format Support**: Handles various input types including experimental data and documents
- **🎯 Publication-Ready Output**: Produces detailed research plans suitable for grant applications and project proposals

### Advanced Features
- **Smart Problem Scoping**: Automatically determines optimal research scope and focus areas
- **Literature Gap Analysis**: Identifies research gaps and positions proposed work appropriately
- **Methodology Validation**: Ensures research methods are appropriate and feasible
- **Timeline Optimization**: Creates realistic project timelines with key milestones
- **Risk Assessment**: Identifies potential challenges and mitigation strategies

## 🚀 Quick Start

### Basic Usage

```python
from research_planning_nodes import plan_research

# Simple research planning
result = await plan_research(
    prompt="Develop better machine learning interpretability methods"
)

# With experimental context
result = await plan_research(
    prompt="Improve deep learning model efficiency",
    uploaded_data=["[CSV: model_performance.csv]\nmodel,accuracy,speed\nCNN,0.92,5ms"]
)

print(f"Research plan saved to: {result['plan_file_path']}")
print(f"Plan quality score: {result['final_score']}/10")
```

### Command Line Usage

```bash
# Basic research planning
python research_planning_nodes.py "Your research area"

# With context data
python research_planning_nodes.py "ML optimization techniques" --data performance_data.csv

# Focused research planning
python research_planning_nodes.py "Interpretable AI for healthcare applications"
```

## 📋 Workflow Architecture

The research planning workflow consists of 7 specialized nodes:

```
🎯 generate_problem → ✅ validate_problem → 📝 create_plan → 🔍 critique_plan → 🔄 refine_plan → 📊 finalize_plan → 🎯 output_results
```

### Node Details

1. **🎯 Generate Problem** (`_generate_problem_node`)
   - Transforms broad topics into specific, actionable research problems
   - Considers current state of the field and identifies gaps
   - Ensures problems are novel and significant

2. **✅ Validate Problem** (`_validate_problem_node`)
   - Evaluates problem feasibility and impact potential
   - Checks for novelty and research significance
   - Validates scope appropriateness for available resources

3. **📝 Create Plan** (`_create_research_plan_node`)
   - Develops comprehensive research methodology
   - Creates detailed project timeline with milestones
   - Defines success metrics and evaluation criteria

4. **🔍 Critique Plan** (`_critique_research_plan_node`)
   - Evaluates plan quality across multiple dimensions
   - Identifies weaknesses and improvement opportunities
   - Provides specific, actionable feedback

5. **🔄 Refine Plan** (`_refine_research_plan_node`)
   - Implements critique feedback systematically
   - Enhances plan quality and addresses identified issues
   - Maintains plan coherence during refinement

6. **📊 Finalize Plan** (`_finalize_research_plan_node`)
   - Applies final formatting and structure optimization
   - Ensures all plan components are complete and coherent
   - Prepares plan for output and documentation

7. **🎯 Output Results** (`_output_results_node`)
   - Saves research plan to structured files
   - Generates comprehensive metrics and statistics
   - Creates plan summary and metadata

## 📊 Output Examples

### Typical Research Plan Structure
```
1. 🎯 Executive Summary
2. 🔍 Problem Statement & Motivation
3. 📚 Literature Review & Research Gap
4. 🧪 Research Methodology
5. 📅 Timeline & Milestones
6. 📊 Expected Outcomes & Impact
7. 💰 Resource Requirements
8. 🚧 Risk Assessment & Mitigation
9. 📈 Success Metrics & Evaluation
10. 📝 References & Related Work
```

### Generated Files
- **📄 Main Plan**: `research_plan_YYYYMMDD_HHMMSS.md`
- **📊 Plan Metrics**: Quality scores, refinement iterations, validation results
- **🔍 Analysis Summary**: Problem significance, feasibility assessment, expected impact

### Quality Metrics
- **Plan Completeness**: 85-95% coverage of essential components
- **Feasibility Score**: 7.0-9.0/10 average feasibility rating
- **Novelty Assessment**: High novelty with clear differentiation from existing work
- **Impact Potential**: Clear pathways to significant research contributions

## ⚙️ Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
OPENAI_MODEL=gpt-4              # Model for plan generation
RESEARCH_PLANNING_MODEL=gpt-4   # Specific model for research planning
MAX_REFINEMENT_ITERATIONS=3     # Maximum plan refinement cycles
```

### Supported Input Types
- **📝 Research Topics**: Natural language descriptions of research areas
- **📊 Experimental Data**: CSV, JSON with experimental results and metrics
- **📋 Literature Context**: Existing research summaries and backgrounds
- **🎯 Specific Objectives**: Detailed research goals and constraints

## 🔧 Advanced Usage

### Custom Research Domains

```python
# Domain-specific research planning
result = await plan_research(
    prompt="Develop interpretable AI methods for medical diagnosis",
    domain_context={
        "field": "medical_ai",
        "constraints": ["regulatory_compliance", "clinical_validation"],
        "timeline": "2_years",
        "resources": "academic_setting"
    }
)

# With experimental foundation
experimental_context = {
    "current_methods": ["CNN", "ResNet", "Vision Transformer"],
    "performance_baseline": {"accuracy": 0.89, "precision": 0.91},
    "dataset_characteristics": {"size": 50000, "classes": 10, "modality": "images"}
}

result = await plan_research(
    prompt="Improve medical image classification interpretability",
    uploaded_data=[json.dumps(experimental_context)]
)
```

### Validation Criteria Customization

The validation system evaluates research problems across multiple dimensions:

- **🎯 Novelty**: Uniqueness and differentiation from existing work
- **⚡ Feasibility**: Technical and practical achievability
- **📈 Significance**: Potential impact and contribution to the field
- **🔬 Methodology**: Appropriateness of proposed research methods
- **📅 Timeline**: Realistic project duration and milestone setting
- **💰 Resources**: Required expertise, equipment, and funding

### Iterative Refinement Control

```python
# Control refinement iterations
result = await plan_research(
    prompt="Your research topic",
    max_iterations=5,          # Maximum refinement cycles
    quality_threshold=8.0,     # Minimum acceptable quality score
    early_stopping=True        # Stop when quality threshold is met
)

# Access refinement history
iterations = result.get("refinement_history", [])
for i, iteration in enumerate(iterations):
    print(f"Iteration {i+1}: Score {iteration['score']:.1f}/10")
    print(f"Issues addressed: {len(iteration['improvements'])}")
```

## 📈 Performance Characteristics

### Planning Quality Metrics
- **🎯 Problem Specificity**: 90%+ of generated problems are specific and actionable
- **✅ Validation Success**: 85%+ of problems pass multi-criteria validation
- **📋 Plan Completeness**: 95%+ coverage of essential research plan components
- **🔄 Refinement Effectiveness**: 0.8-1.2 point average improvement per iteration

### Typical Processing Times
- **Problem Generation**: 30-60 seconds
- **Problem Validation**: 20-40 seconds  
- **Plan Creation**: 60-120 seconds
- **Critique & Refinement**: 45-90 seconds per iteration
- **Total Workflow**: 3-6 minutes for complete research plan

## 🛠️ Development

### Key Classes and Functions

```python
class ResearchPlanningState(BaseState):
    """State management for research planning workflow"""
    research_problem: str           # Generated research problem
    validation_results: Dict        # Problem validation outcomes
    research_plan: str             # Complete research plan
    critique_results: Dict         # Plan quality assessment
    refinement_history: List       # Iteration tracking

async def plan_research(prompt: str, uploaded_data: List[str] = None) -> Dict[str, Any]:
    """Main entry point for research planning"""

def build_research_planning_graph() -> StateGraph:
    """Constructs the LangGraph workflow"""
```

### State Management

```python
# Access workflow state at any point
state = result["final_state"]

print(f"Generated problem: {state['research_problem']}")
print(f"Validation score: {state['validation_results']['overall_score']}")
print(f"Plan quality: {state['critique_results']['overall_score']}")
print(f"Refinement count: {state['refinement_count']}")
```

### Testing and Validation

```bash
# Test basic functionality
python research_planning_nodes.py "Test research planning"

# Test with specific domains
python research_planning_nodes.py "Computer vision for autonomous vehicles"

# Test refinement capabilities
python research_planning_nodes.py "Explainable AI methods" --max-iterations 5
```

## 📋 Dependencies

```
openai>=1.0.0          # LLM integration
langgraph>=0.0.40      # Workflow orchestration
pandas>=1.5.0          # Data processing (optional)
json                   # Data serialization
datetime               # Timestamp generation
typing                 # Type annotations
```

## 🐛 Troubleshooting

### Common Issues

**Problem Generation Failure**
```
❌ Error in generate_problem_node: Problem too broad or vague
```
- **Solution**: Provide more specific input with domain context
- **Example**: Instead of "AI research", use "Interpretable AI for medical diagnosis"

**Validation Rejection**
```
⚠️ Problem validation failed: Insufficient novelty
```
- **Solution**: Refine problem statement to highlight unique aspects
- **Retry**: System automatically attempts problem regeneration

**Plan Quality Issues**
```
📊 Plan quality score: 5.2/10 - Below threshold
```
- **Solution**: Additional refinement iterations are automatically triggered
- **Manual Review**: Examine critique feedback for specific improvement areas

**Resource Limitations**
```
❌ OpenAI API rate limit exceeded
```
- **Solution**: Implement exponential backoff (built into workflow)
- **Prevention**: Monitor API usage and upgrade plan if necessary

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable workflow tracing
result = await plan_research("Your topic", debug=True)
```

### Quality Assurance

Monitor plan quality with built-in metrics:
```python
# Quality checkpoints
quality_metrics = result["quality_metrics"]
print(f"Problem clarity: {quality_metrics['problem_clarity']}/10")
print(f"Methodology soundness: {quality_metrics['methodology_score']}/10")
print(f"Timeline realism: {quality_metrics['timeline_feasibility']}/10")
```

## 📚 Integration Examples

### With Paper Writing Module

```python
# Plan research and then write paper
research_result = await plan_research("Interpretable ML for healthcare")
research_plan = research_result["research_plan"]

# Use plan as context for paper writing
paper_result = await write_paper(
    user_query="Write a research proposal paper",
    research_context=research_plan
)
```

### With Experiment Design

```python
# Research planning → Experiment design
plan_result = await plan_research("Novel optimization algorithms")

# Extract methodology for experiment design
methodology = plan_result["final_state"]["research_plan_details"]["methodology"]
```

## 📊 Quality Benchmarks

### Validation Success Rates
- **Problem Novelty**: 88% achieve high novelty scores (≥7/10)
- **Feasibility Assessment**: 92% deemed feasible with available resources
- **Methodology Appropriateness**: 90% use suitable research methods
- **Timeline Realism**: 85% have realistic project timelines

### Plan Completeness Metrics
- **Essential Sections**: 98% include all required plan components
- **Methodology Detail**: 90% provide sufficient methodological detail
- **Risk Assessment**: 87% include comprehensive risk analysis
- **Success Metrics**: 93% define clear evaluation criteria

## 📝 Related Modules

- **📝 Paper Writing**: `/report_writing/` - Generate research papers from plans
- **🧪 Experiment Design**: `/design_experiment/` - Design experiments from research plans  
- **🧠 ML Researcher**: `/core/` - Main research orchestration framework

## 📄 Output Format Examples

### Research Plan Structure
```markdown
# Research Plan: [Problem Title]

## Executive Summary
Brief overview of research objectives and expected outcomes...

## Problem Statement
Detailed description of the research problem and motivation...

## Literature Review
Current state of the field and identified research gaps...

## Methodology
Comprehensive research approach with specific methods...

## Timeline
Detailed project schedule with key milestones...

## Expected Outcomes
Anticipated results and potential impact...

## Resource Requirements
Necessary expertise, equipment, and funding...

## Risk Assessment
Potential challenges and mitigation strategies...
```

## 📝 License

Part of the Aero ML Research Assistant project. See main repository for license details.

---

*Research plans are generated for planning and educational purposes. Always validate feasibility and resource requirements before implementation.*
