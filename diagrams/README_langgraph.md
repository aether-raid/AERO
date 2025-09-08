# ML Research Tool - LangGraph Version

This is a LangGraph-compatible version of the Machine Learning Research Tool that provides a structured, graph-based workflow for ML research task analysis.

## What's New in the LangGraph Version

### LangGraph Integration
- **State Management**: Uses LangGraph's `StateGraph` for managing workflow state
- **Node-based Architecture**: Each step is implemented as a separate node in the graph
- **Message Passing**: Leverages LangGraph's message system for communication between nodes
- **Workflow Orchestration**: Automatic flow control through the research pipeline

### Graph Structure
The workflow is organized into 7 sequential nodes:

1. **extract_properties** - Analyzes the research task to identify ML categories
2. **decompose_task** - Breaks down the task into key properties and characteristics  
3. **generate_search_query** - Creates optimized arXiv search queries
4. **search_arxiv** - Searches and processes relevant research papers
5. **suggest_models** - Recommends suitable ML models/architectures
6. **identify_problems** - Identifies open research problems and opportunities
7. **create_research_plan** - Generates comprehensive research plan

### Key Benefits

1. **Better State Management**: LangGraph provides robust state management across workflow steps
2. **Modular Design**: Each step is isolated and can be modified independently
3. **Error Handling**: Improved error tracking and recovery
4. **Extensibility**: Easy to add new nodes or modify the workflow
5. **Message History**: Maintains conversation context throughout the process

## Installation

1. Install the LangGraph-specific requirements:
```bash
pip install -r requirements_langgraph.txt
```

2. Configure your API key in `env.example`:
```
OPENAI_API_KEY=your-api-key-here
BASE_URL=https://agents.aetherraid.dev
DEFAULT_MODEL=gemini/gemini-2.5-flash
```

## Usage

### Interactive Mode
```bash
python ml_researcher_langgraph.py
```

### Command Line Mode
```bash
python ml_researcher_langgraph.py "Your research task here"
```

## Workflow Details

### State Object
The workflow maintains a `ResearchState` object containing:
- `messages`: LangGraph message history
- `original_prompt`: User's research task
- `detected_categories`: ML categories identified
- `detailed_analysis`: Task decomposition results
- `arxiv_search_query`: Generated search query
- `arxiv_results`: Found papers and analysis
- `model_suggestions`: Recommended models
- `open_problems`: Identified research gaps
- `research_plan`: Comprehensive research roadmap
- `current_step`: Current workflow step
- `errors`: Error tracking

### Node Functions
Each node function:
- Takes the current state as input
- Performs its specific analysis task
- Updates the state with results
- Returns the updated state
- Handles errors gracefully

### Async Support
The LangGraph version maintains full async support for:
- Concurrent paper processing
- Parallel LLM calls where appropriate
- Non-blocking I/O operations

## Differences from Original Version

### Architecture Changes
- **Original**: Monolithic class with sequential method calls
- **LangGraph**: Graph-based workflow with state management

### State Management
- **Original**: Instance variables and return values
- **LangGraph**: Centralized state object passed between nodes

### Error Handling
- **Original**: Try/catch in individual methods
- **LangGraph**: Centralized error tracking in state

### Extensibility
- **Original**: Requires modifying the main class
- **LangGraph**: Add new nodes or modify graph structure

## Migration Guide

If you're migrating from the original version:

1. The core functionality remains the same
2. Results are saved with `_langgraph` suffix to avoid conflicts
3. All API configurations work identically
4. Output format is preserved for compatibility

## Dependencies

Key new dependencies:
- `langgraph>=0.0.40` - Graph-based workflow framework
- `langchain-core>=0.1.0` - Core LangChain components for message handling

All other dependencies remain the same as the original version.

## Output

Results are saved as `ml_research_analysis_langgraph_TIMESTAMP.json` with the same structure as the original version, plus additional workflow metadata.

## Troubleshooting

### Common Issues

1. **LangGraph Import Error**
   ```bash
   pip install langgraph langchain-core
   ```

2. **State Type Issues**
   - Ensure all state modifications return the updated state object
   - Check that TypedDict annotations match the actual state structure

3. **Graph Compilation Errors**
   - Verify all nodes are properly connected
   - Check that entry point and edges are correctly defined

### Debug Mode

To debug the workflow, you can add print statements in individual nodes or inspect the state object at each step.

## Performance

The LangGraph version maintains similar performance to the original while providing:
- Better memory management through state isolation
- Improved error recovery
- More predictable execution flow

## Future Enhancements

The graph-based architecture enables future improvements like:
- Conditional branching based on analysis results
- Parallel execution of independent nodes
- Dynamic workflow modification
- Integration with other LangGraph tools
