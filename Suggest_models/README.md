# Suggest_models: Model Recommendation Module

The model suggestion workflow—exposed in code as `aero.model_researcher`—provides intelligent recommendations for machine learning research tasks. It analyzes problem characteristics, searches relevant literature, and suggests optimal models with detailed justifications. This README covers both the canonical package in `src/aero/model_researcher` and the mirrored convenience package in this folder.

## 🏗️ Architecture

### Core Components
- **`model_suggestion_nodes.py`**: Main workflow implementation with LangGraph orchestration.
- **`__init__.py`**: Clean API interface exporting `suggest_models`, `stream_model_suggestions`, and the low-level workflow runners.

### State Management
Uses `ModelSuggestionState` to track:
- User prompts and extracted categories
- arXiv search results and paper analysis
- Iterative refinement cycles
- Final model recommendations

## ✨ Key Features

- **Task Analysis**: Automatically extracts ML categories and decomposes complex tasks
- **Literature Integration**: Searches arXiv with semantic ranking and relevance filtering
- **Quality Assurance**: LLM-based critique with up to 4 iterative refinements
- **Comprehensive Output**: Model suggestions with justifications, implementation details, and citations
- **Error Handling**: Graceful degradation with detailed error reporting

## 🚀 Usage

### 1. Environment Setup

1. Install dependencies (the project is managed with `uv`; `pip` works too):
   ```powershell
   uv sync
   ```
2. Create a `.env` file at the project root with at least:
   ```dotenv
   OPENAI_API_KEY=sk-...
   BASE_URL=https://api.openai.com/v1        # or your compatible endpoint
   DEFAULT_MODEL=gpt-4o-mini                 # model name used for reasoning
   ```

   Optional keys (Tavily, etc.) will be used automatically when available.

### 2. Non-Streaming Helper (`suggest_models`)

The easiest way to consume the workflow is through the re-exported helper:

```python
import asyncio
from aero.model_researcher import suggest_models

async def main():
  response = await suggest_models(
    "I need help with object detection in autonomous vehicles"
  )
  print(response["model_suggestions"]["model_suggestions"])

asyncio.run(main())
```

Pass `uploaded_data=[...]` when you want to attach additional textual context. The helper returns the final workflow state once all refinement passes complete.

### 3. Streaming Interfaces

There are two supported patterns when you want incremental updates:

#### A. Flip the Streaming Flag on `suggest_models`

```python
import asyncio
from aero.model_researcher import suggest_models

async def main():
  stream = await suggest_models(
    "Design a streaming anomaly detection pipeline for industrial IoT sensors",
    streaming=True,
  )

  async for update in stream:
    status = update.get("status") or update.get("current_step")
    if status:
      print(f"Step: {status}")

asyncio.run(main())
```

#### B. Use the Dedicated `stream_model_suggestions` Helper

```python
import asyncio
from aero.model_researcher import stream_model_suggestions

async def main():
  async for update in stream_model_suggestions(
    "Optimise transformers for low-resource languages"
  ):
    if "model_suggestions" in update:
      print(update["model_suggestions"]["model_suggestions"][:200])

asyncio.run(main())
```

Both streaming variants yield structured dictionaries that mirror the workflow state. Expect updates for task analysis, arXiv search, paper validation, model drafting, and critique phases.

### 4. Command-Line Smoke Test (Optional)

A ready-made script demonstrates the streaming UX end-to-end:

```powershell
python test_files/test_stream_model_suggestions.py
```

The script loads `.env`, calls `stream_model_suggestions`, prints intermediate statuses, and shows a formatted preview of the final recommendations. Use it when validating new deployments or retrofitting configuration changes.

## 🔄 Workflow Flow

1. **Task Analysis**: Extract ML categories and task characteristics
2. **Literature Search**: Query arXiv with semantic filtering
3. **Paper Analysis**: Extract relevant methodologies and model information
4. **Initial Suggestions**: Generate model recommendations based on analysis
5. **Quality Validation**: LLM critique and refinement (up to 4 iterations)
6. **Final Output**: Comprehensive model suggestions with justifications

## 📋 Output Formats

### JSON Structure
```json
{
  "model_suggestions": {
    "model_suggestions": "# Model Recommendations\n\n## Primary Recommendation...",
    "categories": ["Computer Vision", "Object Detection"],
    "search_iteration": 1,
    "suggestion_iteration": 2
  },
  "arxiv_results": {...},
  "workflow_successful": true
}
```

### Markdown Output
- Detailed model recommendations with pros/cons
- Implementation considerations
- Literature references
- Performance expectations

## ⚠️ Error Handling

- **API Failures**: Automatic retry with exponential backoff
- **Invalid Prompts**: Clear error messages with suggestions
- **Literature Gaps**: Fallback to general best practices
- **Rate Limits**: Queue management and delay handling

## 📦 Dependencies

- `openai`: LLM API integration
- `ArxivPaperProcessor`: Custom arXiv processing utilities
- `langgraph`: Workflow orchestration
- `asyncio`: Asynchronous processing

## 🛠️ Development Guide

### Adding New Models
Extend `model_suggestion_nodes.py` with additional model categories and validation logic.

### Testing
```bash
# Unit tests for individual nodes
python -m pytest tests/test_model_suggestions.py

# Integration tests
python test_model_workflow.py
```

### Extending the Workflow
- Add new analysis nodes in the LangGraph structure
- Implement custom validation criteria
- Integrate additional literature sources

## 🔮 Future Enhancements

- Multi-modal model recommendations
- Performance benchmarking integration
- Automated hyperparameter suggestions
- Real-time model performance tracking

---

**Part of the AERO Framework** | [Main README](../README.md) | [Experiment Suggestions](../Suggest_additional_experiments/README.md)</content>
<parameter name="filePath">c:\Users\Jacobs laptop\Aero--1\Suggest_models\README.md