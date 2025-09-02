# ML Research Assistant 🔬

A LiteLLM-powered tool for generating comprehensive research plans for machine learning topics and projects using Gemini and other AI models.

## Features

- **LiteLLM Proxy Integration**: Uses your custom LiteLLM proxy with Gemini 2.5 Flash
- **Comprehensive Research Plans**: Generates all-in-one research plans covering theory, implementation, applications, and deployment
- **Simplified Interface**: One research type that includes theoretical foundations, practical implementation, real-world applications, and system architecture
- **Interactive Mode**: Conversational interface for exploring multiple research topics
- **Structured Output**: Saves plans in both Markdown and JSON formats
- **Cost Tracking**: Tracks API usage and estimated costs
- **Flexible Configuration**: Supports custom models and base URLs

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure LiteLLM Proxy

1. Your LiteLLM proxy should be running at `https://agents.aetherraid.dev`
2. Add your API key using one of these methods:

**Option A: Update env.example file (Recommended)**
Edit the `env.example` file and replace the placeholder with your actual API key:
```
OPENAI_API_KEY='your-actual-api-key-here'
BASE_URL='https://agents.aetherraid.dev'
DEFAULT_MODEL='gemini/gemini-2.5-flash'
```

**Option B: Environment variable**
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
$env:BASE_URL = "https://agents.aetherraid.dev"
$env:DEFAULT_MODEL = "gemini/gemini-2.5-flash"
```

**Option C: Create .env file**
Create a `.env` file in the project directory:
```
OPENAI_API_KEY=your-api-key-here
BASE_URL=https://agents.aetherraid.dev
DEFAULT_MODEL=gemini/gemini-2.5-flash
```

The tool will automatically look for configuration in this order:
1. Environment variables
2. `.env` file
3. `env.example` file
4. Command line arguments

## Usage

### Command Line Mode

Generate a research plan for a specific topic:
```powershell
python ml_research_assistant.py --topic "Transformer models for time series forecasting"
```

Generate an implementation-focused plan:
```powershell
python ml_research_assistant.py --project "Build a recommendation system using graph neural networks"
```

Specify project type:
```powershell
python ml_research_assistant.py --topic "Federated learning privacy" --type theoretical
```

Save to specific file:
```powershell
python ml_research_assistant.py --topic "Computer vision for medical imaging" --output my_research_plan.md
```

### Interactive Mode

For exploring multiple topics or having a conversation:
```powershell
python ml_research_assistant.py --interactive
```

### Command Line Options

- `--topic, -t`: ML research topic to generate a plan for
- `--project, -p`: ML project description for implementation-focused plans
- `--type`: Plan type (research, implementation, theoretical, application)
- `--interactive, -i`: Run in interactive mode
- `--output, -o`: Output filename for the research plan
- `--api-key`: OpenAI API key (alternative to environment variable)

## Example Research Topics

- "Transformer models for computer vision"
- "Federated learning privacy preservation techniques"
- "Graph neural networks for drug discovery"
- "Self-supervised learning for time series analysis"
- "Reinforcement learning for autonomous vehicles"
- "Large language models for code generation"
- "Multimodal learning for robotics"

## Output Structure

The generated research plans include:

1. **Executive Summary** - Overview and objectives
2. **Background & Literature Review** - Current state and key papers
3. **Research Methodology** - Approach and experimental design
4. **Technical Requirements** - Skills, tools, and resources needed
5. **Implementation Plan** - Phased timeline with milestones
6. **Potential Challenges** - Risk assessment and mitigation
7. **Success Metrics** - Evaluation criteria
8. **Resources & References** - Papers, tutorials, and tools

## Files Generated

- `research_plan_[topic]_[timestamp].md` - Human-readable research plan
- `research_plan_[topic]_[timestamp].json` - Machine-readable data

## Cost Estimation

The tool tracks API usage and provides cost estimates. Typical costs:
- Simple research plan: $0.10 - $0.30
- Comprehensive plan: $0.30 - $0.80
- Interactive session: $1.00 - $3.00

## Examples

### Basic Usage
```powershell
python ml_research_assistant.py --topic "Neural architecture search"
```

### Interactive Session
```powershell
python ml_research_assistant.py -i
# Follow the prompts to explore multiple research topics
```

### Implementation Project
```powershell
python ml_research_assistant.py --project "Build a real-time object detection system for manufacturing quality control"
```

## Troubleshooting

### API Key Issues
- Ensure your OpenAI API key is valid and has sufficient credits
- Check that the environment variable is set correctly
- Verify you have access to GPT-4 API

### Installation Issues
```powershell
# Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Network Issues
- Check your internet connection
- Verify you can access OpenAI's API endpoints
- Consider using a VPN if there are regional restrictions

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
