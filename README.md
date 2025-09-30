# AERO: Automated Exploration, Research & Orchestration – A Framework for Machine Learning Research
> [Jacob Wong](https://www.linkedin.com/in/jacob-wong-160898274/), [Ethan Lau](https://www.linkedin.com/in/ethan-lau-yu-cheng/) and [Charmaine Chua](https://www.linkedin.com/in/charmaine-chua-3a9668274/)

[![PyPI Latest Release](https://img.shields.io/pypi/v/aeroml.svg?logo=python&logoColor=white&color=blue)](https://pypi.org/project/aeroml/)
[![GitHub Release Date](https://img.shields.io/github/release-date/aether-raid/aero?logo=github&label=latest%20release&color=blue)](https://github.com/aether-raid/aero/releases/latest)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aether-raid/aero/python-publish.yml?label=PyPI%20Publish&color=blue)


<!-- <p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p> -->

Modern machine learning research involves many interdependent yet complex steps, from identifying suitable models for a problem to designing reproducible experiments and ultimately communicating results. Researchers often need to consume a large set of rapidly growing literature, populated by various new and emerging models, and translate these to concrete experiments with their research tasks.

This library presents **AERO**, a flexible toolkit for researchers to move faster from ideas to insight and capable of reducing friction in this process, while still ensuring researchers have full control on the research direction.

AERO includes a set of modular, LLM-driven workflows that can be used independently or in combination:
- **Model Researcher**: decomposes ML tasks into properties and recommends suitable models.
- **Research Planner**: builds structured research plans from problem statements.
- **Experiment Designer**: translates plans into experimental setups and even generates code.
- **Experimentalist**: analyzes experimental data and suggests possible next steps.
- **Report Writer**: drafts reports based on results and context.

These workflows are powered by [LangGraph](https://github.com/langchain-ai/langgraph) for graph-based orchestration of LLM nodes, and many utilize the [arXiv API](https://arxiv.org/help/api/index) for paper search and research.

## Table of Contents

1. [Installation and Set-Up](#%EF%B8%8F-installation-and-set-up)
   * [Installing from PyPI](#installing-from-pypi)
   * [Installing from Source](#installing-from-source)
   * [Contributing](#contributing)
   * [Configuring Environment Variables](#configuring-environment-variables)
2. [Getting Started](#-getting-started)
   * [Model Researcher](#model-researcher-)
   * [Research Planner](#research-planner-)
   * [Experiment Designer](#experiment-designer-%EF%B8%8F)
   * [Experimentalist](#experimentalist-)
   * [Report Writer](#report-writer-)
3.  [References](#references)
4.  [About the Team](#about-the-team)

## 🛠️ Installation and Set-Up

### Installing from PyPI

Yes, we have published our framework on PyPI at `aeroml`! To install the `aero` library and all its dependencies, the easiest method would be to use `pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following command in a terminal or Command Prompt / Powershell:

```bash
$ pip install aeroml
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the following command too:

```bash
$ python -m pip install aeroml
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

### Installing from Source

Git is needed to install this repository from source. This is not completely necessary as you can also install the zip file for this repository and store it on a local drive manually. To install Git, follow [this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt:

```bash
$ git clone https://github.com/aether-raid/aero.git
```

This stores a copy in the folder `aero`. You can then navigate into it using `cd aero`. Then, you can run the following:

```bash
$ pip install .
```

This should install `aero` to your local Python instance.

### Contributing

If you are contributing, please clone this repository:

```bash
$ git clone https://github.com/aether-raid/aero.git
```

Thereafter, use `uv` to sync dependencies as follows:

```bash
$ uv sync
```

This will initiate a `.venv/` directory in the repository root directory, which can then be used as the Python Environment for development purposes. Please follow the [`uv` documentation](https://docs.astral.sh/uv/) for detailed steps on how to use `uv` for development.


### Configuring Environment Variables

Copy the provided `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your own API keys and settings:

```
OPENAI_API_KEY='YOUR_OPENAI_KEY'
BASE_URL='YOUR_BASE_URL'
DEFAULT_MODEL='gemini/gemini-2.5-flash'
GOOGLE_API_KEY='YOUR_GOOGLE_KEY'
CX='YOUR_CUSTOM_SEARCH_CX'
TAVILY_API_KEY='YOUR_TAVILY_API_KEY'
```

**Do not** commit your real `.env` file to version control. The `.env.example` file is safe to share and shows users what variables they need.

---

## 🚀 Getting Started

### Model Researcher 🤖

Given a problem statement, this system analyzes the problem characteristics, searches for relevant literature, and recommends the most suitable machine learning models. The final output includes model suggestions with detailed justifications, implementation considerations, and literature references.

#### General Usage

You can import and use the workflow in your own Python scripts:

```python
from aero.model_researcher import suggest_models

# Non-streaming
result = await suggest_models(
   prompt="Classify chest X-rays",
   streaming=False,
)
print(result["model_suggestions"])

# Streaming
async for update in await suggest_models(
   prompt="Classify chest X-rays",
   streaming=True,
):
   handle_stream(update)
```

#### Overall Workflow

1. **Task Analysis**: Extracts research properties and decomposes the task into ML categories (classification, regression, generation, etc.) using predefined ML research categories.
2. **Literature Search**: Generates optimized arXiv search queries and retrieves relevant research papers using semantic search and relevance ranking.
3. **Paper Validation**: Filters and validates retrieved papers for methodological relevance and quality.
4. **Model Suggestion**: Analyzes papers and task characteristics to recommend suitable ML models with detailed justifications.
5. **Critique & Refinement**: LLM-based quality assessment with iterative improvement based on critique feedback (up to 4 iterations).
6. **Final Recommendations**: Produces comprehensive model suggestions with implementation details, performance expectations, and literature citations.

![Model Researcher Workflow](diagrams/model_researcher_flow.png)

### Research Planner 📋

Given a research domain or broad topic, this system generates novel research problems, validates their significance and feasibility through web search, and creates comprehensive research plans with detailed methodologies, timelines, and resource requirements.

#### General Usage

```python
from aero.research_planner import plan_research

# Non-streaming
result = await plan_research(
    prompt="Develop efficient transformer architectures for real-time NLP",
    streaming=False,
)
print(result["research_plan"])

# Streaming
async for update in await plan_research(
    prompt="Investigate novel approaches to few-shot learning",
    streaming=True,
):
    handle_stream(update)
```

#### Overall Workflow

1. **Client Initialization**: Sets up OpenAI and Tavily clients for AI reasoning and web validation.
2. **Problem Generation**: Creates specific, novel research problems from broad domain descriptions using AI analysis.
3. **Problem Validation**: Validates research problems through comprehensive web search to ensure novelty and significance.
4. **Feedback Processing**: Processes rejection feedback and iteratively refines problem statements for quality.
5. **Research Plan Creation**: Develops detailed research plans including methodology, phases, timelines, and resource requirements.
6. **Plan Critique & Refinement**: AI-driven quality assessment with iterative improvement based on critique feedback.
7. **Plan Finalization**: Produces publication-ready research plans with proper structure, citations, and implementation details.

![Research Planner Workflow](diagrams/research_planner_flow.png)

### Experiment Designer ⚗️

Given a research plan, the system extracts key information and retrieves supporting literature to generate experiment ideas and designs. The final output is a detailed experimental design accompanied by executable Python code.

#### General Usage

You can import and use the workflow in your own Python scripts:

```python
 from design_experiment import run_experiment_designer # Full Workflow 

# Regular Usage
 result = run_experiment_designer(user_input)
 print(result["design"])
 print(result["code"])

# Streaming Usage (yields status updates, final output is a dict):
async for update in await run_experiment_designer(user_input, stream=True):
   print(update)
```

#### Overall Workflow

1. Input Processing: Extracts goals, hypotheses, experiment ideas (if provided), and other relevant details from a research plan.
2. Literature Retrieval System: Uses a Hybrid-RAG (Retrieval-Augmented Generation) approach to search and retrieve supporting literature (arXiv API).
3. Idea Generation: Employs [AB-MCTS by Sakana AI](https://github.com/SakanaAI/treequest) to generate promising experiment ideas (when no experiment idea is provided).
4. Design Refinement: Refines experiment ideas into structured experiment designs that include:
   - Datasets
   - Methodologies and implementation steps
   - References
   - Additional supporting details
5. Scoring and Refinement System: Evaluates and refines experiment designs based on key criterions to ensure quality, completeness, and relevance.
6. Code Generation: Produces minimal executable Python code after syntax validation and import checks 

![Experiment Designer Workflow](diagrams/experiment_designer_flow.png)

### Experimentalist 🔬
Given experimental results and research context, the system analyzes findings, determines research direction, searches for supporting literature, and generates comprehensive experiment suggestions. The final output includes detailed experimental designs with literature grounding and iterative validation.

#### General Usage

You can import and use the workflow in your own Python scripts:

```python
from aero.experimentalist import experiment_suggestions

# Non-streaming
result = await experiment_suggestions(
   prompt="I completed CNN experiments for image classification",
   experimental_results={"model_performance": {"accuracy": 0.87}},
)
print(result["experiment_suggestions"])

# Streaming
async for update in await experiment_suggestions(
   prompt="I completed CNN experiments for image classification",
   file_path="data/experiments.xlsx",
   streaming=True,
):
   handle_stream(update)

# Convenience Helper (non-streaming)
from aero.experimentalist import suggest_experiments_nostream

result = await suggest_experiments_nostream(
   prompt="Analyze these results and suggest follow-up experiments",
   experimental_results={"accuracy": 0.89},
)
```

#### Overall Workflow

1. **Findings Analysis**: Analyzes experimental results and research context to understand current state and opportunities.
2. **Analysis Validation**: Ensures the research analysis is comprehensive and well-structured.
3. **Research Direction**: Determines optimal research direction based on analysis and key questions.
4. **Direction Validation**: Validates research direction for feasibility and alignment with goals.
5. **Literature Search**: Generates optimized arXiv queries and retrieves relevant experimental papers.
6. **Paper Validation**: Filters and validates retrieved papers for methodological relevance.
7. **Methodology Distillation**: Extracts key experimental approaches and techniques from literature.
8. **Experiment Generation**: Creates comprehensive experiment suggestions grounded in literature.
9. **Quality Validation**: LLM-based validation with iterative improvement (up to 5 iterations).
10. **Final Suggestions**: Produces validated experiment designs with detailed methodologies, expected outcomes, and success metrics.

![Experimentalist Workflow](diagrams/experimentalist_flow.png)

### Report Writer 📑

Given research topics and experimental data, this system generates complete academic papers with integrated citations, proper formatting, and iterative quality refinement. It combines AI-driven content generation with web-sourced citations for publication-ready outputs.

#### General Usage

```python
from aero.report_writer import write_paper

# Basic usage
result = await write_paper(
    user_query="Write a paper about machine learning fundamentals",
    experimental_data={"accuracy": 0.95, "f1_score": 0.92},
    streaming=False,
)
print(result["formatted_paper"])

# With uploaded files
result = await write_paper(
    user_query="Analyze my research data and write a paper",
    file_paths=["./data/results.csv", "./reports/analysis.docx"],
    target_venue="NeurIPS",
    streaming=False,
)

# Streaming mode
async for update in await write_paper(
    user_query="Write about transformer efficiency techniques",
    experimental_data={"latency_reduction": 0.3},
    streaming=True,
):
    handle_stream(update)
```

#### Overall Workflow

1. **Results Analysis**: Analyzes experimental data and research context to extract key findings and insights.
2. **Paper Setup**: Generates comprehensive paper structure including sections, target audience, and formatting configuration.
3. **Source Discovery**: Performs intelligent web searches using Tavily to find relevant citations and supporting literature.
4. **Content Generation**: Creates complete paper content with proper academic writing, integrated citations, and structured sections.
5. **Quality Critique**: AI-driven assessment of paper quality including coherence, depth, citation usage, and academic rigor.
6. **Iterative Refinement**: Refines content based on critique feedback to improve quality and academic standards.
7. **Paper Finalization**: Produces publication-ready papers with proper formatting, reference lists, and quality metrics.

![Report Writer Workflow](diagrams/report_writer_flow.png)

## References

[1] Inoue, Y., Misaki, K., Imajuku, Y., Kuroki, S., Nakamura, T., & Akiba, T. (2025). Wider or deeper? scaling llm inference-time compute with adaptive branching tree search. _arXiv preprint arXiv:2503.04412_.

[2] Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The ai scientist: Towards fully automated open-ended scientific discovery. _arXiv preprint arXiv:2408.06292_.

[3] Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., ... & Ha, D. (2025). The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. _arXiv preprint arXiv:2504.08066_.

[4] Wehr, G., Rideaux, R., Fox, A. J., Lightfoot, D. R., Tangen, J., Mattingley, J. B., & Ehrhardt, S. E. (2025). Virtuous Machines: Towards Artificial General Science. _arXiv preprint arXiv:2508.13421_.

[5] Gottweis, J., Weng, W. H., Daryin, A., Tu, T., Palepu, A., Sirkovic, P., ... & Natarajan, V. (2025). Towards an AI co-scientist. _arXiv preprint arXiv:2502.18864_.

[6] Schmidgall, S., Su, Y., Wang, Z., Sun, X., Wu, J., Yu, X., ... & Barsoum, E. (2025). Agent laboratory: Using llm agents as research assistants. _arXiv preprint arXiv:2501.04227._

## About the Team

AERO was developed by a team of AI Solutions Interns at [AETHER](https://github.com/aether-raid), the experimentations wing of [RAiD](https://rsaf-agile-inno-digital.defence.gov.sg/) focused on rapid prototyping and harnessing of technologies. This prototype was one of many developed for AETHER's internal Agentic AI Hackathon. The team includes:

- [Jacob Wong](https://www.linkedin.com/in/jacob-wong-160898274/)
- [Ethan Lau](https://www.linkedin.com/in/ethan-lau-yu-cheng/)
- [Charmaine Chua](https://www.linkedin.com/in/charmaine-chua-3a9668274/)


<p align="center">
<a href="https://star-history.com/#aether-raid/modular-search">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=aether-raid/AERO&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=aether-raid/AERO&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=aether-raid/AERO&type=Date" />
  </picture>
</a>
</p>
