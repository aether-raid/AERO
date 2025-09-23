# AERO: Automated Exploration, Research & Orchestration – A Framework for Machine Learning Research

AERO is a modular, end-to-end framework designed to streamline and enhance machine learning research. It consists of **five main workflows** that collectively support the research lifecycle, from problem formulation to publication. The system leverages Large Language Models (LLMs) within a graph-based orchestration engine (LangGraph) to enable flexible, interpretable, and automated guidance for researchers, ensuring that each stage of the workflow is context-aware and informed by prior knowledge.


## 📝 Abstract
Machine learning researchers often face significant challenges in conducting end-to-end research due to the complexity of modern problems, the rapidly growing volume of literature, and the multi-step nature of tasks such as model selection, experiment design, data analysis, and paper writing. In recent years, Large Language Models (LLMs) have shown remarkable capabilities in knowledge extraction, reasoning, code generation, and summarization, making them well-suited to assist researchers across these tasks. A modular architecture is ideal, where core LLM capabilities operate independently of any specific workflow, enabling flexible orchestration and integration.

We propose AERO, a modular and adaptable framework for machine learning research, which integrates LLMs as a core component and orchestrates five main workflows: **model recommendation, research planning, experiment design, data analysis, and paper writing**. This modular and adaptable approach highlights the value of LLM-driven workflows in supporting efficient, systematic, and context-aware machine learning research.


---

## 🛠️ Installation and Set-Up

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy the provided `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and fill in your own API keys and settings:
     ```
     OPENAI_API_KEY='YOUR_OPENAI_KEY'
     BASE_URL='YOUR_BASE_URL'
     DEFAULT_MODEL='gemini/gemini-2.5-flash'
     GOOGLE_API_KEY='YOUR_GOOGLE_KEY'
     CX='YOUR_CUSTOM_SEARCH_CX'
     TAVILY_API_KEY='YOUR_TAVILY_API_KEY'
     ```
   - **Do not** commit your real `.env` file to version control. The `.env.example` file is safe to share and shows users what variables they need.

---

## 🚀 Running the Workflows

### 1. Model Recommendation
blah blah

### 2. Research Planning
blah blah

### 3. Experiment Design
Given a research plan, the system extracts key information and retrieves supporting literature to generate experiment ideas and designs. The final output is a detailed experimental design accompanied by executable Python code.

**Using it as a Python module**
You can import and use the workflow in your own Python scripts:
   ```python
    from design_experiment import run_design_workflow # Full Workflow 
    from design_experiment import design_experiment_workflow # (Optional) Langgraph Only 

    result = run_design_workflow(user_input)
    print(result["design"])
    print(result["code"])
   ```

**General Workflow:**
1. Input Processing: Extracts goals, hypotheses, experiment ideas (if provided), and other relevant details from a research plan.
2. Literature Retrieval System: Uses a Hybrid-RAG (Retrieval-Augmented Generation) approach to search and retrieve supporting literature (arXiv API).
3. Idea Generation: Employs [SakanaAI’s Tree-based Experimentation Module](https://github.com/SakanaAI/treequest) to generate promising experiment ideas (when no experiment idea is provided).
4. Design Refinement: Refines experiment ideas into structured experiment designs that include:
   - Datasets
   - Methodologies and implementation steps
   - References
   - Additional supporting details
5. Scoring and Refinement System: Evaluates and refines experiment designs based on key criterions to ensure quality, completeness, and relevance.
6. Code Generation: Produces minimal executable Python code after syntax validation and import checks 


### 4. Data Analysis & Suggest Experiments
blah blah

### 5. Paper Writing
blah blah

---


## 📄 License
This project is open source and available under the MIT License.


## 🤝 Acknowledgements
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [arXiv API](https://arxiv.org/help/api/index)
- [SakanaAI](https://sakana.ai/)
