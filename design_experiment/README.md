
# Design Experiment

Given a research plan, the system extracts key information and retrieves supporting literature to generate experiment ideas and designs. The final output is a detailed experimental design accompanied by executable Python code.

## 📝 Abstract
Machine learning researchers increasingly struggle with experiment design due to the complexity of modern research problems and the overwhelming amount of available literature. Large Language Models (LLMs) provide strong capabilities in information extraction, knowledge retrieval, and code generation, making them well-suited to address these challenges. Our approach integrates LLMs into the experimental workflow, enabling a more efficient, adaptable, and systematic pathway from research concept to executable experimentation.

## 💡 Key Components
To address these challenges, we developed the Design Experiment Workflow, a LangGraph-based workflow that leverages LLMs to guide the experimental process. From a given research plan, the system performs the following steps:
1. Input Processing: Extracts goals, hypotheses, experiment ideas (if provided), and other relevant details from a research plan.
2. Literature Retrieval System: Uses a Hybrid-RAG (Retrieval-Augmented Generation) approach to search and retrieve supporting literature (arXiv API).
3. Idea Generation: Employs SakanaAI’s Tree-based Experimentation Module to generate promising experiment ideas (when no experiment idea is provided).
4. Design Refinement: Refines experiment ideas into structured experiment designs that includ:
   - Datasets
   - Methodologies and implementation steps
   - References
   - Additional supporting details
5. Scoring and Refinement System: Evaluates and refines experiment designs based on key criterions to ensure quality, completeness, and relevance.
6. Code Generation: Produces minimal executable Python code after syntax validation and import checks 


## 🛠️ Installation and Set-Up

1. **Install dependencies**:
   ```bash
   pip install -r requirements_experiment_tree.txt
   ```
2. **Run the main workflow**:
   ```bash
   python main.py
   ```

## License
See the main project repository for license information.