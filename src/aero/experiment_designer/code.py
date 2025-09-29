import re
import ast
import importlib.util
import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple, Any
from langgraph.graph import StateGraph, END
from aero.experiment_designer.utils import get_llm_response, stream_writer

@dataclass
class CodeGenState:
    experiment_input: str = ""
    tags: List[Tuple[str, str]] = field(default_factory=list)
    generated_code: List[str] = field(default_factory=list)
    validation_results: List[Any] = field(default_factory=list)
    final_output: str = ""

def extract_code_tags(text: str):

    # Match [CODE_NEEDED: ...], [CODE_NEEDED] Description: ..., or [CODE_NEEDED] description
    pattern = r"\[CODE_NEEDED(?::\s*([^\]]+))?\](?:\s*Description:\s*([^\n]+)|\s*([^\n]+))?"
    tags = []
    for m in re.finditer(pattern, text):
        full_tag = m.group(0)
        description = m.group(1) or m.group(2) or m.group(3) or "No description provided"
        tags.append((full_tag, description.strip()))
    return tags

def strip_code_fence(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()

def validate_code(code: str):
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, [], f"Syntax error: {e}"
    imports = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split('.')[0])
    missing = []
    for mod in set(imports):
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return True, missing, None

# Node 1: Extract tags
async def extract_tags_node(state: CodeGenState, writer=None) -> CodeGenState:
    await stream_writer(f"Starting Code Generation Agent...", writer=writer, stream_mode="custom")

    state.tags = extract_code_tags(state.experiment_input)
    return state

# Node 2: Generate code for each tag (in parallel)
async def generate_code_node(state: CodeGenState, writer=None) -> CodeGenState:
    await stream_writer(f"Performing {len(state.tags)} code generation tasks...", writer=writer, stream_mode="custom")

    async def generate_for_tag(full_tag, description):
        messages = [
            {"role": "system", "content": (
                "You are a coding assistant. Generate only the minimal, essential Python code needed to execute the described steps. "
                "Do not include error handling, fallback logic, or alternative code. "
                "Use only brief comments for non-obvious steps. "
                "Do not explain the code, do not add extra print statements, and do not include any code that is not strictly required for the main task."
            )},
            {"role": "user", "content": f"Task: {description}\n\nGenerate only the minimal Python code required for this step. No error handling, no alternatives, no explanations, only the essential code."}
        ]
        code_snippet = await get_llm_response(messages)
        return strip_code_fence(code_snippet)

    # Launch all code generation tasks in parallel
    tasks = [generate_for_tag(full_tag, description) for full_tag, description in state.tags]
    generated = await asyncio.gather(*tasks)
    state.generated_code = generated
    return state

# Node 3: Validate each code cell
async def validate_code_node(state: CodeGenState) -> CodeGenState:
    results = []
    for code in state.generated_code:
        is_valid, missing, error = validate_code(code)
        results.append({
            "is_valid": is_valid,
            "missing_imports": missing,
            "error": error
        })
    state.validation_results = results
    return state

# Node 4: Refine code based on validation results
async def refine_code_node(state: CodeGenState, writer=None) -> CodeGenState:
    await stream_writer("Refining code based on validation results...", writer=writer, stream_mode="custom")

    refined_code = state.generated_code.copy()
    for idx, (code, val, (full_tag, description)) in enumerate(zip(state.generated_code, state.validation_results, state.tags)):
        # Fix syntax errors
        if not val["is_valid"]:
            messages = [
                {"role": "system", "content": (
                    "You are a coding assistant. Fix the provided Python code so it passes syntax validation. "
                    "Only return the corrected code, no explanations."
                )},
                {"role": "user", "content": (
                    f"Original task: {description}\n"
                    f"Previous code:\n{code}\n"
                    f"Validation error: {val['error']}\n"
                    "Please fix the code."
                )}
            ]
            new_code = await get_llm_response(messages)
            new_code = strip_code_fence(new_code)
            refined_code[idx] = new_code
        # Add pip install comments for missing imports
        elif val["missing_imports"]:
            pip_comments = "\n".join([f"# To use this code, run: pip install {mod}" for mod in val["missing_imports"]])
            # Only add the comment if not already present
            if pip_comments not in code:
                refined_code[idx] = f"{pip_comments}\n{code}"
    state.generated_code = refined_code
    return state

# Node 5: Assemble final output (all code in one cell, with section comments)
async def assemble_output_node(state: CodeGenState) -> CodeGenState:
    code_blocks = []
    for idx, ((full_tag, description), code, val) in enumerate(zip(state.tags, state.generated_code, state.validation_results)):
        section_comment = f"# --- {description} ---"
        validation_msg = ""
        if val["is_valid"]:
            unresolved = []
            for mod in val["missing_imports"]:
                pip_comment = f"# To use this code, run: pip install {mod}"
                if pip_comment not in code:
                    unresolved.append(mod)
            if unresolved:
                validation_msg = f"\n# ⚠️ Missing imports: {', '.join(unresolved)}"
        else:
            validation_msg = f"\n# ❌ Code validation failed: {val['error']}"
        code_block = f"{section_comment}\n{code}{validation_msg}"
        code_blocks.append(code_block)
    # Join all code blocks into one code cell
    state.final_output = f"```python\n" + "\n\n".join(code_blocks) + "\n```"
    return state

# Conditional: If any errors or missing imports, refine; else assemble
async def needs_refine(state: CodeGenState):
    for code, val in zip(state.generated_code, state.validation_results):
        # If code is invalid, needs fixing
        if not val["is_valid"]:
            return "refine_code"
        # If missing imports and no pip install comment present, needs fixing
        for mod in val["missing_imports"]:
            pip_comment = f"# To use this code, run: pip install {mod}"
            if pip_comment not in code:
                return "refine_code"
    return "assemble_output"

# Build the LangGraph workflow
def build_codegen_graph(writer=None):
    graph = StateGraph(CodeGenState)
    async def extract_tags(state):
        return await extract_tags_node(state, writer=writer)
    
    async def generate_code(state):
        return await generate_code_node(state, writer=writer)
    
    async def refine_code(state):
        return await refine_code_node(state, writer=writer)


    graph.add_node("extract_tags", extract_tags)
    graph.add_node("generate_code", generate_code)
    graph.add_node("validate_code", validate_code_node)
    graph.add_node("refine_code", refine_code)
    graph.add_node("assemble_output", assemble_output_node)
    graph.add_edge("extract_tags", "generate_code")
    graph.add_edge("generate_code", "validate_code")
    graph.add_conditional_edges("validate_code", needs_refine)
    graph.add_edge("refine_code", "validate_code")
    graph.add_edge("assemble_output", END)
    graph.set_entry_point("extract_tags")
    return graph.compile()
    