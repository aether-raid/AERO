from ..shared_defs import ModelSuggestionState, PropertyHit, BaseState, Evidence, _write_stream, _clean_text_for_utf8, ML_RESEARCH_CATEGORIES
import asyncio
from langchain_core.messages import AIMessage

import json
async def _analyze_properties_and_task_node(state: ModelSuggestionState) -> ModelSuggestionState:
    """Combined node for extracting properties and decomposing task concurrently."""
   # print("\n🤖 Step 1: Analyzing properties and decomposing task concurrently...")
    state["current_step"] = "analyze_properties_and_task"
    _write_stream("Started property and task analysis.")
    
    # Extract dependencies from state
    client = state["client"]
    model = state["model"]
    
    async def extract_properties():
        """Extract properties using LLM analysis."""
        _write_stream("Extracting properties from task description.")
      
      
        try:
            categories_list = "\n".join([f"- {category}" for category in ML_RESEARCH_CATEGORIES])
            
            content = f"""
                You are an expert machine learning researcher. Analyze the following research task and determine which of the predefined categories apply.

                Research Task: {state["original_prompt"]}

                Categories to analyze:
                {categories_list}

                For each category that applies to this research task, provide:
                1. The category name (exactly as listed above)
                2. A confidence score between 0.0 and 1.0 (how certain you are this category applies, Refer to the calibration table)
                3. A brief explanation of why this category applies
                4. Specific evidence from the task description that supports this categorization

                Confidence calibration (0.0–1.0):
                - 0.95–1.00: Category is explicitly stated or entailed by multiple strong cues.
                - 0.80–0.94: Strong single cue or multiple moderate cues; unlikely to be wrong.
                - 0.60–0.79: Reasonable inference with at least one clear cue; some uncertainty.
                - <0.60: Category is highly unlikely to apply, and can be safely ignored.

                Explanations:
                - 1–2 sentences, specific and non-generic, referencing how the evidence meets the category's definition.
                - Avoid restating the evidence verbatim; interpret it.

                Evidence rules:
                - "evidence" must be short verbatim quotes or near-verbatim spans from the task (≤ 20 words each). If paraphrase is unavoidable, mark with ~ at start (e.g., "~streaming data implies temporal order").
                - Provide 1–3 evidence snippets per category, concatenated with " | " if multiple.
                - No invented facts; no external knowledge.

                Do not filter categories down to only the applicable ones, you want to always return the full set, but include a confidence score for each (so the tool/user can judge relevance).

                Format your response as a JSON array like this:
                [
                {{
                    "category": "temporal_structure",
                    "confidence": 0.95,
                    "explanation": "The task explicitly mentions time series data which has temporal dependencies",
                    "evidence": "time series forecasting"
                }},
                {{
                    "category": "variable_length_sequences", 
                    "confidence": 0.85,
                    "explanation": "Task mentions variable length sequences",
                    "evidence": "variable length sequences"
                }}
                ]
                Always return valid JSON. For any field that may contain multiple values (e.g., evidence), output them as a JSON array of strings instead of separating by commas inside a single string.

                Return only the JSON array, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            # Parse the LLM response
            llm_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if llm_response.startswith("```json"):
                    llm_response = llm_response[7:]
                if llm_response.endswith("```"):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()
                
                properties_data = json.loads(llm_response)
                
                # Convert to PropertyHit objects and then to dict
                property_hits = []
                for prop_data in properties_data:
                    evidence = [Evidence(
                        snippet=prop_data.get("evidence", ""),
                        source=f"llm_analysis:{prop_data['category']}",
                        score=prop_data.get("confidence", 0.5)
                    )]
                    
                    property_hit = PropertyHit(
                        name=prop_data["category"],
                        evidence=evidence
                    )
                    property_hits.append(property_hit.to_dict())
                
                _write_stream(f"Property extraction completed: Found {len(property_hits)} properties")
                return {"success": True, "properties": property_hits}
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse LLM JSON response: {e}"
                _write_stream(f"Error: {error_msg}")
                return {"success": False, "error": error_msg, "properties": []}
        
        except Exception as e:
            error_msg = f"LLM property extraction failed: {str(e)}"
            _write_stream(f"Error:   {error_msg}")
            return {"success": False, "error": error_msg, "properties": []}

    async def decompose_task():
        """Decompose task using LLM analysis."""
        _write_stream("Analyzing task decomposition.")
        try:
            content = f"""
                You are an expert machine learning researcher. Analyze the following research task and decompose it into key properties and characteristics.

                Task: {state["original_prompt"]}

                Please identify and analyze the following aspects:

                1. **Data Type**: What kind of data is involved? (text, images, time series, tabular, etc.)
                2. **Learning Type**: What type of learning is this? (supervised, unsupervised, reinforcement, etc.)
                3. **Task Category**: What is the main ML task? (classification, regression, generation, clustering, etc.)
                4. **Architecture Requirements**: What types of models or architectures might be suitable?
                5. **Key Challenges**: What are the main technical challenges?
                6. **Data Characteristics**: 
                - Variable length sequences?
                - Fixed or variable input dimensions?
                - Temporal structure?
                - Multi-modal data?
                7. **Performance Metrics**: What metrics would be appropriate for evaluation?
                8. **Domain Specifics**: Any domain-specific considerations?

                Provide your analysis in a structured JSON format with clear explanations for each identified property.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            detailed_analysis = {
                "llm_analysis": response.choices[0].message.content,
                "model_used": model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown"
            }
            
            _write_stream("Task decomposition completed.")
            return {"success": True, "analysis": detailed_analysis}
        
        except Exception as e:
            error_msg = f"LLM decomposition failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "analysis": {"error": error_msg, "llm_analysis": None}}

    # Run both tasks concurrently
    #print("🔄 Running property extraction and task decomposition in parallel...")
    properties_result, decomposition_result = await asyncio.gather(
        extract_properties(),
        decompose_task(),
        return_exceptions=True
    )
    
    # Handle results
    if isinstance(properties_result, Exception):
        error_msg = f"Property extraction failed: {str(properties_result)}"
        state["errors"].append(error_msg)
        state["detected_categories"] = []
        _write_stream(f"Error: {error_msg}")
    elif properties_result["success"]:
        state["detected_categories"] = properties_result["properties"]
        #for prop in properties_result["properties"][:5]:
            
            #_write_stream(f"  - {prop['name']}: {prop['confidence']:.2f} confidence")
    else:
        state["errors"].append(properties_result["error"])
        state["detected_categories"] = properties_result["properties"]
    
    if isinstance(decomposition_result, Exception):
        error_msg = f"Task decomposition failed: {str(decomposition_result)}"
        state["errors"].append(error_msg)
        state["detailed_analysis"] = {"error": error_msg, "llm_analysis": None}
        _write_stream(f"Error: {error_msg}")
    elif decomposition_result["success"]:
        state["detailed_analysis"] = decomposition_result["analysis"]
    else:
        state["errors"].append(decomposition_result["error"])
        state["detailed_analysis"] = decomposition_result["analysis"]
    
    # Add success messages
    if properties_result.get("success") and decomposition_result.get("success"):
        state["messages"].append(
            AIMessage(content=f"Successfully analyzed task properties ({len(properties_result['properties'])} categories) and decomposed task characteristics concurrently.")
        )
    
    #print(f"DEBUG: State after _analyze_properties_and_task_node: detected_categories={len(state.get('detected_categories', []))}, errors={len(state.get('errors', []))}")
    _write_stream(f"Completed property and task analysis with {len(state.get('detected_categories', []))} detected categories and {len(state.get('errors', []))} errors.")
    return state

