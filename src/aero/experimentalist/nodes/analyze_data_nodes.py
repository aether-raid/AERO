
from ..shared_defs import ExperimentSuggestionState, _write_stream
import json
import time
from typing import List
import re

async def _analyze_experiment_findings_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Node for analyzing experimental findings and research context for experiment suggestions."""
    _write_stream("Experiment Analysis: Analyzing current findings and research context...")
    
    # Extract dependencies from state (adapting from class method)
    client = state["client"]
    model = state["model"]
    
    try:
        # Extract research context from the original prompt and any provided data
        original_prompt = state.get("original_prompt", "")
        uploaded_data = state.get("uploaded_data", [])
        experimental_results = state.get("experimental_results", {})
        
        # Combine user query with uploaded data (CSV or other file contents)
        full_prompt_context = _combine_query_and_data(original_prompt, uploaded_data)
        
        
        # Display uploaded data info if present
       
        # Check for previous analysis iterations and validation feedback
        analysis_iterations = state.get("analysis_iterations", [])
        analysis_validation_results = state.get("analysis_validation_results", {})
        current_iteration = len(analysis_iterations) + 1
        
        # DEBUG: Detailed iteration tracking
        #print(f"🐛 DEBUG _analyze_experiment_findings_node:")
        #print(f"  analysis_iterations length: {len(analysis_iterations)}")
        #print(f"  current_iteration calculated: {current_iteration}")
        #if analysis_iterations:
         #   #print(f"  last iteration in history: {analysis_iterations[-1].get('iteration', 'NO_ITERATION')}")
        #print(f"  state keys: {sorted(state.keys())}")

        # Extract validation feedback for improvement
        validation_feedback = ""
        if analysis_validation_results and current_iteration > 1:
            critical_issues = analysis_validation_results.get("critical_issues", [])
            completeness_gaps = analysis_validation_results.get("completeness_gaps", [])
            accuracy_concerns = analysis_validation_results.get("accuracy_concerns", [])
            improvement_recommendations = analysis_validation_results.get("improvement_recommendations", [])
            
            validation_feedback = f"""
                PREVIOUS VALIDATION FEEDBACK (Iteration {current_iteration - 1}):

                Critical Issues to Address:
                {chr(10).join(f"• {issue}" for issue in critical_issues[:3])}

                Completeness Gaps to Fill:
                {chr(10).join(f"• {gap}" for gap in completeness_gaps[:3])}

                Accuracy Concerns to Resolve:
                {chr(10).join(f"• {concern}" for concern in accuracy_concerns[:3])}

                Improvement Recommendations:
                {chr(10).join(f"• {rec}" for rec in improvement_recommendations[:3])}

                CRITICAL: Address ALL the above issues in this iteration. Provide specific, accurate, and complete analysis.
            """

        _write_stream(f"Generating analysis (iteration {current_iteration})")
        if validation_feedback:
            _write_stream(f"Incorporating validation feedback from previous iteration")

        # Enhanced analysis prompt for experiment suggestions
        iteration_context = f"ITERATION {current_iteration}" if current_iteration > 1 else "INITIAL ANALYSIS"
        
        analysis_prompt = f"""
        You are an expert machine learning researcher analyzing experimental findings to suggest follow-up experiments.
        
        **{iteration_context}**
        {validation_feedback}
        
        Original Research Question/Problem: "{original_prompt}"
        
        Experimental Results/Context: {experimental_results if experimental_results else "User described their current experimental situation in the prompt above"}
        
        Uploaded data: {uploaded_data if uploaded_data else "No additional data provided"}
        
        **CRITICAL**: If the user hasn't explicitly described their problem domain, you must infer it from:
        - Keywords in their prompt (computer vision, NLP, object detection, classification, etc.)
        - Data types mentioned (images, text, sensor data, time series, etc.)
        - Models or techniques referenced (CNNs, transformers, YOLO, etc.)
        - Applications mentioned (autonomous driving, medical imaging, etc.)
        
        VALIDATION REQUIREMENTS FOR ANALYSIS ACCURACY:
        1. **Be SPECIFIC and TECHNICAL** - avoid generic observations
        2. **Provide ACTIONABLE insights** - researchers must be able to act on your analysis
        3. **Ground analysis in DOMAIN EXPERTISE** - demonstrate deep understanding of the field
        4. **Include PRECISE technical details** - methods, datasets, metrics, benchmarks
        5. **Address validation feedback** (if provided above) - fix all identified issues
        6. **Ensure COMPLETENESS** - cover all required analysis sections thoroughly
        
        Please analyze this research context and provide a comprehensive analysis for experiment planning:
        
        1. **Domain Inference and Analysis** (BE SPECIFIC):
           - Primary research domain (computer vision, NLP, reinforcement learning, robotics, etc.)
           - Specific task type (object detection, classification, segmentation, generation, etc.)
           - Application area (autonomous vehicles, medical imaging, robotics, etc.)
           - Data characteristics (images, text, sensor data, time series, tabular, etc.)
           - Technical complexity level and requirements
           
        2. **Current State Assessment** (PROVIDE DETAILS):
           - What has been accomplished so far?
           - What are the key findings or results?
           - What metrics or performance indicators are being used?
           - Current model architectures or approaches being used
           - Performance baseline and targets
           
        3. **Technical Context** (INCLUDE SPECIFICS):
           - Frameworks and methodologies currently employed
           - Datasets being used or dataset characteristics
           - Computational constraints or requirements
           - Evaluation benchmarks and metrics
           - Hardware and software requirements
           
        4. **Research Gaps and Opportunities** (BE ACTIONABLE):
           - What questions remain unanswered in this domain?
           - What aspects need deeper investigation?
           - What are common failure modes or limitations in this area?
           - Areas for improvement or optimization
           - Specific research directions to pursue
           
        5. **Domain-Specific Considerations** (DEMONSTRATE EXPERTISE):
           - Key challenges specific to this research domain
           - Standard experimental practices in this field
           - Important datasets, benchmarks, or evaluation protocols
           - State-of-the-art methods and their limitations
           - Future research trends and opportunities
          
        
        Return your analysis in JSON format with clear, domain-specific insights that will inform targeted literature search and experiment suggestions.
        
        Example for computer vision/object detection:
        {{
            "domain_analysis": {{
                "primary_domain": "computer vision",
                "task_type": "object detection",
                "application_area": "autonomous driving",
                "data_type": "images/video"
            }},
            "current_state": {{
                "findings": "Model achieving X mAP on dataset Y",
                "current_approach": "YOLO-based detection pipeline"
            }},
            "research_opportunities": [
                "Multi-scale detection improvements",
                "Real-time inference optimization",
                "Small object detection enhancement"
            ]
        }}
        """
        
        
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role": "user", "content": analysis_prompt}],
        )

        
        # Parse the analysis
        analysis_text = response.choices[0].message.content.strip()
        #print(f"📋 Research Analysis: {analysis_text}")
        
        # Try to extract JSON from response
        try:
            # Clean and extract JSON from response
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text[7:]
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]
            analysis_text = analysis_text.strip()
            
            # Look for JSON in the response
            start = analysis_text.find('{')
            end = analysis_text.rfind('}') + 1
            if start != -1 and end != -1:
                analysis_json = json.loads(analysis_text[start:end])
                #print(f"✅ Successfully parsed research analysis JSON")
                
                # Validate and ensure required fields are present
                if "research_opportunities" not in analysis_json:
                    analysis_json["research_opportunities"] = ["Experimental validation", "Comparative studies", "Performance optimization"]
                    #print("⚠️ Added missing research_opportunities field")
                
                if "current_state" not in analysis_json:
                    analysis_json["current_state"] = {"status": "Analysis from LLM", "findings": "Based on user prompt and analysis"}
                    #print("⚠️ Added missing current_state field")
                    
            else:
                # Fallback: create structured analysis from domain inference
                #print("No JSON found, creating structured analysis...")
                
                # Try to infer domain from original prompt
                prompt_lower = original_prompt.lower()
                domain_info = _infer_domain_from_prompt(prompt_lower)
                
                analysis_json = {
                    "domain_analysis": domain_info,
                    "current_state": {"status": "Initial analysis", "findings": "Based on user prompt"},
                    "research_opportunities": ["Experimental validation", "Comparative studies", "Performance optimization"],
                    "summary": "Analysis based on prompt content and domain inference"
                }
                
        except json.JSONDecodeError as e:
            _write_stream(f"JSON parsing failed: {e}, creating fallback analysis...")
            # Fallback: create structured analysis
            prompt_lower = original_prompt.lower()
            domain_info = _infer_domain_from_prompt(prompt_lower)
            
            analysis_json = {
                "domain_analysis": domain_info,
                "current_state": {"status": "Initial analysis", "findings": "Based on user prompt"},
                "research_opportunities": ["Experimental validation", "Comparative studies", "Performance optimization"],
                "summary": f"Research analysis for {domain_info.get('primary_domain', 'machine learning')} project"
            }
                
        except Exception as e:
            _write_stream(f"JSON parsing failed: {e}, using fallback analysis")
            # Fallback analysis with extracted key information
            prompt_lower = original_prompt.lower()
            domain_info = _infer_domain_from_prompt(prompt_lower)
            
            analysis_json = {
                "domain_analysis": domain_info,
                "current_state": {"status": "Analysis from prompt", "findings": original_prompt[:200]},
                "research_opportunities": ["Follow-up experiments", "Comparative analysis"],
                "summary": f"Fallback analysis for {domain_info.get('primary_domain', 'machine learning')} research"
            }
        
        # Store the analysis and update state
        return {
            **state,
            "findings_analysis": analysis_json,
            "current_analysis_iteration": current_iteration,
            "research_context": {
                "original_prompt": original_prompt,
                "domain": analysis_json.get("domain_analysis", {}).get("primary_domain", "machine learning"),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "analysis_completed": True,
            "current_step": "findings_analyzed"
        }
        
    except Exception as e:
        print(f"Error in analyze_experiment_findings: {str(e)}")
        return {
            **state,
            "current_analysis_iteration": state.get("current_analysis_iteration", 1),
            "errors": state.get("errors", []) + [f"Findings analysis error: {str(e)}"],
            "analysis_completed": False,
            "current_step": "analysis_error"
        }








async def _validate_analysis_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Node for validating the generated data analysis with hyper-strict criteria."""
    client = state["client"]
    model = state["model"]
    _write_stream("Analysis Validation: Evaluating generated data analysis for accuracy, completeness, and grounding.")
    
    try:
        # Extract current analysis and context
        findings_analysis = state.get("findings_analysis", {})
        original_prompt = state.get("original_prompt", "")
        experimental_results = state.get("experimental_results", {})
        uploaded_data = state.get("uploaded_data", [])
        
        # Track analysis iteration history
        analysis_iterations = state.get("analysis_iterations", [])
        current_iteration = len(analysis_iterations) + 1
        
        # DEBUG: Track validation iteration calculation
        #print(f"🐛 DEBUG _validate_analysis_node:")
        ##print(f"  analysis_iterations length: {len(analysis_iterations)}")
        #print(f"  current_iteration calculated: {current_iteration}")
        #if analysis_iterations:
        #    print(f"  last iteration in history: {analysis_iterations[-1].get('iteration', 'NO_ITERATION')}")
        
        # Add current analysis to history
        current_analysis_record = {
            "iteration": current_iteration,
            "analysis": str(findings_analysis)[:500] if findings_analysis else "No analysis generated",
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        analysis_iterations.append(current_analysis_record)
        
        # Create iteration history context
        iteration_history = ""
        if len(analysis_iterations) > 1:
            iteration_history = "\n\nPREVIOUS ANALYSIS ITERATIONS:\n"
            for i, iteration in enumerate(analysis_iterations[:-1], 1):
                iteration_history += f"Iteration {iteration['iteration']}: {iteration['analysis'][:100]}...\n\n"
        
        validation_prompt = f"""
            You are a HYPER-STRICT data analysis validator. Your job is to rigorously evaluate the generated analysis for **accuracy**, **completeness**, **logical consistency**, **domain expertise**, and **actionable insights**.

            ORIGINAL RESEARCH REQUEST:
            {original_prompt}

            EXPERIMENTAL CONTEXT:
            {experimental_results}
            
            Uploaded data: 
            {uploaded_data if uploaded_data else "No additional data provided"}

            GENERATED ANALYSIS:
            {findings_analysis}

            CURRENT ITERATION: {current_iteration}
            {iteration_history}

            HYPER-STRICT VALIDATION CRITERIA (ALL MUST BE SATISFIED):
            1. **Domain Accuracy**: Is the domain identification and characterization correct and specific?
            2. **Technical Completeness**: Does the analysis include all necessary technical details (datasets, methods, metrics)?
            3. **Logical Consistency**: Are all conclusions logically supported by the provided context?
            4. **Insight Quality**: Does the analysis provide actionable, meaningful insights beyond obvious observations?
            5. **Contextual Grounding**: Is the analysis properly grounded in the user's specific research context?
            6. **Gap Identification**: Are research gaps and opportunities clearly and accurately identified?
            7. **Technical Depth**: Does the analysis demonstrate appropriate domain expertise and technical understanding?
            8. **Actionability**: Can researchers actually use this analysis to make informed decisions?
            9. **Specificity**: Are recommendations specific enough to be implementable rather than vague?
            10. **Accuracy Verification**: Are all technical claims and domain assertions verifiable and correct?

            ZERO-TOLERANCE REQUIREMENTS:
            - No generic or template-like responses
            - No vague or aspirational language without specifics
            - No technical inaccuracies or domain mischaracterizations
            - No missing critical analysis components
            - No unsupported claims or assumptions

            ITERATION ANALYSIS:
            - If this is iteration 1: Apply HYPER-STRICT validation
            - If iteration 2+: Ensure ALL previous issues resolved AND significant improvement demonstrated

            Return your assessment in this exact JSON format:
            {{
                "validation_result": "PASS" | "FAIL",
                "overall_score": 0.0-1.0,
                "detailed_scores": {{
                    "domain_accuracy": 0.0-1.0,
                    "technical_completeness": 0.0-1.0,
                    "logical_consistency": 0.0-1.0,
                    "insight_quality": 0.0-1.0,
                    "contextual_grounding": 0.0-1.0,
                    "gap_identification": 0.0-1.0,
                    "technical_depth": 0.0-1.0,
                    "actionability": 0.0-1.0,
                    "specificity": 0.0-1.0,
                    "accuracy_verification": 0.0-1.0
                }},
                "critical_issues": ["list", "of", "critical", "problems"],
                "completeness_gaps": ["missing", "analysis", "components"],
                "accuracy_concerns": ["technical", "inaccuracies", "or", "concerns"],
                "improvement_recommendations": ["specific", "actionable", "improvements"],
                "decision_rationale": "Clear explanation of pass/fail decision focusing on accuracy and completeness",
                "iteration_assessment": "Analysis of improvement from previous iterations (if applicable)",
                "confidence_in_validation": 0.0-1.0
            }}

            RUTHLESS PASSING THRESHOLD: Overall score ≥ 0.90 AND no critical issues AND no completeness gaps AND no accuracy concerns AND all detailed scores ≥ 0.85
            BE ABSOLUTELY STRICT: Only pass analyses that are **technically perfect**, **completely accurate**, **deeply insightful**, and **fully actionable**.
        """

        # Call LLM for validation
        
        
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[ {"role": "system", "content": "You are a ruthlessly strict data analysis validator. Provide objective, rigorous assessments in valid JSON format. Be ultra-conservative - only pass analyses that are technically perfect and deeply insightful."},
                    {"role": "user", "content": validation_prompt}]
        )

        validation_content = response.choices[0].message.content.strip()
        
        # Parse validation response
        try:
            # Clean and extract JSON
            json_match = re.search(r'\{.*\}', validation_content, re.DOTALL)
            if json_match:
                validation_json = json.loads(json_match.group(0))
            else:
                raise json.JSONDecodeError("No JSON found", validation_content, 0)
            
            validation_result = validation_json.get("validation_result", "FAIL").upper()
            overall_score = validation_json.get("overall_score", 0.0)
            critical_issues = validation_json.get("critical_issues", [])
            completeness_gaps = validation_json.get("completeness_gaps", [])
            accuracy_concerns = validation_json.get("accuracy_concerns", [])
            improvement_recommendations = validation_json.get("improvement_recommendations", [])
            _write_stream(f"Analysis Validation Result: {validation_result} with overall score {overall_score:.2f}, needs 0.9 to pass.")
            _write_stream(f"Critical Issues: {len(critical_issues)}, Completeness Gaps: {len(completeness_gaps)}, Accuracy Concerns: {len(accuracy_concerns)}")
            # Safety check: Enforce RUTHLESS thresholds
            if overall_score < 0.90 or len(critical_issues) > 0 or len(completeness_gaps) > 0 or len(accuracy_concerns) > 0:
                validation_result = "FAIL"
            
            # Check iteration limit (max 3 iterations to prevent infinite loops)
            if current_iteration >= 3 and validation_result == "FAIL":
                _write_stream(f"⚠️ Maximum analysis iterations reached ({current_iteration}). Forcing continuation with current analysis with {len(critical_issues)}, critical issues, {len(completeness_gaps)} completeness gaps, {len(accuracy_concerns)} accuracy concerns.")
               
                validation_result = "PASS"
                validation_json["forced_pass"] = True
                validation_json["decision_rationale"] = f"Forced pass after {current_iteration} iterations to prevent infinite loop. Original validation failed due to: {len(critical_issues)} critical issues, {len(completeness_gaps)} completeness gaps, {len(accuracy_concerns)} accuracy concerns."
            
            # Store validation results in state
            updated_state = {
                **state,
                "analysis_validation_results": validation_json,
                "analysis_iterations": analysis_iterations,
                "analysis_validation_decision": validation_result,
                "current_analysis_iteration": current_iteration,
                "current_step": "analysis_validated"
            }

            _write_stream(f"DEBUG VALIDATION RESULT FORM STATE SET: {updated_state['analysis_validation_decision']}")

            forced_pass = validation_json.get("forced_pass", False)
            
            # Safety check: After 3 iterations, force continue to avoid infinite loops
            if current_iteration >= 3:
                if forced_pass:
                    _write_stream(f"Maximum analysis iterations reached ({current_iteration}). Forced pass due to iteration limit - continuing to research direction despite validation issues.")
                else:
                    _write_stream(f"Maximum analysis iterations reached ({current_iteration}). Continuing to research direction.")
                updated_state["next_node"] = "decide_research_direction"
            # Check validation result - but distinguish between genuine pass and forced pass
            elif validation_result == "PASS":
                if forced_pass:
                    _write_stream(f"Analysis validation was FORCED to pass after max iterations. Continuing to research direction with unresolved issues.")
                else:
                    _write_stream(f"Analysis validation passed. Continuing to research direction.")
                updated_state["next_node"] = "decide_research_direction"
            else:
                _write_stream(f"Analysis validation failed. Iterating to improve analysis (iteration {current_iteration + 1}).")
                updated_state["next_node"] = "analyze_findings"
            
            return updated_state
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse analysis validation JSON: {e}")
            # Default to FAIL for safety
            fallback_validation = {
                "validation_result": "FAIL",
                "overall_score": 0.4,
                "critical_issues": ["JSON parsing error in analysis validation"],
                "improvement_recommendations": ["Regenerate analysis with clearer structure"],
                "decision_rationale": "Analysis validation failed due to parsing error",
                "error": str(e)
            }
            
            return {
                **state,
                "analysis_validation_results": fallback_validation,
                "analysis_iterations": analysis_iterations,
                "analysis_validation_decision": "FAIL",
                "current_analysis_iteration": current_iteration,
                "current_step": "analysis_validation_error",
                "next_node": "analyze_findings"  # Always retry on error
            }
            
    except Exception as e:
        print(f"Error in validate_analysis: {str(e)}")
        # Default to PASS to avoid blocking the workflow
        error_validation = {
            "validation_result": "PASS",
            "overall_score": 0.6,
            "decision_rationale": f"Analysis validation error occurred: {str(e)}. Defaulting to PASS to continue workflow.",
            "error": str(e)
        }
        
        return {
            **state,
            "analysis_validation_results": error_validation,
            "analysis_iterations": analysis_iterations,
            "analysis_validation_decision": "FAIL",
            "current_analysis_iteration": current_iteration,
            "errors": state.get("errors", []) + [f"Analysis validation error: {str(e)}"],
            "current_step": "analysis_validation_error_pass",
            "next_node": "analyze_findings"  # Always retry on error
        }



#----------------- some helper functions -----------------


def _infer_domain_from_prompt(prompt_lower: str) -> dict:
    """Infer research domain from user prompt keywords."""
    domain_info = {
        "primary_domain": "machine learning",
        "task_type": "experimental",
        "application_area": "",
        "data_type": ""
    }
    
    # Computer Vision keywords
    if any(kw in prompt_lower for kw in ["computer vision", "cv", "image", "video", "visual", "detection", "segmentation", "yolo", "cnn", "resnet"]):
        domain_info["primary_domain"] = "computer vision"
        if "detection" in prompt_lower:
            domain_info["task_type"] = "object detection"
        elif "segmentation" in prompt_lower:
            domain_info["task_type"] = "segmentation"
        elif "classification" in prompt_lower and "image" in prompt_lower:
            domain_info["task_type"] = "image classification"
        domain_info["data_type"] = "images"
        
    # NLP keywords
    elif any(kw in prompt_lower for kw in ["nlp", "text", "language", "transformer", "bert", "gpt", "sentiment", "translation"]):
        domain_info["primary_domain"] = "NLP"
        if "classification" in prompt_lower:
            domain_info["task_type"] = "text classification"
        elif "generation" in prompt_lower:
            domain_info["task_type"] = "text generation"
        elif "translation" in prompt_lower:
            domain_info["task_type"] = "machine translation"
        domain_info["data_type"] = "text"
        
    # Robotics keywords
    elif any(kw in prompt_lower for kw in ["robot", "autonomous", "control", "navigation", "manipulation"]):
        domain_info["primary_domain"] = "robotics"
        domain_info["task_type"] = "control"
        domain_info["data_type"] = "sensor data"
        
    # Time series keywords
    elif any(kw in prompt_lower for kw in ["time series", "temporal", "forecasting", "lstm", "sequence"]):
        domain_info["primary_domain"] = "time series analysis"
        domain_info["task_type"] = "forecasting"
        domain_info["data_type"] = "time series"
        
    # Application areas
    if any(kw in prompt_lower for kw in ["medical", "healthcare", "clinical"]):
        domain_info["application_area"] = "medical"
    elif any(kw in prompt_lower for kw in ["autonomous", "driving", "vehicle"]):
        domain_info["application_area"] = "autonomous driving"
    elif any(kw in prompt_lower for kw in ["finance", "financial", "trading"]):
        domain_info["application_area"] = "finance"
        
    return domain_info


def _combine_query_and_data(user_query: str, uploaded_data: List[str]) -> str:
    """Combine user query with uploaded data for LLM prompts when needed."""
    if not uploaded_data:
        return user_query
    
    combined = user_query
    combined += "\n\nAttached Contexts:\n" + "\n\n".join(uploaded_data)
    return combined