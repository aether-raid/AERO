from ..shared_defs import ExperimentSuggestionState, _write_stream
import re
import json

async def _decide_research_direction_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Node for deciding the research direction based on analysis findings."""
    _write_stream("Research Direction: Determining optimal research path with justification.")
    client = state["client"]
    model = state["model"]
    try:
        # Extract analysis context
        original_prompt = state.get("original_prompt", "")
        findings_analysis = state.get("findings_analysis", {})
        experimental_results = state.get("experimental_results", {})
        research_context = state.get("research_context", {})
        
        # Extract iteration history and validation feedback
        direction_iterations = state.get("direction_iterations", [])
        validation_results = state.get("direction_validation_results", {})
        current_iteration = len(direction_iterations) + 1
        
        # Prepare context for direction decision
        findings_summary = findings_analysis.get("summary", "No analysis available")
        key_insights = findings_analysis.get("key_insights", [])
        limitations = findings_analysis.get("limitations", [])
        
        # Create iteration history context
        iteration_context = ""
        validation_feedback = ""
        
        if current_iteration > 1:
            iteration_context = "\n\nPREVIOUS RESEARCH DIRECTION ATTEMPTS:\n"
            for i, iteration in enumerate(direction_iterations, 1):
                iteration_context += f"Attempt {iteration['iteration']}: {iteration['direction']}\n"
                iteration_context += f"  Issues: Failed validation\n"
                iteration_context += f"  Confidence: {iteration['confidence_level']}\n\n"
            
            if validation_results.get("critical_issues"):
                validation_feedback = f"\n\nCRITICAL ISSUES FROM PREVIOUS VALIDATION:\n"
                for issue in validation_results.get("critical_issues", [])[:3]:
                    validation_feedback += f"• {issue}\n"
            
            if validation_results.get("improvement_recommendations"):
                validation_feedback += f"\nIMPROVEMENT RECOMMENDATIONS:\n"
                for rec in validation_results.get("improvement_recommendations", [])[:3]:
                    validation_feedback += f"• {rec}\n"
        
        # Create comprehensive prompt for direction decision
        direction_prompt = f"""
        You are a senior research strategist. Based on the experimental findings and analysis, determine the most promising research direction to pursue next.

        ORIGINAL RESEARCH CONTEXT:
        {original_prompt}

        EXPERIMENTAL FINDINGS SUMMARY:
        {findings_analysis if findings_analysis else "No analysis available"}

        KEY INSIGHTS FROM ANALYSIS:
        {chr(10).join(f"• {insight}" for insight in key_insights[:5])}

        IDENTIFIED LIMITATIONS:
        {chr(10).join(f"• {limitation}" for limitation in limitations[:3])}

        EXPERIMENTAL DATA OVERVIEW:
        {str(experimental_results)[:500] if experimental_results else "No experimental data provided"}

        ITERATION CONTEXT:
        Current Iteration: {current_iteration}
        {iteration_context}
        {validation_feedback}

        INSTRUCTIONS FOR ITERATION {current_iteration}:
        {"INITIAL DIRECTION GENERATION:" if current_iteration == 1 else f"IMPROVED DIRECTION GENERATION (addressing previous failures):"}

        Your task is to:

        - Identify 2 novel future research directions that go beyond the current study’s scope, inspired by its findings but not repeating them.

        - Select the most promising direction and provide a clear justification for this choice.

        - Explain why this direction is optimal given the strengths and limitations of the current results.

        - Outline the specific aspects, variables, or methodologies that should be investigated to pursue this new direction.
        
        {"CRITICAL FOR ITERATION " + str(current_iteration) + ": The previous direction failed validation. You MUST significantly improve by addressing validation feedback and avoiding previous issues." if current_iteration > 1 else ""}
        
        Return your response in this exact JSON format:
        {{
            "potential_directions": [
                {{
                    "direction": "Brief direction description",
                    "rationale": "Why this direction is promising",
                    "feasibility": "Assessment of feasibility (High/Medium/Low)"
                }}
            ],
            "selected_direction": {{
                "direction": "Chosen research direction",
                "justification": "Detailed explanation of why this direction was selected",
                "expected_impact": "What outcomes this direction could achieve",
                "key_questions": ["Question 1", "Question 2", "Question 3"],
                "confidence_level": "High/Medium/Low"
            }},
            "reasoning": "Overall strategic reasoning for the decision"
        }}
        """

        # Call LLM for direction decision
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"content": direction_prompt, "role": "user"}]
        )
        
        direction_content = response.choices[0].message.content.strip()
        
        # Clean and parse JSON response
       
        
        # Clean the response to extract JSON
        json_match = re.search(r'\{.*\}', direction_content, re.DOTALL)
        if json_match:
            clean_json = json_match.group()
            try:
                direction_decision = json.loads(clean_json)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                direction_decision = {
                    "selected_direction": {
                        "direction": "Continue current research with refinements",
                        "justification": "Based on analysis, refining current approach shows promise",
                        "expected_impact": "Improved understanding of the research problem",
                        "key_questions": ["How to optimize current methods?", "What are the key bottlenecks?", "What alternative approaches exist?"],
                        "confidence_level": "Medium"
                    },
                    "reasoning": "Default direction based on analysis findings"
                }
        else:
            # Fallback direction
            direction_decision = {
                "selected_direction": {
                    "direction": "Investigate identified limitations and explore alternatives",
                    "justification": "Analysis revealed limitations that need addressing",
                    "expected_impact": "Better understanding of constraints and potential solutions",
                    "key_questions": ["What causes the identified limitations?", "What alternative methods exist?", "How can we validate improvements?"],
                    "confidence_level": "Medium"
                },
                "reasoning": "Focus on addressing key limitations identified in analysis"
            }

        _write_stream(f"Research direction decided: {direction_decision.get('selected_direction', {}).get('direction', 'Unknown')}")
        
        return {
            **state,
            "research_direction": direction_decision,
            "current_step": "direction_decided"
        }
        
    except Exception as e:
        print(f"❌ Error in decide_research_direction: {str(e)}")
        # Provide fallback direction
        fallback_direction = {
            "selected_direction": {
                "direction": "Continue systematic investigation",
                "justification": "Maintain research momentum while addressing any identified issues",
                "expected_impact": "Steady progress toward research objectives",
                "key_questions": ["What are the next logical steps?", "How can we improve current methods?", "What additional data is needed?"],
                "confidence_level": "Medium"
            },
            "reasoning": "Fallback direction due to processing error"
        }
        
        return {
            **state,
            "research_direction": fallback_direction,
            "errors": state.get("errors", []) + [f"Direction decision error: {str(e)}"],
            "current_step": "direction_error"
        }


async def _validate_research_direction_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Node for validating the proposed research direction and goals with strict evaluation."""
    _write_stream("Research Direction Validation: Evaluating proposed research goals and methodology.")
    model= state["model"]
    client = state["client"]
    try:
        # Extract current research direction and past iterations
        research_direction = state.get("research_direction", {})
        selected_direction = research_direction.get("selected_direction", {})
        original_prompt = state.get("original_prompt", "")
        findings_analysis = state.get("findings_analysis", {})
        
        # Track iteration history
        direction_iterations = state.get("direction_iterations", [])
        current_iteration = len(direction_iterations) + 1
        
        # Add current direction to history
        current_direction_record = {
            "iteration": current_iteration,
            "direction": selected_direction.get("direction", ""),
            "justification": selected_direction.get("justification", ""),
            "key_questions": selected_direction.get("key_questions", []),
            "confidence_level": selected_direction.get("confidence_level", "Medium"),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        direction_iterations.append(current_direction_record)
        
        # Create validation prompt with iteration history
        iteration_history = ""
        if len(direction_iterations) > 1:
            iteration_history = "\n\nPREVIOUS ITERATION HISTORY:\n"
            for i, iteration in enumerate(direction_iterations[:-1], 1):
                iteration_history += f"Iteration {iteration['iteration']}: {iteration['direction']}\n"
                iteration_history += f"  Confidence: {iteration['confidence_level']}\n"
                iteration_history += f"  Key Questions: {', '.join(iteration['key_questions'][:2])}\n\n"
        
        direction_text = selected_direction.get("direction", "")
        justification = selected_direction.get("justification", "")
        key_questions = selected_direction.get("key_questions", [])
        confidence_level = selected_direction.get("confidence_level", "Medium")
        
        validation_prompt = f"""
            You are a strict research methodology validator. Your job is to rigorously evaluate the proposed research direction for both **scientific soundness** and **strategic value**. You must decide not only if the direction is valid, but also if it is a *worthwhile and meaningful path to pursue* given the research context.

            ORIGINAL RESEARCH REQUEST:
            {original_prompt}

            PROPOSED RESEARCH DIRECTION:
            Direction: {direction_text}
            Justification: {justification}
            Key Questions: {chr(10).join(f"• {q}" for q in key_questions[:5])}
            Confidence Level: {confidence_level}

            CURRENT ITERATION: {current_iteration}
            {iteration_history}

            RESEARCH CONTEXT:
            {findings_analysis}

            STRICT VALIDATION CRITERIA:
            1. **Alignment**: Does this direction directly address the original research request?
            2. **Scientific Rigor**: Are the research questions testable, well-formulated, and methodologically sound?
            3. **Feasibility**: Can this direction realistically be pursued with typical resources (time, compute, expertise)?
            4. **Novelty & Value**: Does this offer meaningful new insights or open promising lines of inquiry, rather than repeating well-trodden ground?
            5. **Impact Potential**: Could this direction lead to significant contributions, practical applications, or field-shaping results?
            6. **Clarity**: Are the objectives, approach, and rationale clearly and coherently defined?
            7. **Scope**: Is the scope balanced (not too broad to be vague, not too narrow to be trivial)?
            8. **Strategic Fit**: Given the research context, is this a *good direction to pursue now* compared to alternatives?

            ITERATION ANALYSIS:
            - If this is iteration 1: Apply standard validation
            - If iteration 2+: Ensure improvements over previous attempts, avoid repeated mistakes

            Return your assessment in this exact JSON format:
            {{
            "validation_result": "PASS" | "FAIL",
            "overall_score": 0.0-1.0,
            "detailed_scores": {{
                "alignment": 0.0-1.0,
                "scientific_rigor": 0.0-1.0,
                "feasibility": 0.0-1.0,
                "novelty_value": 0.0-1.0,
                "impact_potential": 0.0-1.0,
                "clarity": 0.0-1.0,
                "scope": 0.0-1.0,
                "strategic_fit": 0.0-1.0
            }},
            "critical_issues": ["list", "of", "critical", "problems"],
            "improvement_recommendations": ["specific", "actionable", "improvements"],
            "decision_rationale": "Clear explanation of pass/fail decision, weighing novelty, impact, and feasibility",
            "iteration_assessment": "Analysis of improvement from previous iterations (if applicable)",
            "confidence_in_validation": 0.0-1.0
            }}

            PASSING THRESHOLD: Overall score ≥ 0.75 AND no critical issues AND all detailed scores ≥ 0.6
            BE STRICT: Only pass directions that are both **methodologically solid** and **worth pursuing** for novelty and impact.
            """

        # Call LLM for validation
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"content": validation_prompt, "role": "user"}]
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
            improvement_recommendations = validation_json.get("improvement_recommendations", [])
            
            # Safety check: Enforce strict thresholds
            if overall_score < 0.75 or len(critical_issues) > 0:
                validation_result = "FAIL"
            
            # Check iteration limit (max 3 iterations to prevent infinite loops)
            if current_iteration >= 3 and validation_result == "FAIL":
                _write_stream(f"Maximum iterations reached ({current_iteration}). Forcing continuation with current direction with {len(critical_issues)} critical issues.")
               
               
                validation_result = "PASS"
                validation_json["forced_pass"] = True
                validation_json["decision_rationale"] = f"Forced pass after {current_iteration} iterations to prevent infinite loop. Original validation failed due to: {len(critical_issues)} critical issues."
            ''''
            print("\n" + "=" * 80)
            print("🔍 RESEARCH DIRECTION VALIDATION RESULTS")
            print("=" * 80)
            print(f"📊 Iteration: {current_iteration}")
            if validation_json.get("forced_pass"):
                print(f"🎯 Validation Result: {validation_result} (⚠️ FORCED PASS - VALIDATION FAILED)")
            else:
                print(f"🎯 Validation Result: {validation_result}")
            print(f"📈 Overall Score: {overall_score:.2f}/1.0")
            print(f"🔴 Critical Issues: {len(critical_issues)}")
            hallucination_flags = validation_json.get("hallucination_flags", [])
            
            if critical_issues:
                print("Critical Issues:")
                for issue in critical_issues[:3]:
                    print(f"  • {issue}")
            
            if validation_result == "FAIL" and improvement_recommendations:
                print("🔧 Improvement Recommendations:")
                for rec in improvement_recommendations[:3]:
                    print(f"  • {rec}")
            
            print(f"💭 Decision Rationale: {validation_json.get('decision_rationale', 'No rationale provided')}")
            print("=" * 80)
            '''
           
            # Check if this was a forced pass due to max iterations
            forced_pass = validation_json.get("forced_pass", False)
            
            # Safety check: After 3 iterations, force continue to avoid infinite loops
            if current_iteration >= 3:
                if forced_pass:
                    _write_stream(f"Maximum direction iterations reached ({current_iteration}). Forced pass due to iteration limit - continuing to experiments despite validation issues.")
                else:
                    _write_stream(f"Maximum direction iterations reached ({current_iteration}). Continuing to experiments.")
                next_node = "generate_experiment_search_query"
                
            # Check validation result - but distinguish between genuine pass and forced pass
            elif validation_result == "PASS":
                if forced_pass:
                    _write_stream(f"Research direction validation was FORCED to pass after max iterations. Continuing to experiments with unresolved issues.")
                else:
                    _write_stream(f"Research direction validation passed. Continuing to experiments.")
                next_node = "generate_experiment_search_query"
            else:
                _write_stream(f"Research direction validation failed. Iterating to improve direction (iteration {current_iteration + 1}).")
                next_node = "decide_research_direction"
            
            # Store validation results in state
            return {
                **state,
                "direction_validation_results": validation_json,
                "direction_iterations": direction_iterations,
                "direction_validation_decision": validation_result,
                "current_iteration": current_iteration,
                "current_step": "research_direction_validated",
                "next_node": next_node
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse validation JSON: {e}")
            # Default to FAIL for safety
            fallback_validation = {
                "validation_result": "FAIL",
                "overall_score": 0.4,
                "critical_issues": ["JSON parsing error in validation"],
                "improvement_recommendations": ["Regenerate research direction with clearer objectives"],
                "decision_rationale": "Validation failed due to parsing error",
                "error": str(e)
            }
            
            return {
                **state,
                "direction_validation_results": fallback_validation,
                "direction_iterations": direction_iterations,
                "direction_validation_decision": "FAIL",
                "current_iteration": current_iteration,
                "current_step": "validation_error",
                "next_node": "decide_research_direction"  # Always retry on error
            }
            
    except Exception as e:
        print(f"❌ Error in validate_research_direction: {str(e)}")
        # Default to PASS to avoid blocking the workflow
        error_validation = {
            "validation_result": "PASS",
            "overall_score": 0.6,
            "decision_rationale": f"Validation error occurred: {str(e)}. Defaulting to PASS to continue workflow.",
            "error": str(e)
        }
        
        return {
            **state,
            "direction_validation_results": error_validation,
            "direction_iterations": direction_iterations,
            "direction_validation_decision": "PASS",
            "current_iteration": current_iteration,
            "errors": state.get("errors", []) + [f"Direction validation error: {str(e)}"],
            "current_step": "validation_error_pass",
            "next_node": "generate_experiment_search_query"  # Continue on error with PASS
        }
