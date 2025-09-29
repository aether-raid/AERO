
from ..shared_defs import ExperimentSuggestionState, _write_stream 
import json
import re
from langchain_core.messages import AIMessage
import time

async def _distill_paper_methodologies_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Distill and condense methodology and experimental information from validated papers."""
    _write_stream("Distilling methodology and experimental information from papers...")
    state["current_step"] = "distill_methodologies"

    client = state["client"]
    model = state["model"]
    validated_papers = state.get("validated_experiment_papers", [])

    if not validated_papers:
        print("⚠️ No validated papers to distill")
        state["distilled_methodologies"] = {}
        return state

    try:
        distilled_methodologies = {}

        for i, paper in enumerate(validated_papers[:5], 1):  # Process top 5 papers
            paper_title = paper.get('title', f'Paper {i}')
            paper_content = paper.get('content', '')

            if not paper_content:
                #print(f"⚠️ Skipping paper {i} - no content available")
                continue

            #print(f"🔬 Distilling paper {i}: {paper_title[:50]}...")

            # Create distillation prompt with clearer instructions for brevity
            distillation_prompt = f"""Extract key methodology from this research paper in exactly 800 characters or less.

FOCUS ON:
• Model architecture (specific networks, layers, components)
• Training setup (batch size, learning rate, optimizer, epochs)
• Dataset and preprocessing
• Key experimental details

FORMAT: Bullet points only. Be concise but specific.

PAPER: {paper_title}
CONTENT: {paper_content}
IMPORTANT:
KEEP RESPONSES TO A MAXIMUM OF 600 CHARACTERS.
Provide methodology summary in under 600 characters:"""

            try:
                # Remove max_tokens to avoid truncation issues with LiteLLM proxy
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,  # Low temperature for factual extraction
                    messages=[{"content": distillation_prompt, "role": "user"}]
                )

                distilled_content = response.choices[0].message.content
                '''
                # Debug response info
                print(f"🔍 API response finish_reason: {response.choices[0].finish_reason}")
                print(f"🔍 Raw content type: {type(distilled_content)}")
                print(f"🔍 Raw content length: {len(distilled_content) if distilled_content else 0}")
                '''
                # Handle None content
                if distilled_content is None:
                    print(f"⚠️ API returned None content for paper {i}")
                    distilled_content = "API returned no content - processing failed"
                
                # Ensure content is within 1200 character limit
                if distilled_content and len(distilled_content) > 1200:
                    distilled_content = distilled_content[:1197] + "..."

                distilled_methodologies[f"paper_{i}"] = {
                    "title": paper_title,
                    "distilled_content": distilled_content or "No methodology information extracted",
                    "character_count": len(distilled_content) if distilled_content else 0
                }

                _write_stream(f"Distilled paper {i}: {len(distilled_content) if distilled_content else 0} characters")
                
            except Exception as api_error:
                print(f"API error for paper {i}: {api_error}")
                distilled_methodologies[f"paper_{i}"] = {
                    "title": paper_title,
                    "distilled_content": f"API error: {str(api_error)}",
                    "character_count": 0
                }

        state["distilled_methodologies"] = distilled_methodologies
        for i in range(len(validated_papers), 5):
            distilled_methodologies[f"paper_{i+1}"] = {
                "title": f"Paper {i+1}",
                "distilled_content": "No paper available",
                "character_count": 0
            }

        total_chars = sum(info['character_count'] for info in distilled_methodologies.values())
        successful_distillations = sum(1 for info in distilled_methodologies.values() if info['character_count'] > 0)
        
        #print(f"📚 Successfully distilled methodologies from {successful_distillations}/{len(distilled_methodologies)} papers")
        _write_stream(f"Total distilled content: {total_chars} characters")

    except Exception as e:
        error_msg = f"Methodology distillation failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        state["errors"].append(error_msg)
        state["distilled_methodologies"] = {}
        # Continue even on error - workflow will proceed to experiment generation

    return state

async def _suggest_experiments_tree_2_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Generate comprehensive experiment suggestions."""
    # Extract dependencies from state
    client = state["client"]
    model = state["model"]

    
    _write_stream("Generating experiment suggestions.")
    state["current_step"] = "suggest_experiments"
    
    try:
        research_direction = state.get("research_direction", {})
        findings_analysis = state.get("findings_analysis", {})
        validated_papers = state.get("validated_experiment_papers", [])
        original_prompt = state["original_prompt"]

        # Get validation feedback from previous iteration
        validation_feedback = state.get("validation_feedback", "")
        current_iteration = state.get("current_experiment_iteration", 0)

        # Store current experiment suggestions as previous before generating new ones
        current_suggestions = state.get("experiment_suggestions", "")
        if current_suggestions and current_iteration > 0:
            state["previous_experiment_suggestions"] = current_suggestions
           # print(f"📝 Stored previous experiment suggestions ({len(current_suggestions)} chars) for improvement guidance")
        
        # Get accumulated past mistakes for learning
        past_mistakes = state.get("past_experiment_mistakes", [])
        
        # Get previous experiment suggestions for direct improvement
        previous_experiment_suggestions = state.get("previous_experiment_suggestions", "")
        
        # Prepare context from distilled methodologies instead of raw papers
        distilled_methodologies = state.get("distilled_methodologies", {})
        papers_context = ""
        
        # Check if we have meaningful distilled content (not just empty entries)
        has_meaningful_distillation = any(
            info.get('distilled_content', '') and len(info.get('distilled_content', '')) > 10
            for info in distilled_methodologies.values()
        )
        
        if has_meaningful_distillation:
            papers_context = "Distilled Methodology and Experimental Information from Research Papers:\n\n"
            for paper_key, paper_info in distilled_methodologies.items():
                papers_context += f"Paper: {paper_info['title']}\n"
                papers_context += f"Distilled Content ({paper_info['character_count']} chars):\n"
                papers_context += f"{paper_info['distilled_content']}\n\n"
        elif validated_papers:
            # Fallback to raw papers if distillation failed or produced no content
            _write_stream("Using raw paper content as fallback (distillation produced no meaningful content)")
            papers_context = "Relevant Research Papers:\n"
            for i, paper in enumerate(validated_papers[:5], 1):
                papers_context += f"{i}. {paper.get('title', 'Unknown')}\n"
                papers_context += f"Content: {paper.get('content', 'No Content')[:2000]}...\n\n"
        else:
            papers_context = "No paper context available for experiment generation."
        
        # Add improvement guidance for iterations with past mistakes history
        improvement_guidance = ""
        if current_iteration > 0:
            improvement_parts = []
            
            if validation_feedback:
                improvement_parts.append(f"""
        CRITICAL IMPROVEMENT REQUIREMENTS (Iteration {current_iteration + 1}):
        Previous validation feedback: {validation_feedback}""")
            
            # Include detailed history of past mistakes to prevent repetition
            if past_mistakes:
                improvement_parts.append(f"""
        
        PAST MISTAKES HISTORY (LEARN FROM THESE - DO NOT REPEAT):
        You have failed validation {len(past_mistakes)} time(s) before. Study these mistakes carefully:""")
                
                for i, mistake in enumerate(past_mistakes, 1):
                    improvement_parts.append(f"""
        Iteration {mistake['iteration']} Failure (Score: {mistake['validation_score']:.2f}):
        - Critical Issues: {'; '.join(mistake['critical_issues'][:3]) if mistake['critical_issues'] else 'None'}
        - Direction Problems: {'; '.join(mistake['direction_misalignment'][:2]) if mistake['direction_misalignment'] else 'None'}
        - Novelty Issues: {'; '.join(mistake['novelty_concerns'][:2]) if mistake['novelty_concerns'] else 'None'}
        - Required Fixes: {'; '.join(mistake['improvement_recommendations'][:3]) if mistake['improvement_recommendations'] else 'None'}""")
            
            # Include the most recent experiment suggestions for direct improvement
            if previous_experiment_suggestions:
                improvement_parts.append(f"""
        
        PREVIOUS EXPERIMENT SUGGESTIONS (IMPROVE UPON THESE):
        Here are your most recent experiment suggestions that failed validation. Study them carefully and make targeted improvements:
        
        {previous_experiment_suggestions}
        
        Key areas to improve based on previous attempt:
        - Address any missing or incomplete sections
        - Enhance technical depth and specificity
        - Improve paper integration and citations
        - Strengthen methodology and evaluation procedures
        - Fix any structural or content issues identified in validation""")
            
            improvement_parts.append("""
        
        To pass strict validation, ensure:
        1. Design 2 comprehensive experiments.
        2. Include ALL required sections: **Objective**, **Hypothesis**, **Methodology**, **Expected Outcomes**, **Success Metrics**
        3. Integrate insights from at least 3 research papers with specific citations
        4. Provide detailed technical depth (datasets, models, training procedures, evaluation metrics)
        5. Include implementation timeline, resource requirements, and risk assessment
        6. Add statistical analysis and validation procedures
        7. Address ALL previous validation failures - do not repeat past mistakes""")
            
            improvement_guidance = "".join(improvement_parts)
        
        # Create experiment suggestion prompt with LITERATURE-GROUNDED but FLEXIBLE approach
        first_experiment_prompt = f"""
        You are an expert experimental researcher designing novel experiments grounded in existing literature. You can use techniques, models, and datasets from the provided papers, and combine them in novel ways to address the research questions.

        FLEXIBLE LITERATURE GROUNDING:
        - Use models, datasets, and techniques explicitly mentioned in the provided papers
        - Combine different approaches from the literature in novel ways
        - Reference specific papers for all experimental components
        - Create experiments that build upon and extend the existing work
        - If needed, use well-established techniques that complement the literature

        CRITICAL REQUIREMENTS:
        - Every experiment MUST reference at least 2 different papers from the provided literature
        - Models and datasets MUST be cited from the papers (e.g., "ResNet-50 (He et al., 2016)")
        - Experiments should address the key research questions directly
        - Include all required sections: Objective, Hypothesis, Methodology, Expected Outcomes, Success Metrics

        Original Research Context:
        {original_prompt}

        Research Direction:
        {json.dumps(research_direction, indent=2)}

        Findings Analysis:
        {json.dumps(findings_analysis, indent=2)}

        LITERATURE CONTEXT (MANDATORY REFERENCE MATERIAL):
        {papers_context}

        {improvement_guidance}

        EXPERIMENT DESIGN REQUIREMENTS:

        1. **LITERATURE GROUNDING**: Every experiment component MUST reference specific content from the provided papers
        2. **MODEL CITATIONS**: Every model mentioned MUST include original paper citation
        3. **DATASET CITATIONS**: Every dataset MUST be real with proper academic citation
        4. **TECHNIQUE VALIDATION**: Only use techniques explicitly described in the literature
        5. **PAPER INTEGRATION**: Reference at least 2 different papers for this experiment

        Generate ONE detailed experiment in the following format:

        ### Experiment 1: [Experiment Title - Reference specific paper technique]
        **Objective**: Clear statement grounded in literature findings
        **Hypothesis**: Testable hypothesis based on literature insights
        **Methodology**:
        - Use ONLY techniques from provided papers
        - Cite specific papers for each methodological choice
        - Reference exact models/datasets from literature
        **Expected Outcomes**: Based on literature expectations
        **Success Metrics**: Metrics mentioned in the papers
        **Resources Needed**: Based on literature resource requirements
        **Risk Assessment**: Risks identified in the literature
        **Literature References**: Specific papers supporting this experiment

        CRITICAL: Your experiment MUST be 100% grounded in the provided literature. If you cannot design an experiment using ONLY the content from these papers, state this explicitly rather than introducing external knowledge.
        """

        _write_stream("Generating first experiment...")
        response1 = client.chat.completions.create(
            model=model,
            temperature=0.3,
           # max_tokens=4000,
            messages=[{"content": first_experiment_prompt, "role": "user"}]
        )

        first_experiment = response1.choices[0].message.content

        # Ensure we have a valid string response
        if first_experiment is None:
            raise ValueError("LLM returned None response for first experiment")

        first_experiment = str(first_experiment)
        _write_stream("First experiment generated successfully")

        # Create experiment suggestion prompt for SECOND EXPERIMENT - ensuring novelty
        second_experiment_prompt = f"""
        You are an expert experimental researcher designing a SECOND experiment that complements the first but uses different approaches from the literature. You can combine techniques from the provided papers in novel ways to create a distinct experimental approach.

        FLEXIBLE LITERATURE GROUNDING FOR SECOND EXPERIMENT:
        - Use different models, datasets, or techniques than the first experiment
        - Combine approaches from the literature in novel ways
        - Reference specific papers for all experimental components
        - Create experiments that explore different aspects of the research questions
        - Build upon the literature while exploring alternative approaches

        CRITICAL REQUIREMENTS FOR SECOND EXPERIMENT:
        - MUST be fundamentally different from the first experiment in approach or methodology
        - Every experiment MUST reference at least 2 different papers from the provided literature
        - Models and datasets MUST be cited from the papers
        - Should address different aspects of the key research questions
        - Include all required sections: Objective, Hypothesis, Methodology, Expected Outcomes, Success Metrics

        FIRST EXPERIMENT (DO NOT REPEAT OR BE SIMILAR TO THIS):
        {first_experiment}

        Original Research Context:
        {original_prompt}

        Research Direction:
        {json.dumps(research_direction, indent=2)}

        Findings Analysis:
        {json.dumps(findings_analysis, indent=2)}

        LITERATURE CONTEXT (MANDATORY REFERENCE MATERIAL):
        {papers_context}

        {improvement_guidance}

        EXPERIMENT DESIGN REQUIREMENTS FOR SECOND EXPERIMENT:

        1. **LITERATURE GROUNDING**: Every experiment component MUST reference specific content from the provided papers
        2. **MODEL CITATIONS**: Every model mentioned MUST include original paper citation
        3. **DATASET CITATIONS**: Every dataset MUST be real with proper academic citation
        4. **TECHNIQUE VALIDATION**: Only use techniques explicitly described in the literature
        5. **PAPER INTEGRATION**: Reference at least 2 different papers for this experiment
        6. **NOVELTY REQUIREMENT**: This experiment MUST be fundamentally different from the first experiment in approach, methodology, models, or research focus
        7. **DIFFERENT LITERATURE BASIS**: Use different papers or different aspects of the same papers than the first experiment

        Generate ONE detailed experiment that explores a completely different angle or approach:

        ### Experiment 2: [Experiment Title - Reference specific paper technique - DIFFERENT from Experiment 1]
        **Objective**: Clear statement grounded in literature findings (different focus from Experiment 1)
        **Hypothesis**: Testable hypothesis based on literature insights (different from Experiment 1)
        **Methodology**:
        - Use ONLY techniques from provided papers (different techniques than Experiment 1)
        - Cite specific papers for each methodological choice (different papers/aspects than Experiment 1)
        - Reference exact models/datasets from literature (different from Experiment 1)
        **Expected Outcomes**: Based on literature expectations (different outcomes than Experiment 1)
        **Success Metrics**: Metrics mentioned in the papers (different metrics than Experiment 1)
        **Resources Needed**: Based on literature resource requirements
        **Risk Assessment**: Risks identified in the literature
        **Literature References**: Specific papers supporting this experiment (different papers than Experiment 1)

        CRITICAL REQUIREMENTS:
        - Your experiment MUST be 100% grounded in the provided literature
        - KEEP RESPONSE TO 500 WORDS MAXIMUM
        - Your experiment MUST be NOVEL and DIFFERENT from the first experiment
        - If you cannot design a sufficiently different experiment using ONLY the content from these papers, state this explicitly rather than creating a similar experiment
        """

        _write_stream("Generating second experiment (ensuring novelty)...")
        response2 = client.chat.completions.create(
            model=model,
            temperature=0.4,  # Slightly higher temperature for more creativity in second experiment
           # max_tokens=4000,
            messages=[{"content": second_experiment_prompt, "role": "user"}]
        )

        second_experiment = response2.choices[0].message.content

        # Ensure we have a valid string response
        if second_experiment is None:
            raise ValueError("LLM returned None response for second experiment")

        second_experiment = str(second_experiment)
        _write_stream("second experiment generated successfully")

        # Combine both experiments into final suggestions
        experiment_suggestions = f"""
# Experiment Design Recommendations

## Primary Experiments

{first_experiment}

{second_experiment}

## Quality Control Measures

- Statistical methods from the literature
- Reproducibility requirements mentioned in papers
- Validation procedures from literature

## Expected Timeline and Deliverables

- Timeline based on literature implementation times
- Deliverables aligned with literature outcomes
"""

        _write_stream("Both experiments combined successfully")
        
        state["experiment_suggestions"] = experiment_suggestions
        state["suggestion_source"] = "llm_generated"
        state["current_experiment_iteration"] = state.get("current_experiment_iteration", 0) + 1
        
        # Create summary with error handling
        try:
            total_experiments = len(re.findall(r'### Experiment \d+:', experiment_suggestions))
        except (TypeError, AttributeError) as e:
            print(f"⚠️ Error counting experiments: {e}")
            total_experiments = 0
        
        state["experiment_summary"] = {
            "total_experiments": total_experiments,
            "research_direction": research_direction.get("recommended_direction", {}).get("title", "Unknown"),
            "papers_used": len(validated_papers),
            "generation_successful": True,
            "iteration": state["current_experiment_iteration"]
        }
        
        _write_stream("Experiment suggestions generated successfully")
        _write_stream(f"Experiments designed: {state['experiment_summary']['total_experiments']}")
        
        # Set next node for workflow routing
        state["next_node"] = "validate_experiments_tree_2"
        
        # Add success message
        state["messages"].append(
            AIMessage(content=f"Generated two distinct experiment suggestions with {state['experiment_summary']['total_experiments']} detailed experiments, ensuring novelty between approaches.")
        )
        
    except Exception as e:
        error_msg = f"Experiment suggestion generation failed: {str(e)}"
        print(f"Error: {error_msg}")
        state["errors"].append(error_msg)
        
        # Create fallback suggestions
        state["experiment_suggestions"] = f"""
        # Experiment Design Recommendations

        ## Primary Experiment

        ### Experiment 1: Follow-up Validation Study
        **Objective**: Validate and extend the findings from the original research
        **Hypothesis**: The observed results can be reproduced and generalized
        **Methodology**: 
        - Replicate original experimental setup with variations
        - Expand dataset or parameter ranges
        - Apply statistical validation methods
        **Expected Outcomes**: Confirmation of original findings with extended insights
        **Success Metrics**: Reproducibility score > 0.8, statistical significance p < 0.05
        **Resources Needed**: Similar to original experiment setup
        **Risk Assessment**: Low risk, builds directly on existing work

        ## Implementation Roadmap

        **Phase 1**: Setup and replication (2 weeks)
        **Phase 2**: Extension experiments (2 weeks)  
        **Phase 3**: Analysis and validation (1 week)

        Note: This is a fallback experiment design. For detailed recommendations, please provide more specific experimental context.
        """
        
        state["experiment_summary"] = {
            "total_experiments": 1,
            "research_direction": "Fallback validation study",
            "papers_used": 0,
            "generation_successful": False,
            "iteration": 1
        }
        
        # Set next node even for fallback
        state["next_node"] = "validate_experiments_tree_2"
    
    return state


def _validate_experiments_tree_2_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Validate the generated experiment suggestions and decide whether to finalize or iterate."""
    _write_stream("Validating experiment suggestions.")
    state["current_step"] = "validate_experiments_tree_2"
    
    # Add infinite loop protection
    current_iteration = state.get("current_experiment_iteration", 0)
    if current_iteration >= 7:  # Higher safety limit
        _write_stream(f"Maximum experiment iterations reached ({current_iteration}). Forcing completion...")
        state["next_node"] = "END"
        state["experiments_validation_decision"] = "FORCE_PASS"
        return state
    experiment_suggestions = state.get("experiment_suggestions", "")
    
    if not experiment_suggestions:
        _write_stream("Experiment suggestions validation failed - no suggestions")
        state["next_node"] = "suggest_experiments_tree_2"
        state["validation_feedback"] = "No experiment suggestions generated"
        return state
    
    try:
        # Extract context for LLM validation
        experiment_summary = state.get("experiment_summary", {})
        total_experiments = experiment_summary.get("total_experiments", 0)
        papers_used = experiment_summary.get("papers_used", 0)
        
        # Get research direction and domain context for comprehensive validation
        research_direction = state.get("research_direction", {})
        findings_analysis = state.get("findings_analysis", {})
        validated_papers = state.get("validated_experiment_papers", [])
        original_prompt = state.get("original_prompt", "")

        # DEBUG: Check validated_papers state
       
        # Extract research direction details
        selected_direction = research_direction.get("selected_direction", {})
        direction_text = selected_direction.get("direction", "")
        key_questions = selected_direction.get("key_questions", [])
        direction_justification = selected_direction.get("justification", "")
        expected_impact = selected_direction.get("expected_impact", "")
        
        # Extract domain information
        domain_analysis = findings_analysis.get("domain_analysis", {})
        primary_domain = domain_analysis.get("primary_domain", "machine learning")
        task_type = domain_analysis.get("task_type", "")
        application_area = domain_analysis.get("application_area", "")
        model = state.get("model", "gemini/gemini-2.5-flash")
        model="gemini/gemini-2.5-flash-lite"
        client = state.get("client")
        
        # Build literature context from validated papers (use same format as original generation)
        literature_context = ""
        if validated_papers:
            literature_parts = []
            for i, paper in enumerate(validated_papers[:5], 1):
                title = paper.get("title", f"Paper {i}")
                abstract = paper.get("summary", paper.get("abstract", ""))
                content = paper.get("content", "")
                
                paper_content = f"**PAPER {i}: {title}**\n"
                
                if content:
                    paper_content += f"Key Content: {content}\n"
                
                literature_parts.append(paper_content)
            
            literature_context = "\n\n".join(literature_parts)
        
        # Get current iteration to match original generation prompt format
        current_iteration = state.get("current_experiment_iteration", 1)
        iteration_context = f"ITERATION {current_iteration}" if current_iteration > 1 else "INITIAL GENERATION"
        
        # SIMPLIFIED LLM-BASED VALIDATION
        validation_prompt = f"""
                You are an expert experimental methodology validator. Evaluate the proposed experiments for technical soundness, literature grounding, and research alignment.

                **RESEARCH CONTEXT:**
                - Direction: {direction_text}
                - Key Questions: {chr(10).join(f"• {q}" for q in key_questions[:3])}
                - Original Request: {original_prompt}

                **LITERATURE CONTEXT:**
                You have {len(validated_papers)} research papers that MUST be the foundation for all experiments:

                {literature_context}

                **PROPOSED EXPERIMENTS:**
                {experiment_suggestions}

                **VALIDATION REQUIREMENTS:**

                1. **LITERATURE GROUNDING (40% weight):**
                   - Every experiment component MUST reference specific content from the provided papers
                   - Use exact methodologies, datasets, and models mentioned in the literature
                   - Cite papers properly (e.g., "as described in Paper 1" or specific technique names)
                   - NO external techniques or models not mentioned in the papers

                2. **RESEARCH ALIGNMENT (35% weight):**
                   - Experiments MUST directly address the research direction: {direction_text}
                   - Each experiment should help answer the key questions
                   - Stay within the scope of transfer learning and fine-tuning strategies

                3. **TECHNICAL SOUNDNESS (25% weight):**
                   - Clear objectives, hypotheses, and methodologies
                   - Realistic resource requirements and timelines
                   - Proper evaluation metrics and success criteria
                   - Implementable experimental procedures

                **MANDATORY REQUIREMENTS:**
                - Every model/dataset MUST be cited from the provided papers
                - Experiments must include: Objective, Hypothesis, Methodology, Expected Outcomes, Success Metrics
                - At least 2 experiments must be proposed
                - Experiments should build upon the literature rather than require groundbreaking novelty

                **NOVELTY ASSESSMENT:**
                - For literature-grounded experiments, focus on practical combinations and extensions of documented methods
                - Value lies in systematic application and comparison of established approaches
                - Reasonable novelty comes from combining different techniques or applying them to new contexts within the literature

                Return your assessment in this exact JSON format:
                {{
                    "validation_result": "PASS" | "FAIL",
                    "overall_score": 0.0-1.0,
                    "detailed_scores": {{
                        "research_direction_alignment": 0.0-1.0,
                        "novelty_potential": 0.0-1.0,
                        "justification_quality": 0.0-1.0
                    }},
                    "critical_issues": ["list", "of", "critical", "problems"],
                    "direction_misalignment": ["ways", "experiments", "dont", "align"],
                    "novelty_concerns": ["lack", "of", "novelty", "issues"],
                    "improvement_recommendations": ["specific", "actionable", "suggestions"],
                    "decision_rationale": "Clear explanation of pass/fail decision"
                }}

                IMPORTANT: Respond with ONLY the JSON object, nothing else.
                """

        _write_stream("Starting LLM-based hyper-strict validation...")
        
       # print('VALIDAITON rompt: '+ validation_prompt)
        #print ("===="*30)
        
        # Initialize default values for error handling
        validation_result = "FAIL"
        overall_score = 0.0
        detailed_scores = {"research_direction_alignment": 0.0, "novelty_potential": 0.0, "justification_quality": 0.0}
        critical_issues = ["Validation initialization failed"]
        direction_misalignment = []
        novelty_concerns = []
        improvement_recommendations = ["Fix validation system", "Regenerate experiments"]
        decision_rationale = "Validation failed to initialize"
        validation_json = {}
        
        # SPECIAL HANDLING FOR NO PAPERS CASE
        if len(validated_papers) == 0:
            print(f"⚠️ No validated papers available for literature context. Using simplified validation criteria...")
            
            # Check if experiments have basic required sections
            experiment_content = experiment_suggestions.lower()
            required_sections = ["objective", "methodology", "resource", "timeline"]
            sections_found = sum(1 for section in required_sections if section in experiment_content)
            
            # Basic content validation
            has_experiments = len(re.findall(r'experiment \d+', experiment_content)) >= 2
            has_reasonable_length = len(experiment_suggestions) > 1000
            has_methodology = any(word in experiment_content for word in ['dataset', 'model', 'training', 'evaluation'])
            
            basic_score = 0.0
            if has_experiments:
                basic_score += 0.3
            if has_reasonable_length:
                basic_score += 0.2
            if has_methodology:
                basic_score += 0.3
            if sections_found >= 3:
                basic_score += 0.2
            
            print(f"📊 Basic validation score: {basic_score:.2f}/1.0")
            print(f"   - Has multiple experiments: {has_experiments}")
            print(f"   - Has reasonable length: {has_reasonable_length}")
            print(f"   - Has methodology: {has_methodology}")
            print(f"   - Required sections found: {sections_found}/4")
            
            if basic_score >= 0.7:  # Lower threshold when no papers available
                _write_stream("Experiment suggestions pass basic validation (no papers scenario)")
                overall_score = basic_score
                validation_result = "PASS"
                critical_issues = []
                improvement_recommendations = ["Add more baseline comparisons", "Include ablation studies", "Consider hyperparameter optimization"]
                decision_rationale = f"Basic validation passed with score {basic_score:.2f}/1.0 (no literature papers available)"
                validation_json = {
                    "validation_result": "PASS",
                    "overall_score": basic_score,
                    "detailed_scores": {"research_direction_alignment": basic_score, "novelty_potential": basic_score, "justification_quality": basic_score},
                    "critical_issues": [],
                    "direction_misalignment": [],
                    "novelty_concerns": [],
                    "improvement_recommendations": improvement_recommendations,
                    "decision_rationale": decision_rationale
                }
            else:
                _write_stream(f"Experiment suggestions failed basic validation - Score: {basic_score:.2f}/1.0")
                validation_result = "FAIL"
                overall_score = basic_score
                critical_issues = ["Insufficient experiment detail", "Missing required sections"]
                improvement_recommendations = ["Add more detailed methodology", "Include all required roadmap sections", "Expand experiment descriptions"]
                decision_rationale = f"Basic validation failed with score {basic_score:.2f}/1.0"
                validation_json = {
                    "validation_result": "FAIL",
                    "overall_score": basic_score,
                    "detailed_scores": {"research_direction_alignment": basic_score, "novelty_potential": basic_score, "justification_quality": basic_score},
                    "critical_issues": critical_issues,
                    "direction_misalignment": [],
                    "novelty_concerns": [],
                    "improvement_recommendations": improvement_recommendations,
                    "decision_rationale": decision_rationale
                }
        else:
            # Standard LLM validation when papers are available
            # Call LLM for validation (access client from state)
            try:
                client = state.get('client')
                if not client:
                    print("OpenAI client not found in state")
                    return state
                    
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    max_tokens=8000,
                    messages=[
                        {"role": "system", "content": f"You are a ruthlessly strict experimental methodology validator specializing in {primary_domain}. Respond with VALID JSON only - no markdown, no explanations, just the JSON object with all required fields."},
                        {"role": "user", "content": validation_prompt}
                    ]
                )
                
                validation_response = response.choices[0].message.content.strip()
                
                # Remove markdown formatting if present
                if validation_response.startswith("```json"):
                    validation_response = validation_response[7:]
                if validation_response.endswith("```"):
                    validation_response = validation_response[:-3]
                validation_response = validation_response.strip()
                
                # Parse validation JSON
                validation_json = json.loads(validation_response)
                
                # Extract results
                validation_result = validation_json.get("validation_result", "FAIL").upper()
                overall_score = float(validation_json.get("overall_score", 0.0))
                detailed_scores = validation_json.get("detailed_scores", detailed_scores)
                critical_issues = validation_json.get("critical_issues", [])
                direction_misalignment = validation_json.get("direction_misalignment", [])
                novelty_concerns = validation_json.get("novelty_concerns", [])
                improvement_recommendations = validation_json.get("improvement_recommendations", ["Improve experiment design"])
                decision_rationale = validation_json.get("decision_rationale", "")
                
                # MODERATE VALIDATION ENFORCEMENT for literature-grounded experiments
                min_score_met = all(score >= 0.75 for score in detailed_scores.values())  # Increased from 0.70
                if (overall_score < 0.80 or  # Increased from 0.75
                    not min_score_met or
                    len(critical_issues) > 1):  # Reduced from 2 to be stricter
                    validation_result = "FAIL"
               
                _write_stream(f"Validation Result: {validation_result}")
                
                for dimension, score in detailed_scores.items():
                    status = "PASS" if score >= 0.85 else "FAIL"
                    _write_stream(f"   {dimension.replace('_', ' ').title()}: {score:.2f}/1.0 {status}")
                

            except (json.JSONDecodeError, Exception) as e:
                print(f"LLM validation failed: {str(e)}")
                validation_result = "FAIL"
                overall_score = 0.0
                validation_json = {
                    "validation_result": "FAIL",
                    "overall_score": 0.0,
                    "detailed_scores": {"research_direction_alignment": 0.0, "novelty_potential": 0.0, "justification_quality": 0.0},
                    "critical_issues": ["LLM validation parsing failed"],
                    "direction_misalignment": [],
                    "novelty_concerns": [],
                    "improvement_recommendations": ["Fix validation system", "Regenerate experiments"],
                    "decision_rationale": f"Validation failed due to error: {str(e)}"
                }

        # STRICT VALIDATION CRITERIA with LLM results and special handling for no papers
        papers_available = len(validated_papers) > 0

        # DEBUG: Log validation state
        #print(f"🔍 VALIDATION DEBUG: papers_available={papers_available}, validation_result='{validation_result}', overall_score={overall_score:.3f}, current_iteration={current_iteration}")

        # Adjust thresholds based on paper availability - MODERATE STRICTNESS for literature-grounded experiments
        if papers_available:
            min_score_required = 0.80  # Increased from 0.75 for better quality
            pass_condition = validation_result == "PASS" and overall_score >= min_score_required
        else:
            min_score_required = 0.70  # Slightly increased from 0.65
            pass_condition = validation_result == "PASS" and overall_score >= min_score_required
        
        if pass_condition:
            context_note = "with literature context" if papers_available else "without literature context (relaxed criteria)"
            _write_stream(f"Experiment suggestions validation PASSED (LLM score: {overall_score:.2f}/{min_score_required}, {context_note})")
            state["next_node"] = "END"
            state["experiments_validation_decision"] = "PASS"
            
            # Create final outputs with LLM validation results
            final_outputs = {
                "markdown": experiment_suggestions,
                "summary": f"""
# Experiment Suggestion Summary

**Research Direction**: {experiment_summary.get('research_direction', 'Unknown')}
**Total Experiments Designed**: {total_experiments}
**Papers Referenced**: {papers_used}
**LLM Validation Score**: {overall_score:.2f}/1.0
**Validation Status**: {validation_result}
**Generation Status**: Success

## LLM Validation Results
- Overall Score: {overall_score:.2f}/1.0 (Required: ≥0.80)
- Research Direction Alignment: {detailed_scores.get('research_direction_alignment', 0):.2f}/1.0
- Novelty Potential: {detailed_scores.get('novelty_potential', 0):.2f}/1.0
- Justification Quality: {detailed_scores.get('justification_quality', 0):.2f}/1.0

## Quick Overview
{experiment_suggestions[:500]}...

*For complete experiment details, see the full markdown output.*
                """.strip(),
                "json": json.dumps({
                    "experiment_suggestions": experiment_suggestions,
                    "summary": experiment_summary,
                    "llm_validation_results": validation_json,
                    "metadata": {
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "iteration": state.get("current_experiment_iteration", 1),
                        "total_papers_found": len(state.get("experiment_papers", [])),
                        "validated_experiment_papers": len(state.get("validated_experiment_papers", []))
                    }
                }, indent=2)
            }
            
            state["final_outputs"] = final_outputs
            
            # Add completion message
            state["messages"].append(
                AIMessage(content=f"Experiment suggestion workflow completed successfully. LLM validation score: {overall_score:.2f}/1.0. Comprehensive experiment designs are ready for implementation.")
            )
        elif current_iteration < 5:  # Allow more retries for experiment generation
            threshold_note = f"minimum {min_score_required:.2f} required" + (" (relaxed - no papers)" if not papers_available else " (strict - with papers)")
            _write_stream(f"Experiment suggestions failed LLM validation - Score: {overall_score:.2f}/1.0 ({threshold_note})")
            state["next_node"] = "suggest_experiments_tree_2"
            state["experiments_validation_decision"] = "RETRY"
            
            # Provide detailed LLM feedback for improvement
            feedback_parts = [f"LLM validation score: {overall_score:.2f}/1.0 (need 0.90+)"]
            if critical_issues:
                feedback_parts.append(f"Critical issues: {'; '.join(critical_issues[:3])}")
            if direction_misalignment:
                feedback_parts.append(f"Direction misalignment: {'; '.join(direction_misalignment[:2])}")
            if novelty_concerns:
                feedback_parts.append(f"Novelty concerns: {'; '.join(novelty_concerns[:2])}")
            if improvement_recommendations:
                feedback_parts.append(f"Recommendations: {'; '.join(improvement_recommendations[:3])}")
            
            current_feedback = " | ".join(feedback_parts)
            state["validation_feedback"] = current_feedback
            
            # ACCUMULATE PAST MISTAKES FOR LEARNING
            past_mistakes = state.get("past_experiment_mistakes", [])
            
            # Create detailed mistake record for this iteration
            mistake_record = {
                "iteration": current_iteration,  # The iteration that just failed validation
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "validation_score": overall_score,
                "critical_issues": critical_issues,
                "direction_misalignment": direction_misalignment,
                "novelty_concerns": novelty_concerns,
                "improvement_recommendations": improvement_recommendations,
                "experiment_summary": experiment_summary,
                "feedback_summary": current_feedback
            }
            
            past_mistakes.append(mistake_record)
            state["past_experiment_mistakes"] = past_mistakes
            
            #print(f" Accumulated {len(past_mistakes)} past mistake records for learning")
            
        else:
            # Force pass after 3 iterations
            print(f"⚠️ Maximum iterations reached. Forcing completion with LLM score: {overall_score:.2f}/1.0")
            state["next_node"] = "END"
            state["experiments_validation_decision"] = "FORCE_PASS"
            
            # Create outputs even for forced pass
            state["final_outputs"] = {
                "markdown": experiment_suggestions,
                "summary": f"Experiment suggestions generated but failed LLM validation (Score: {overall_score:.2f}/1.0) - Forced completion after 3 iterations",
                "llm_validation_results": validation_json,
                "forced_completion": True
            }
        
    except Exception as e:
        error_msg = f"Experiment validation failed: {str(e)}"
        print(f"❌ {error_msg}")
        state["errors"].append(error_msg)
        state["next_node"] = "END"  # Exit on error
        state["experiments_validation_decision"] = "ERROR"
    
    return state

