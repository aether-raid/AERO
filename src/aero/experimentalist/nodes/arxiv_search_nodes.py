from ..shared_defs import ExperimentSuggestionState, _write_stream , create_custom_ranking_prompt, _clean_text_for_utf8
import json
import urllib.request as libreq
import xml.etree.ElementTree as ET
from ...utils.arxiv import format_search_string
from concurrent.futures import ThreadPoolExecutor, as_completed

def _generate_experiment_search_query_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Generate ArXiv search query for domain-specific experimental guidance papers."""
    search_iteration = state.get("experiment_search_iteration", 0)
    validation_results = state.get("experiment_paper_validation_results", {})
    is_search_new = validation_results.get("decision") == "search_new"
    
    if search_iteration == 0:
        _write_stream("Experiment Search Query: Generating targeted search for experimental guidance.")
    else:
        _write_stream(f"Experiment Search Query (Retry {search_iteration + 1}): Generating refined search based on validation feedback...")
        
    model = state["model"]
    client = state["client"]
    try:
        # Extract context
        original_prompt = state.get("original_prompt", "")
        research_direction = state.get("research_direction", {})
        findings_analysis = state.get("findings_analysis", {})
        
        selected_direction = research_direction.get("selected_direction", {})
        direction_text = selected_direction.get("direction", "")
        key_questions = selected_direction.get("key_questions", [])
        
        # Extract domain information from analysis
        domain_analysis = findings_analysis.get("domain_analysis", {})
        primary_domain = domain_analysis.get("primary_domain", "machine learning")
        task_type = domain_analysis.get("task_type", "")
        application_area = domain_analysis.get("application_area", "")
        data_type = domain_analysis.get("data_type", "")
        
        # Get previous search query and validation feedback for retries
        previous_query = state.get("experiment_search_query", "")
        search_guidance = {}
        
        if is_search_new and validation_results.get("validation_data"):
            search_guidance = validation_results["validation_data"].get("search_guidance", {})
            _write_stream("Using validation feedback to generate improved search query...")
        
        # Generate domain-specific search query with conditional previous query info
        query_prompt = f"""
        Generate a focused ArXiv search query to find papers in the same research domain that contain experimental methodologies and guidance.

        RESEARCH DOMAIN: {primary_domain}
        TASK TYPE: {task_type}
        APPLICATION: {application_area}
        DATA TYPE: {data_type}
        
        ORIGINAL RESEARCH: {original_prompt}
        
        RESEARCH DIRECTION: {direction_text}
        
        KEY QUESTIONS: {chr(10).join(f"• {q}" for q in key_questions[:3])}"""
        
        # Only add previous query info if it exists and this is a retry
        if previous_query and search_iteration > 0:
            query_prompt += f"""
            
        PREVIOUS FAILED QUERY: {previous_query}
        SEARCH ITERATION: {search_iteration + 1}
        
        VALIDATION FEEDBACK:
        {validation_results.get("reasoning", "Previous papers lacked sufficient experimental methodology")}
        
        SEARCH GUIDANCE FROM VALIDATION:
        - New search terms: {search_guidance.get('new_search_terms', [])}
        - Focus areas: {search_guidance.get('focus_areas', [])}
        - Avoid terms: {search_guidance.get('avoid_terms', [])}
        
        Generate 4 DIFFERENT search terms for ArXiv API, separated by forward slashes (/):
        
        Rules for RETRY query:
        - Term 1: Different primary technique than "{previous_query.split('/')[0] if '/' in previous_query else ''}"
        - Term 2: Alternative task perspective or methodology focus
        - Term 3: Stronger experimental focus (e.g., "experimental validation", "empirical study", "systematic evaluation")
        - Term 4: Specific experimental context (e.g., "ablation", "benchmark", "comparison study")
        
        Focus on finding papers with STRONGER experimental methodology than the previous search.
        Use the suggested new search terms if provided: {search_guidance.get('new_search_terms', [])}"""
        else:
            query_prompt += """

        Generate 4 search terms for ArXiv API, separated by forward slashes (/), that will find papers in the SAME DOMAIN with experimental guidance:
        
        Rules:
        - Term 1: Primary domain-specific technique or model (e.g., "YOLO", "transformer", "CNN", "LSTM")
        - Term 2: Specific task type (e.g., "object detection", "classification", "segmentation")  
        - Term 3: Experimental aspect (e.g., "ablation study", "evaluation", "comparison", "benchmark")
        - Term 4: Domain/application context (e.g., "autonomous driving", "medical imaging", "NLP")
        
        Examples:
        - For computer vision: "YOLO/object detection/ablation study/autonomous driving"
        - For NLP: "transformer/text classification/evaluation/sentiment analysis"
        - For medical AI: "CNN/medical imaging/comparison/radiology"
        
        SPECIAL CASE FOR SELF-SUPERVISED LEARNING: If the research direction involves self-supervised learning, use terms like:
        - "SimCLR/contrastive learning/CIFAR-10/pretraining"
        - "BYOL/self-supervised learning/image classification/fine-tuning"
        - "MAE/masked autoencoder/CIFAR-10/generalization"
        - "MoCo/momentum contrast/CIFAR-10/representation learning"
        
        Focus on finding papers that will have similar experimental setups and methodologies, NOT generic methodology papers.
        Return ONLY the 4-term query string (no explanation).
        """
    
        # Use higher temperature for retries to get different results
        temperature = 0.3 if search_iteration > 0 else 0.1
        
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "Generate focused domain-specific ArXiv search queries. For retries, ensure the new query is significantly different from previous attempts."},
                {"role": "user", "content": _clean_text_for_utf8(query_prompt)}
            ]
        )

        search_query = response.choices[0].message.content.strip()
        
        # Clean the query (remove quotes, extra spaces)
        search_query = search_query.replace('"', '').replace("'", "").strip()
        
        # Validate that retry query is actually different
        if search_iteration > 0 and previous_query and search_query == previous_query:
            _write_stream("Generated query is identical to previous - creating fallback different query...")
            
            # Create a guaranteed different query
            base_terms = previous_query.split('/') if '/' in previous_query else ["machine learning", "experimental", "evaluation", "methodology"]
            
            # Alternative search strategies for different domains
            if "computer vision" in primary_domain.lower() or "cv" in primary_domain.lower():
                alternative_queries = [
                    "ResNet/image classification/systematic evaluation/ImageNet",
                    "convolutional neural network/experimental analysis/performance comparison/computer vision",
                    "deep learning/ablation study/benchmark evaluation/visual recognition"
                ]
            elif "nlp" in primary_domain.lower() or "text" in primary_domain.lower():
                alternative_queries = [
                    "BERT/text classification/experimental evaluation/NLP",
                    "transformer/language model/systematic comparison/text analysis",
                    "neural language processing/empirical study/benchmark/natural language"
                ]
            else:
                alternative_queries = [
                    "neural network/experimental methodology/empirical evaluation/machine learning",
                    "deep learning/systematic study/performance analysis/artificial intelligence",
                    "machine learning/experimental validation/comparative study/methodology"
                ]
            
            # Pick different query based on iteration
            search_query = alternative_queries[search_iteration % len(alternative_queries)]
        
        # Ensure it has the right format (4 terms separated by /)
        if search_query.count('/') != 3:
            # Fallback: create domain-specific query from extracted info
            term1 = task_type or primary_domain
            term2 = "experimental" if not task_type else task_type
            term3 = "evaluation"
            term4 = application_area or primary_domain
            search_query = f"{term1}/{term2}/{term3}/{term4}"
        
        _write_stream(f"Generated {'refined' if search_iteration > 0 else 'initial'} search query: {search_query}")
        if search_iteration > 0 and previous_query:
            _write_stream(f"Previous query was: {previous_query}")
        
        return {
            **state,
            "experiment_search_query": search_query,
            "experiment_search_domain": primary_domain,
            "experiment_search_task": task_type,
            "current_step": "search_query_generated"
        }
        
    except Exception as e:
        print(f"Error generating experiment search query: {str(e)}")
        # Enhanced fallback for retries
        prompt_lower = original_prompt.lower()
        direction_lower = direction_text.lower()
        
        if search_iteration > 0:
            fallback_queries = [
                "experimental methodology/systematic evaluation/empirical study/research methodology",
                "performance analysis/comparative study/experimental validation/machine learning",
                "ablation study/experimental design/empirical analysis/systematic comparison"
            ]
            fallback_query = fallback_queries[search_iteration % len(fallback_queries)]
        elif "self-supervised" in direction_lower or "ssl" in direction_lower:
            fallback_query = "SimCLR/contrastive learning/CIFAR-10/pretraining"
        elif "object detection" in prompt_lower or "detection" in prompt_lower:
            fallback_query = "object detection/evaluation/experimental/computer vision"
        elif "classification" in prompt_lower:
            fallback_query = "classification/experimental/evaluation/machine learning"
        elif "segmentation" in prompt_lower:
            fallback_query = "segmentation/experimental/evaluation/computer vision"
        elif "nlp" in prompt_lower or "text" in prompt_lower:
            fallback_query = "text classification/experimental/evaluation/NLP"
        else:
            fallback_query = "machine learning/experimental/evaluation/methodology"
        
        return {
            **state,
            "experiment_search_query": fallback_query,
            "experiment_search_domain": "machine learning",
            "errors": state.get("errors", []) + [f"Search query generation error: {str(e)}"],
            "current_step": "search_query_error"
        }

async def _search_experiment_papers_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Search ArXiv for experimental methodology papers using optimized workflow."""
    search_iteration = state.get("experiment_search_iteration", 0)
    validation_results = state.get("experiment_paper_validation_results", {})
    is_backup_search = validation_results.get("decision") == "search_backup"
    
    if search_iteration == 0:
        _write_stream("Experiment Papers Search: Searching ArXiv for experimental guidance...")
    elif is_backup_search:
        _write_stream("Experiment Search (Backup): Searching for additional experimental papers...")
    else:
        _write_stream(f"Experiment Search (New Search {search_iteration + 1}): Searching with refined query...")
        
    state["current_step"] = "search_experiment_papers"
    
    # Import required modules for ArXiv search
    
    
    # Initialize variables
    papers = []
    total_results = 0
    formatted_query = ""
    
    # For backup searches, preserve existing papers
    existing_papers = []
    if is_backup_search and state.get("experiment_papers"):
        existing_papers = state["experiment_papers"]
        _write_stream(f"Preserving {len(existing_papers)} papers from previous search")
    
    # Safety check: After 3 iterations, force continue to avoid infinite loops
    if search_iteration >= 3:
        _write_stream(f"Maximum search iterations reached ({search_iteration}). Proceeding with existing papers...")
        state["experiment_papers"] = existing_papers if existing_papers else []
        state["experiment_search_completed"] = True
        return state
    
    try:
        search_query = state.get("experiment_search_query", "experimental methodology")
        research_direction = state.get("research_direction", {})
        original_prompt = state.get("original_prompt", "")
        
        # Use ArXiv processor for paper processing
        arxiv_processor = state["arxiv_processor"]
        if not arxiv_processor:
            raise Exception("ArXiv processor not available")
        
        # Determine search parameters based on search type and iteration
        if search_iteration == 0:
            # Initial search: get 100 papers for ranking
            max_results = 100
            start_offset = 0
        elif is_backup_search:
            # Backup search: get additional papers with offset
            existing_count = len(existing_papers) if existing_papers else 0
            start_offset = max(100, existing_count)
            max_results = 100
        else:
            # New search with different query: get 100 fresh papers
            max_results = 100
            start_offset = 0
        
        
        
        # Format search query and build URL
        formatted_query = format_search_string(search_query)
        url = f"http://export.arxiv.org/api/query?search_query={formatted_query}&start={start_offset}&max_results={max_results}"
        
        #_write_stream(f"Formatted query: {formatted_query}")
        _write_stream(f"Full search URL: {url}")
        
        # Fetch and parse ArXiv results
        with libreq.urlopen(url) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
        }
        
        # Get total results
        total_results_elem = root.find('opensearch:totalResults', ns)
        total_results = int(total_results_elem.text) if total_results_elem is not None else 0
        
        _write_stream(f"Total papers found: {total_results}")
        
        if total_results == 0:
            print("⚠️ No papers found for experiment search query")
            return {
                **state,
                "experiment_papers": existing_papers,
                "current_step": "no_papers_found"
            }
        
        # Extract paper entries
        entries = root.findall('atom:entry', ns)
        if len(entries) == 0:
            entries = root.findall('.//entry')  # Fallback without namespace
        
        _write_stream(f"Processing {len(entries)} paper entries...")
        
        # Stage 1: Extract basic info (title, abstract, metadata) without downloading PDFs
        _write_stream(f"Stage 1: Extracting basic info for {len(entries)} experimental papers...")
        papers = []
        for i, entry in enumerate(entries, 1):
            try:
                paper_info = arxiv_processor.extract_basic_paper_info(entry, ns, i)
                papers.append(paper_info)
               # print(f"✅ Basic info extracted for paper #{i}: {paper_info['title'][:50]}...")
            except Exception as e:
                print(f"⚠️ Error processing paper entry {i}: {e}")
                continue
        
        # Stage 2: Rank papers by relevance using enhanced analysis context
        _write_stream(f"Ranking experimental papers by relevance using extracted analysis...")
        
        # Create enhanced ranking context from the analysis findings
        # Note: These are utility functions that should be called as standalone functions
        ranking_context = create_experiment_ranking_context_from_analysis(state)
        #print(f"📊 Using enhanced context for ranking: {ranking_context[:100]}...")
        
        # Create custom prompt for experimental ranking
        custom_prompt = create_custom_ranking_prompt("experimental")
        
        papers = await arxiv_processor.rank_papers_by_relevance(papers, ranking_context, custom_prompt)
        
        # Stage 3: Download full content for top 5 papers only
        top_papers = papers[:5]  # Get top 5 papers
        
        _write_stream(f"Downloading full PDF content for top {len(top_papers)} experimental papers...")

        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrent downloads
            # Submit download tasks for top papers only
            future_to_paper = {
                executor.submit(arxiv_processor.download_paper_content, paper): paper 
                for paper in top_papers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                updated_paper = future.result()
                # Update the paper in the original list
                for i, paper in enumerate(papers):
                    if paper['id'] == updated_paper['id']:
                        papers[i] = updated_paper
                        break
        
        _write_stream(f"PDF download stage completed for experimental papers.")
        ''''
        # Print ranked results
        print("\n" + "=" * 80)
        print("📋 RANKED EXPERIMENTAL PAPERS (by relevance):")
        print("=" * 80)
        
        for i, paper in enumerate(papers[:5], 1):  # Show top 5
            relevance_score = paper.get('relevance_score', 0)
            has_content = paper.get('pdf_downloaded', False)
            content_status = "📄 FULL CONTENT" if has_content else "📝 TITLE+ABSTRACT"
            
            print(f"\n📄 EXPERIMENTAL PAPER #{i} ({content_status}) - Relevance: {relevance_score:.1f}/10.0")
            print("-" * 60)
            print(f"Title: {paper['title']}")
            print(f"ID: {paper['id']}")
            print(f"Published: {paper['published']}")
            print(f"URL: {paper['url']}")
            
            if paper.get('summary'):
                print(f"Summary: {paper['summary'][:300]}...")
            if has_content and paper.get('content'):
                print(f"Content Snippet: {paper['content'][:500]}...")
            
            print("-" * 60)
        '''
        # Combine with existing papers if this is a backup search
        final_papers = papers
        if is_backup_search and existing_papers:
            # Merge papers, avoiding duplicates
            existing_ids = {p['id'] for p in existing_papers}
            new_papers = [p for p in papers if p['id'] not in existing_ids]
            final_papers = existing_papers + new_papers
            
            # Sort by relevance score and keep only top 5
            final_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            final_papers = final_papers[:7]
        else:
            # For non-backup searches, also limit to top 5
            final_papers = papers[:5]
        return {
            **state,
            "experiment_papers": final_papers,
            "experiment_search_iteration": search_iteration + 1,
            "current_step": "papers_downloaded"
        }
        
    except Exception as e:
        print(f"❌ Error in experiment papers search: {str(e)}")
        return {
            **state,
            "experiment_papers": [],
            "errors": state.get("errors", []) + [f"Experiment papers search error: {str(e)}"],
            "current_step": "search_error"
        }

def _validate_experiment_papers_node(state: ExperimentSuggestionState) -> ExperimentSuggestionState:
    """Node to validate if retrieved experiment papers can answer the user's query and decide next steps."""
    model= state["model"]
    client = state["client"]
    _write_stream("Validating experiment paper relevance and determining next steps.")
    state["current_step"] = "validate_experiment_papers"
    
    # Early bypass: If max iterations reached, skip validation and proceed directly
    search_iteration = state.get("experiment_search_iteration", 0)
    if search_iteration >= 3:
        _write_stream(f"⚠️ Maximum iterations ({search_iteration}) reached. Skipping validation and proceeding to experiment generation...")
        
        # CRITICAL FIX: Transfer papers to the correct state keys for new clean architecture
        papers = state.get("experiment_papers", [])
        #print(f"📚 Transferring {len(papers)} papers to validated_experiment_papers for clean architecture")
        
        state["experiment_paper_validation_decision"] = "PROCEED_DIRECT"
        state["validated_experiment_papers"] = papers  # NEW: Transfer papers to the key the clean architecture expects
        state["experiment_paper_validation_results"] = {
            "validation_result": "SKIP",
            "decision": "PROCEED_DIRECT", 
            "papers_count": len(papers),
            "reason": "max_iterations_reached"
        }
        state["next_node"] = "distill_paper_methodologies"  # Route to distillation step first
        return state
    
    try:
        papers = state.get("experiment_papers", [])
        user_query = state["original_prompt"]
        research_direction = state.get("research_direction", {})
        
        # Prepare paper summaries for validation with methodology checks
        papers_summary = ""
        full_content_papers = [p for p in papers if p.get('pdf_downloaded', False)]
        
        # Include information about all papers with enhanced methodology analysis
        methodology_count = 0
        experiment_count = 0
        
        for i, paper in enumerate(papers[:10], 1):  # Top 10 papers
            clean_title = _clean_text_for_utf8(paper.get('title', 'Unknown Title'))
            clean_abstract = _clean_text_for_utf8(paper.get('summary', 'No abstract available'))
            full_content = _clean_text_for_utf8(paper.get('content', ''))
            relevance_score = paper.get('relevance_score', 0)
            has_content = paper.get('pdf_downloaded', False)
            content_status = "FULL CONTENT" if has_content else "TITLE+ABSTRACT"
            
            # Check for methodology and experiments sections
            has_methodology = False
            has_experiments = False
            
            if has_content and full_content:
                content_lower = full_content.lower()
                # Look for methodology indicators
                methodology_keywords = ['methodology', 'method', 'approach', 'algorithm', 'procedure', 'framework', 'implementation']
                has_methodology = any(keyword in content_lower for keyword in methodology_keywords)
                
                # Look for experiment indicators
                experiment_keywords = ['experiment', 'evaluation', 'result', 'performance', 'benchmark', 'dataset', 'accuracy', 'precision', 'recall']
                has_experiments = any(keyword in content_lower for keyword in experiment_keywords)
                
                if has_methodology:
                    methodology_count += 1
                if has_experiments:
                    experiment_count += 1
            
            methodology_status = "✅ METHODOLOGY" if has_methodology else "❌ NO METHODOLOGY"
            experiment_status = "✅ EXPERIMENTS" if has_experiments else "❌ NO EXPERIMENTS"
            
            papers_summary += f"""
                Paper {i} [{content_status}] - Relevance: {relevance_score:.1f}/10.0:
                Title: {clean_title}
                Abstract: {clean_abstract}
                content_snippet: {_clean_text_for_utf8(full_content[:4000])}...
                Content Analysis: {methodology_status} | {experiment_status}
                ---
            """
        
        # Extract research direction context
        selected_direction = research_direction.get("selected_direction", {})
        direction_text = selected_direction.get("direction", "General experimental guidance")
        key_questions = selected_direction.get("key_questions", [])
        
        # Create enhanced validation prompt with HYPER-STRICT requirements and clear JSON format
        validation_prompt = f"""
            You are a HYPER-STRICT research analyst. Only papers with DETAILED METHODOLOGY and CONCRETE EXPERIMENTS should pass validation.

            USER'S QUERY: {_clean_text_for_utf8(user_query)}
            RESEARCH DIRECTION: {_clean_text_for_utf8(direction_text)}
            KEY QUESTIONS: {_clean_text_for_utf8(', '.join(key_questions[:3]) if key_questions else 'General experimental guidance')}
            CURRENT SEARCH ITERATION: {search_iteration + 1}

            RETRIEVED PAPERS:
            {_clean_text_for_utf8(papers_summary)}

            SEARCH STATISTICS:
            - Total papers found: {len(papers)}
            - Papers with full content: {len(full_content_papers)}
            - Papers with methodology sections: {methodology_count}
            - Papers with experiment sections: {experiment_count}
            - Average relevance score: {sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0:.2f}/10.0

            HYPER-STRICT REQUIREMENTS FOR EXPERIMENTAL GUIDANCE PAPERS:
            1. **METHODOLOGY REQUIREMENT**: Papers MUST contain detailed experimental methodologies, algorithms, or implementation details
            2. **EXPERIMENTS REQUIREMENT**: Papers MUST contain actual experimental results, evaluations, or empirical validation
            3. **RELEVANCE REQUIREMENT**: Papers MUST be directly relevant to the research direction (≥8.0/10.0 for "continue")
            4. **COMPLETENESS REQUIREMENT**: Papers MUST provide actionable experimental guidance, not just theoretical discussions
            5. **TECHNICAL DEPTH REQUIREMENT**: Papers MUST include specific experimental procedures, datasets, metrics, or protocols

            STRICT DECISION CRITERIA:
            - "continue": ≥5 papers with BOTH methodology AND experiments, avg relevance ≥8.0, comprehensive experimental coverage
            - "search_backup": 3-4 papers with methodology/experiments, avg relevance ≥7.0, partial experimental coverage
            - "search_new": <3 papers with methodology/experiments, avg relevance <7.0, insufficient experimental guidance

            WARNING: Be EXTREMELY STRICT. Only "continue" if papers provide CONCRETE, ACTIONABLE experimental methodologies.

            REQUIRED JSON FORMAT (return ONLY this JSON, no other text):
            {{
                "relevance_assessment": "excellent" | "good" | "fair" | "poor",
                "methodology_coverage": "comprehensive" | "partial" | "insufficient",
                "experiment_coverage": "comprehensive" | "partial" | "insufficient", 
                "actionable_guidance": "high" | "medium" | "low",
                "technical_depth": "detailed" | "moderate" | "superficial",
                "decision": "continue" | "search_backup" | "search_new",
                "confidence": 0.95,
                "reasoning": "Brief explanation focusing on methodology and experimental content quality",
                "missing_aspects": ["aspect1", "aspect2", "aspect3"],
                "methodology_gaps": ["gap1", "gap2", "gap3"],
                "search_guidance": {{
                    "new_search_terms": ["term1", "term2", "term3"],
                    "focus_areas": ["area1", "area2", "area3"],
                    "avoid_terms": ["avoid1", "avoid2"]
                }}
            }}

            BE RUTHLESS: If papers lack concrete experimental methodologies or detailed experimental procedures, choose "search_new".
            Return only valid JSON with all required fields.
        """

        # Call LLM for validation
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"content": _clean_text_for_utf8(validation_prompt), "role": "user"}
            ]
        )
        
        validation_response = response.choices[0].message.content.strip()
        
        # Parse validation response
        try:
            # Remove any markdown formatting
            if validation_response.startswith("```json"):
                validation_response = validation_response[7:]
            if validation_response.endswith("```"):
                validation_response = validation_response[:-3]
            validation_response = validation_response.strip()
            
            validation_data = json.loads(validation_response)
            
            # Store validation results in state - use unique key to avoid conflicts
            state["experiment_paper_validation_results"] = {
                "validation_successful": True,
                "validation_data": validation_data,
                "decision": validation_data.get("decision", "continue"),
                "reasoning": validation_data.get("reasoning", "No reasoning provided"),
                "missing_aspects": validation_data.get("missing_aspects", []),
                "search_guidance": validation_data.get("search_guidance", {}),
                "iteration": search_iteration + 1
            }
            
            # ALSO store decision in a separate key to avoid conflicts with other workflows
            state["experiment_paper_validation_decision"] = validation_data.get("decision", "continue")
            
        
            # Safe extraction with defaults and formatting
            relevance = str(validation_data.get('relevance_assessment', 'Unknown')).title()
            methodology_coverage = str(validation_data.get('methodology_coverage', 'Unknown')).title()
            experiment_coverage = str(validation_data.get('experiment_coverage', 'Unknown')).title()
            actionable_guidance = str(validation_data.get('actionable_guidance', 'Unknown')).title()
            technical_depth = str(validation_data.get('technical_depth', 'Unknown')).title()
            decision = str(validation_data.get('decision', 'continue')).upper()
            confidence = float(validation_data.get('confidence', 0))
            reasoning = str(validation_data.get('reasoning', 'No reasoning provided'))
            '''
            print(f"🎯 Relevance Assessment: {relevance}")
            print(f"🔬 Methodology Coverage: {methodology_coverage}")
            print(f"🧪 Experiment Coverage: {experiment_coverage}")
            print(f"📋 Actionable Guidance: {actionable_guidance}")
            print(f"⚙️ Technical Depth: {technical_depth}")
            print(f"🚀 Decision: {decision}")
            print(f"🎲 Confidence: {confidence:.2f}")
            print(f"💭 Reasoning: {reasoning}")
            '''
            # Handle missing aspects with proper formatting
            missing_aspects = validation_data.get('missing_aspects', [])
            '''
            if missing_aspects and isinstance(missing_aspects, list):
                print(f"🔍 Missing Experimental Aspects:")
                for i, aspect in enumerate(missing_aspects[:5], 1):  # Limit to 5 items
                    print(f"   {i}. {str(aspect)}")
            
            # Handle methodology gaps
            methodology_gaps = validation_data.get('methodology_gaps', [])
            if methodology_gaps and isinstance(methodology_gaps, list):
                print(f"🔧 Methodology Gaps:")
                for i, gap in enumerate(methodology_gaps[:3], 1):  # Limit to 3 items
                    print(f"   {i}. {str(gap)}")
            '''
            # Handle search guidance for non-continue decisions
            if decision != 'CONTINUE':
                search_guidance = validation_data.get('search_guidance', {})
                if isinstance(search_guidance, dict):
                    new_search_terms = search_guidance.get('new_search_terms', [])
                    focus_areas = search_guidance.get('focus_areas', [])
                    avoid_terms = search_guidance.get('avoid_terms', [])
                    
                    if new_search_terms and isinstance(new_search_terms, list):
                        _write_stream(f"Suggested Search Terms: {', '.join(str(term) for term in new_search_terms[:7])}")
                    if focus_areas and isinstance(focus_areas, list):
                        _write_stream(f"Focus Areas: {', '.join(str(area) for area in focus_areas[:5])}")
                    if avoid_terms and isinstance(avoid_terms, list):
                        _write_stream(f"Avoid Terms: {', '.join(str(term) for term in avoid_terms[:5])}")
            
   
            # CRITICAL FIX: Make the routing decision here instead of in separate function
            validation_decision = validation_data.get("decision", "continue").upper()
            
            # Map validation decision to next node
            if validation_decision == "CONTINUE":
                next_node = "distill_paper_methodologies"  # Route to distillation step first
                _write_stream(f"Experiment papers are adequate. Continuing to experiment suggestions.")
                
                state["validated_experiment_papers"] = papers
                
                
            elif validation_decision == "SEARCH_BACKUP":
                next_node = "search_experiment_papers"
                _write_stream(f"Papers need backup. Searching for additional papers.")
            elif validation_decision == "SEARCH_NEW":
                next_node = "generate_experiment_search_query"
                _write_stream(f"Papers inadequate. Generating new search query.")
            else:
                # Default fallback
                next_node = "distill_paper_methodologies"  # Route to distillation step first
                _write_stream(f"Unknown validation decision '{validation_decision}'. Defaulting to continue.")
                
            
                state["validated_experiment_papers"] = papers
            
            # Increment search iteration counter
            state["experiment_search_iteration"] = search_iteration + 1
            state["next_node"] = next_node
            
            # Return state after successful validation
            return state
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse validation JSON: {e}"
            print(f"ERROR: {error_msg}")
            
            # Fallback decision based on paper quality
            avg_score = sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0
            decision = "continue" if avg_score >= 6.0 else "search_backup"
            
            state["experiment_paper_validation_results"] = {
                "validation_successful": False,
                "error": error_msg,
                "decision": decision,
                "reasoning": f"Fallback decision based on average score: {avg_score:.2f}",
                "iteration": search_iteration + 1
            }
            
            # ALSO store decision in backup key for error cases
            state["experiment_paper_validation_decision"] = decision
            
            # Add routing for error case
            if decision == "continue":
                state["next_node"] = "distill_paper_methodologies"  # Route to distillation step first
                # CRITICAL FIX: Transfer papers for continue decision in JSON error case
               
                state["validated_experiment_papers"] = papers
            else:
                state["next_node"] = "search_experiment_papers"
            
            state["experiment_search_iteration"] = search_iteration + 1
            
            
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Default to continue on error
        state["experiment_paper_validation_results"] = {
            "validation_successful": False,
            "error": error_msg,
            "decision": "continue",
            "reasoning": "Error occurred, defaulting to continue",
            "iteration": state.get("experiment_search_iteration", 0) + 1
        }
        
        # ALSO store decision in backup key for error cases
        state["experiment_paper_validation_decision"] = "continue"
        
        # Add routing for error case  
        state["next_node"] = "distill_paper_methodologies"  # Route to distillation step first
        
        # CRITICAL FIX: Transfer papers for general error case too
        papers = state.get("experiment_papers", [])
       # print(f"📚 Transferring {len(papers)} papers to validated_experiment_papers (general error)")
        state["validated_experiment_papers"] = papers
        
        state["experiment_search_iteration"] = state.get("experiment_search_iteration", 0) + 1
    
    return state



#---------------------- some utility functions ----------------------
def create_experiment_ranking_context_from_analysis(state: ExperimentSuggestionState) -> str:
    """Create enhanced ranking context for experiment suggestions using extracted analysis."""
    # Start with the original user query
    context_parts = [f"User Query: {state['original_prompt']}"]
    
    # Add findings analysis if available
    findings_analysis = state.get("findings_analysis", {})
    if findings_analysis:
        # Add domain information
        domain_analysis = findings_analysis.get("domain_analysis", {})
        if domain_analysis:
            context_parts.append("Research Domain Context:")
            if domain_analysis.get("primary_domain"):
                context_parts.append(f"- Primary Domain: {domain_analysis['primary_domain']}")
            if domain_analysis.get("task_type"):
                context_parts.append(f"- Task Type: {domain_analysis['task_type']}")
            if domain_analysis.get("application_area"):
                context_parts.append(f"- Application: {domain_analysis['application_area']}")
            if domain_analysis.get("data_type"):
                context_parts.append(f"- Data Type: {domain_analysis['data_type']}")
        
        # Add research opportunities 
        opportunities = findings_analysis.get("research_opportunities", [])
        if opportunities:
            context_parts.append("Research Focus Areas:")
            for opp in opportunities[:3]:  # Top 3 opportunities
                context_parts.append(f"- {opp}")
        
        # Add current state information
        current_state = findings_analysis.get("current_state", {})
        if current_state and current_state.get("findings"):
            context_parts.append(f"Current Research State: {current_state['findings']}")
    
    # Add research direction if available
    research_direction = state.get("research_direction", {})
    if research_direction:
        selected_direction = research_direction.get("selected_direction", {})
        if selected_direction.get("direction"):
            context_parts.append(f"Research Direction: {selected_direction['direction']}")
        
        # Add key questions
        key_questions = selected_direction.get("key_questions", [])
        if key_questions:
            context_parts.append("Key Research Questions:")
            for question in key_questions[:2]:  # Top 2 questions
                context_parts.append(f"- {question}")
    
    # Combine all parts
    ranking_context = '\n'.join(context_parts)
    
    # Limit total length to avoid token issues
    if len(ranking_context) > 1500:
        ranking_context = ranking_context[:1500] + "..."
    
    return ranking_context


