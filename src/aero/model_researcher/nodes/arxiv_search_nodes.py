from ..shared_defs import ModelSuggestionState, _write_stream,_clean_text_for_utf8
from langchain_core.messages import AIMessage 
import urllib.request as libreq
import xml.etree.ElementTree as ETS
from concurrent.futures import ThreadPoolExecutor, as_completed
import json



async def _search_arxiv_node(state: ModelSuggestionState) -> ModelSuggestionState:
    """Node for searching arXiv papers using optimized workflow with backup search support."""
    
    # Extract dependencies from state
    arxiv_processor = state["arxiv_processor"]
    
    search_iteration = state.get("search_iteration", 0)
    validation_results = state.get("validation_results", {})
    is_backup_search = validation_results.get("decision") == "search_backup"
    
    state["current_step"] = "search_arxiv"
    
    # Initialize variables
    papers = []
    total_results = 0
    formatted_query = ""
    
    # For backup searches, preserve existing papers
    existing_papers = []
    if is_backup_search and state.get("arxiv_results", {}).get("papers"):
        existing_papers = state["arxiv_results"]["papers"]
       # print(f"📚 Preserving {len(existing_papers)} papers from previous search")
    
    try:
        search_query = state["arxiv_search_query"]
        original_prompt = state["original_prompt"]
        
        # Determine search parameters based on search type and iteration
        if search_iteration == 0:
            # Initial search: get 100 papers
            max_results = 200
            start_offset = 0
        elif is_backup_search:
            # Backup search: get additional papers with offset to avoid duplicates
            # Use offset based on how many papers we already have
            existing_count = len(existing_papers) if existing_papers else 0
            start_offset = max(100, existing_count)  # Start after existing papers
            max_results = 50  # Get additional papers
        else:
            # New search with different query: get 100 fresh papers
            max_results = 100  
            start_offset = 0
        
       # print("=" * 80)
        
        # Format the search query
        formatted_query = format_search_string(search_query)
        _write_stream(f"Formatted query: {formatted_query}")
        
        # Build the URL with proper offset
        url = f"http://export.arxiv.org/api/query?search_query={formatted_query}&start={start_offset}&max_results={max_results}"
        _write_stream(f"Full URL: {url}")
        
        with libreq.urlopen(url) as response:
            xml_data = response.read()
        
        # Parse XML
        root = ETS.fromstring(xml_data)
        
        # Namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
        }
        
        # Get total results
        total_results_elem = root.find('opensearch:totalResults', ns)
        total_results = int(total_results_elem.text) if total_results_elem is not None else 0
        
        #print(f"Total papers found: {total_results}")
        
        if total_results > 0:
         #   print("=" * 80)
            
            # Get all paper entries
            entries = root.findall('atom:entry', ns)
            
            # Alternative - try without namespace as fallback
            entries_no_ns = root.findall('.//entry')
            
            # If no entries found with namespace, try alternative approach
            if len(entries) == 0 and len(entries_no_ns) > 0:
                entries = entries_no_ns
            
            # If we got very few results compared to total, try a simpler query
            if len(entries) < 5 and total_results > 1000:
                
                # Try a simpler query by removing the most specific terms
                query_parts = search_query.split('/')
                if len(query_parts) > 2:
                    # Keep only the first two most important terms
                    fallback_query = '/'.join(query_parts[:2])
                    formatted_fallback = format_search_string(fallback_query)
                    fallback_url = f"http://export.arxiv.org/api/query?search_query={formatted_fallback}&start=0&max_results={max_results}"
                    _write_stream(f"Fallback query: {fallback_query}")
                    _write_stream(f"Fallback URL: {fallback_url}")
                    
                    try:
                        with libreq.urlopen(fallback_url) as fallback_response:
                            fallback_xml_data = fallback_response.read()
                        
                        fallback_root = ETS.fromstring(fallback_xml_data)
                        fallback_entries = fallback_root.findall('atom:entry', ns)
                        
                        if len(fallback_entries) > len(entries):
                            _write_stream(f"Fallback found {len(fallback_entries)} entries - using fallback results")
                            entries = fallback_entries
                            xml_data = fallback_xml_data  # Update for consistency
                            root = fallback_root
                        else:
                            _write_stream(f"Fallback only found {len(fallback_entries)} entries - keeping original")
                            
                            
                    except Exception as fallback_error:
                        _write_stream(f"Fallback query failed: {fallback_error}")

            
            
            # Stage 1: Extract basic info (title, abstract, metadata) without downloading PDFs
            _write_stream(f"Extracting basic info for {len(entries)} papers...")
            papers = []
            for i, entry in enumerate(entries, 1):
                paper_info = arxiv_processor.extract_basic_paper_info(entry, ns, i)
                papers.append(paper_info)
                #print(f"  - [{i}] {paper_info['title']} ({paper_info['id']})")
            
            # Stage 2: Rank papers by relevance using enhanced analysis context
            _write_stream(f"Ranking papers by relevance.")

            # Create enhanced ranking context from the detailed analysis
            ranking_context = _create_ranking_context_from_analysis(state)
            #print(f"📊 Using enhanced context for ranking: {ranking_context[:100]}...")
            
            # Create custom prompt for model suggestion ranking
            custom_prompt = _create_custom_ranking_prompt("model_suggestion")
            
            papers = await arxiv_processor.rank_papers_by_relevance(papers, ranking_context, custom_prompt)
            
            # Stage 3: Download full content for top 5 papers only
            top_papers = papers  # Get top 5 papers

            _write_stream(f"Stage 3: Downloading full PDF content for top {len(top_papers)} papers.")

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

            _write_stream(f"PDF download stage completed. Top 5 papers now have full content.")

            # Print final results (now ranked by relevance)
            #print("\n" + "=" * 80)
            #print("📋 RANKED RESULTS (by relevance):")
            #print("=" * 80)
         
            
            
            
            ttl = 0
            scores = []
            '''
            for i, paper in enumerate(papers, 1):
                relevance_score = paper.get('relevance_score', 0)
                ttl += float(relevance_score)
                scores.append(float(relevance_score))
                has_content = paper.get('pdf_downloaded', False)
                content_status = "📄 FULL CONTENT" if has_content else "📝 TITLE+ABSTRACT"
                
                print(f"\n📄 PAPER #{i} ({content_status}) - Relevance: {relevance_score:.1f}/10.0")
                print("-" * 60)
                print(f"Title: {paper['title']}")
                print(f"ID: {paper['id']}")
                print(f"Published: {paper['published']}")
                print(f"URL: {paper['url']}")
                
                
                # Show summary for all papers
                if paper.get('summary'):
                    print(f"Summary: {paper['summary'][:300]}...")
                
                # Show content preview only if downloaded
                if paper.get('content'):
                    print(f"Full Content Preview:\n{paper['content'][:500]}...")
                elif not has_content and i <= 5:
                    print("Full Content: [Available in top 5 - check PDF download status]")
                else:
                    print("Full Content: [Not downloaded - not in top 5]")
                print("-" * 60)
            
            # Calculate statistics
            if scores:
                avg = ttl / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                print(f"\n📊 RELEVANCE SCORE STATISTICS:")
                print(f"Average score: {avg:.2f}/10.0")
                print(f"Maximum score: {max_score:.2f}/10.0")
                print(f"Minimum score: {min_score:.2f}/10.0")
                print(f"Score range: {min_score:.2f} - {max_score:.2f}")
            else:
                print(f"Average score: 0.00/10.0")
            '''
            
            # Set the arxiv_results in state for successful search
            state["arxiv_results"] = {
                "search_successful": True,
                "total_results": str(total_results),
                "papers_returned": len(papers),
                "papers": papers,
                "formatted_query": formatted_query,
                "original_query": search_query,
                "search_type": "backup" if is_backup_search else "new",
                "iteration": search_iteration + 1
            }
            
        
        
        # Add success message
        state["messages"].append(
            AIMessage(content=f"ArXiv search completed. Found {total_results} total papers, processed {len(papers) if total_results > 0 else 0} papers.")
        )
                
    except Exception as e:
        error_msg = f"Error searching arXiv: {type(e).__name__}: {str(e)}"
        _write_stream(f"Full error details: {error_msg}")
        import traceback
        traceback.print_exc()
        
        state["errors"].append(error_msg)
        state["arxiv_results"] = {
            "search_successful": False,
            "error": error_msg,
            "total_results": "0",
            "papers_returned": 0,
            "papers": [],
            "formatted_query": formatted_query,
            "original_query": state["arxiv_search_query"]
        }
    
    return state



def _validate_papers_node(state: ModelSuggestionState) -> ModelSuggestionState:
    """Node to validate if retrieved papers can answer the user's query and decide next steps."""
    
    # Extract dependencies from state
    client = state["client"]
    model = state["model"]

    _write_stream("Validating paper relevance and determining next steps.")
    state["current_step"] = "validate_papers"
    
    try:
        papers = state["arxiv_results"].get("papers", [])
        _write_stream(f"Validating {len(papers)} retrieved papers...")
        user_query = state["original_prompt"]
        search_iteration = state.get("search_iteration", 0)
        
        # Prepare paper summaries for validation
        papers_summary = ""
        full_content_papers = [p for p in papers if p.get('pdf_downloaded', False)]
        
        # Include information about all papers (not just those with full content)
        for i, paper in enumerate(papers[:10], 1):  # Top 10 papers
            clean_title = _clean_text_for_utf8(paper.get('title', 'Unknown Title'))
            clean_abstract = _clean_text_for_utf8(paper.get('summary', 'No abstract available'))
            relevance_score = paper.get('relevance_score', 0)
            has_content = paper.get('pdf_downloaded', False)
            content_status = "FULL CONTENT" if has_content else "TITLE+ABSTRACT"
            
            papers_summary += f"""
                Paper {i} [{content_status}] - Relevance: {relevance_score:.1f}/10.0:
                Title: {clean_title}
                Abstract: {clean_abstract}
                ---
            """
          
        
        # Create enhanced validation prompt with decision guidance
        validation_prompt = f"""
You are an expert research analyst. Evaluate the retrieved papers and determine the best course of action.

USER'S QUERY: {_clean_text_for_utf8(user_query)}
CURRENT SEARCH ITERATION: {search_iteration + 1}

RETRIEVED PAPERS:
{papers_summary}

SEARCH STATISTICS:
- Total papers found: {len(papers)}
- Papers with full content: {len(full_content_papers)}
- Average relevance score: {sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0:.2f}/10.0

Please provide your assessment in the following JSON format:

{{
"relevance_assessment": "excellent" | "good" | "fair" | "poor",
"coverage_analysis": "complete" | "partial" | "insufficient",
"quality_evaluation": "high" | "medium" | "low",
"decision": "continue" | "search_backup" | "search_new",
"confidence": 0.0-1.0,
"reasoning": "Brief explanation of the decision",
"missing_aspects": ["list", "of", "missing", "aspects"],
"search_guidance": {{
    "new_search_terms": ["alternative", "search", "terms"],
    "focus_areas": ["areas", "to", "focus", "on"],
    "avoid_terms": ["terms", "to", "avoid"]
}}
}}

DECISION CRITERIA:
- "continue": Papers are sufficient (relevance ≥7.0, good coverage)
- "search_backup": Papers are decent but could use backup (relevance 5.0-6.9, partial coverage)  
- "search_new": Papers are insufficient (relevance <5.0, poor coverage, or major gaps)

If search_iteration ≥ 2, bias toward "continue" unless papers are truly inadequate.

Return only the JSON object, no additional text.
"""

        # Call LLM for validation
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"content": validation_prompt, "role": "user"}]
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
            state["validation_results"] = {
                "validation_successful": True,
                "validation_data": validation_data,
                "decision": validation_data.get("decision", "continue"),
                "reasoning": validation_data.get("reasoning", "No reasoning provided"),
                "missing_aspects": validation_data.get("missing_aspects", []),
                "search_guidance": validation_data.get("search_guidance", {}),
                "iteration": search_iteration + 1
            }
            
            # ALSO store decision in a separate key to avoid conflicts with other workflows
            state["paper_validation_decision"] = validation_data.get("decision", "continue")
            
            # Print validation results
            '''
            print("\n" + "=" * 70)
            print("📋 PAPER VALIDATION & DECISION RESULTS")
            print("=" * 70)
            print(f"🎯 Relevance Assessment: {validation_data.get('relevance_assessment', 'unknown').title()}")
            print(f"📊 Coverage Analysis: {validation_data.get('coverage_analysis', 'unknown').title()}")
            print(f"⭐ Quality Evaluation: {validation_data.get('quality_evaluation', 'unknown').title()}")
            print(f"🚀 Decisiodn: {validation_data.get('decision', 'continue').upper()}")
            print(f"🎲 Confidence: {validation_data.get('confidence', 0):.2f}")
            _write_stream(f"Reasoning: {validation_data.get('reasoning', 'No reasoning provided')}")
            '''
            _write_stream(f"Paper validation decision and reasoning: {validation_data.get('decision', 'continue').upper()}")
            
            if validation_data.get('missing_aspects'):
                _write_stream(f"Missing Aspects: {', '.join(validation_data['missing_aspects'])}")
            
            if validation_data.get('decision') != 'continue':
                search_guidance = validation_data.get('search_guidance', {})
                if search_guidance.get('new_search_terms'):
                    _write_stream(f"Suggested Search Terms: {', '.join(search_guidance['new_search_terms'])}")
                if search_guidance.get('focus_areas'):
                    _write_stream(f"Focus Areas: {', '.join(search_guidance['focus_areas'])}")


            # Increment search iteration counter    
            state["search_iteration"] = search_iteration + 1
            
            # Return state after successful validation
            return state
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse validation JSON: {e}"
            print(f"⚠️ {error_msg}")
            
            # Fallback decision based on paper quality
            avg_score = sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0
            decision = "continue" if avg_score >= 6.0 else "search_backup"
            
            state["validation_results"] = {
                "validation_successful": False,
                "error": error_msg,
                "decision": decision,
                "reasoning": f"Fallback decision based on average score: {avg_score:.2f}",
                "iteration": search_iteration + 1
            }
            
            # ALSO store decision in backup key for error cases
            state["paper_validation_decision"] = decision
            
            state["search_iteration"] = search_iteration + 1
            
            
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        _write_stream(f"ERROR: {error_msg}")
        
        # Default to continue on error
        state["validation_results"] = {
            "validation_successful": False,
            "error": error_msg,
            "decision": "continue",
            "reasoning": "Error occurred, defaulting to continue",
            "iteration": state.get("search_iteration", 0) + 1
        }
        
        # ALSO store decision in backup key for error cases
        state["paper_validation_decision"] = "continue"
        
        state["search_iteration"] = state.get("search_iteration", 0) + 1
    
    return state





def _generate_search_query_node(state: ModelSuggestionState) -> ModelSuggestionState:
    """Node for generating arXiv search query with optional guidance from validation."""
    
    # Extract dependencies from state
    client = state["client"]
    model = state["model"]
    
    search_iteration = state.get("search_iteration", 0)
    validation_results = state.get("validation_results", {})
    
   # if search_iteration == 0:
   #     print("\n📚 Step 2: Generating initial arXiv search query...")
   # else:
   #     print(f"\n🔄 Step 2 (Iteration {search_iteration + 1}): Generating refined search query based on validation guidance...")
        
    state["current_step"] = "generate_search_query"
    
    try:
        # Extract key properties with high confidence
        high_confidence_props = [prop for prop in state["detected_categories"] if prop.get("confidence", 0) > 0.7]
        prop_names = [prop["name"] for prop in high_confidence_props]
        
        # Prepare guidance from validation if available
        guidance_context = ""
        if search_iteration > 0 and validation_results.get("search_guidance"):
            search_guidance = validation_results["search_guidance"]
            missing_aspects = validation_results.get("missing_aspects", [])
            
            guidance_context = f"""
            
            ## SEARCH REFINEMENT GUIDANCE (from validation)
            Previous search was insufficient. Please incorporate this guidance:
            
            Missing Aspects: {', '.join(missing_aspects)}
            Suggested New Terms: {', '.join(search_guidance.get('new_search_terms', []))}
            Focus Areas: {', '.join(search_guidance.get('focus_areas', []))}
            Terms to Avoid: {', '.join(search_guidance.get('avoid_terms', []))}
            
            IMPORTANT: Generate a DIFFERENT query that addresses these missing aspects.
            """
        
        content = f"""
            Based on the following machine learning research task analysis, generate ONE concise arXiv API search query (exactly 4 terms, separated by forward slashes).
            The query should be optimized to find the most relevant papers that are able to suggest models that can be used to address the task.

            Original Task: {state["original_prompt"]}

            Detected Categories: {', '.join(prop_names)}

            Detailed Analysis: {state["detailed_analysis"].get('llm_analysis', 'Not available')}
            {guidance_context}

            Rules for constructing the query:
            - EXACTLY 4 terms, separated by "/" (no quotes, no extra spaces).
            - Include:
            1) a MODEL keyword (e.g., transformer, ViT, DETR, RT-DETR, Deformable DETR, YOLOS),
            2) the TASK (e.g., object detection, segmentation),
            3) a DEPLOYMENT/CONSTRAINT or TOOLING term if present (e.g., real-time, edge deployment, TensorRT, quantization, INT8).
            4) a DOMAIN or APPLICATION term if relevant (e.g., medical imaging, remote sensing, autonomous vehicles).
            - Prefer task-specific + model-specific terms over generic ones.
            - Avoid vague terms like "deep learning" or "machine learning" unless nothing better fits.
            - Prefer dataset/benchmark anchors (e.g., KITTI, nuScenes, Waymo) OVER broad domain words (e.g., autonomous vehicles). Use the domain ONLY if it is essential and not overly broad.
            - If computer vision is relevant, make the TASK a CV term (e.g., object detection, instance segmentation).
            - Do NOT include arXiv category labels (cs.CV, cs.LG) in the query terms.
            - Return ONLY the query string (no explanation, no punctuation besides "/").

            Good examples:
            - transformer/object detection/real-time/autonomous vehicles
            - RT-DETR/object detection/TensorRT/KITTI
            - Deformable DETR/object detection/KITTI/autonomous driving
            - vision transformer/object detection/edge deployment/medical imaging
        """

        response = client.chat.completions.create(
            model=model,
            temperature=0 if search_iteration == 0 else 0.3,  # Add some randomness for refinements
            messages=[{"content": content, "role": "user"}]
        )
        
        search_query = response.choices[0].message.content.strip()
        
        # Store search query with iteration tracking
        if "search_queries" not in state:
            state["search_queries"] = []
        state["search_queries"].append(search_query)
        state["arxiv_search_query"] = search_query
        
     #   if search_iteration == 0:
     #       print(f"Generated initial search query: '{search_query}'")
     #   else:
     #       print(f"Generated refined search query: '{search_query}'")
     #       print(f"Previous queries: {', '.join(state['search_queries'][:-1])}")
        
        # Add success message
        state["messages"].append(
            AIMessage(content=f"Generated arXiv search query (iteration {search_iteration + 1}): '{search_query}'")
        )
        
    
    except Exception as e:
        # Fallback to simple keyword extraction with slashes
        keywords = []
        prompt = state["original_prompt"].lower()
        if "neural" in prompt or "deep" in prompt:
            keywords.append("neural network")
        if "time series" in prompt or "temporal" in prompt:
            keywords.append("time series")
        if "classification" in prompt:
            keywords.append("classification")
        if "clustering" in prompt:
            keywords.append("clustering")
        if "anomaly detection" in prompt:
            keywords.append("anomaly detection")
        if "autoencoder" in prompt:
            keywords.append("autoencoder")
        
        search_query = "/".join(keywords) if keywords else "drone detection"
        state["arxiv_search_query"] = search_query
        
        error_msg = f"Search query generation failed, using fallback: {str(e)}"
        state["errors"].append(error_msg)
        _write_stream(f"ERROR: {error_msg}")
    
    return state












#---------------------------some helper functions---------------------------


# Utility function for search string formatting
def format_search_string(input_string):
    """Convert string to arXiv search format handling slash-separated terms.
    
    Input: "deep learning/time series/forecasting/variable length"
    Output: 'all:%22deep+learning%22+AND+all:%22time+series%22+AND+all:forecasting+AND+all:%22variable+length%22'
    """
    # Split by forward slashes
    terms = input_string.strip().split('/')
    parts = []
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
        
        # If term has spaces, treat it as a phrase (add quotes and encoding)
        if ' ' in term:
            # Replace spaces with + and add URL encoding for quotes
            formatted_term = term.replace(' ', '+')
            parts.append(f'all:%22{formatted_term}%22')
        else:
            # Single word, no quotes needed
            parts.append(f'all:{term}')
    
    # Join with AND
    return '+'.join(parts) if parts else ""





def _create_ranking_context_from_analysis(state: ModelSuggestionState) -> str:
    """Create enhanced ranking context using extracted analysis information."""
    # Start with the original user query
    context_parts = [f"User Query: {state['original_prompt']}"]
    
    # Add detected categories if available
    categories = state.get("detected_categories", [])
    if categories:
        relevant_categories = [cat for cat in categories if cat.get("confidence", 0) > 0.5]
        if relevant_categories:
            context_parts.append("Relevant Research Categories:")
            for cat in relevant_categories[:3]:  # Top 3 categories
                context_parts.append(f"- {cat['name']} (confidence: {cat['confidence']:.2f})")
    
    # Add structured analysis if available
    detailed_analysis = state.get("detailed_analysis", {})
    if detailed_analysis and "llm_analysis" in detailed_analysis:
        llm_analysis = detailed_analysis["llm_analysis"]
        if llm_analysis and isinstance(llm_analysis, str):
            # Extract key components from the LLM analysis
            analysis_lines = llm_analysis.split('\n')
            relevant_lines = []
            
            # Look for specific sections that would help with paper ranking
            for line in analysis_lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in [
                    'domain:', 'task type:', 'approach:', 'methodology:', 'technique:',
                    'model type:', 'application:', 'requirements:', 'constraints:'
                ]):
                    relevant_lines.append(line)
            
            if relevant_lines:
                context_parts.append("Key Analysis Points:")
                context_parts.extend(relevant_lines[:5])  # Top 5 relevant lines
    
    # Combine all parts
    ranking_context = '\n'.join(context_parts)
    
    # Limit total length to avoid token issues
    if len(ranking_context) > 1500:
        ranking_context = ranking_context[:1500] + "..."
    
    return ranking_context





def _create_custom_ranking_prompt(prompt_type: str = "default") -> str:
    """Create a custom ranking prompt based on prompt type."""
    
    if prompt_type == "experimental":
        return """
            You are an expert experimental methodology researcher.  
            Your task: Estimate how relevant this paper is to **experimental research needs** using ONLY the paper’s title and summary (abstract).  

            OUTPUT FORMAT (STRICT):
            - Return EXACTLY one floating-point number with ONE decimal (e.g., 8.7).  
            - No words, no JSON, no units, no symbols, no explanation.  
            - Single line only (no leading/trailing spaces or extra lines).  

            SCORING CRITERIA (use inference from title/summary):  
            - methodology_relevance (40%): Does the summary explicitly mention experimental methodology, benchmarks, protocols, or evaluation setups?  
            - experimental_evidence (30%): Does it mention results, experiments, performance comparisons, or ablation studies?  
            - implementation_guidance (20%): Does it provide or strongly imply practical details like datasets, code availability, reproducibility, or implementation notes?  
            - research_alignment (10%): Does it align with the given research direction and questions?  

            COMPUTE:  
            - Let m,e,i,r ∈ [0,1], estimated from the title/summary.  
            - score = round((0.40*m + 0.30*e + 0.20*i + 0.10*r) * 10, 1).  
            - If the title/summary clearly lacks experimental content (all four < 0.15), output **1.0**.  
            - Clip final result to [1.0, 10.0].  

            PRIORITIZATION:  
            - Favor papers with explicit mention of **empirical studies, benchmarks, datasets, or evaluation frameworks**.  
            - Penalize papers that are purely theoretical, conceptual, or survey-style with no experimental grounding.  
            Research context:
            \"\"\"{query}\"\"\"

            Paper title:
            \"\"\"{title}\"\"\"

            Paper summary:
            \"\"\"{content}\"\"\"
        """.strip()
    
    elif prompt_type == "model_suggestion":
        return """
            You are an expert ML model selection researcher. Score how relevant this paper is to model selection and architecture research on a 1–10 scale.

            OUTPUT FORMAT (STRICT):
            - Return EXACTLY one floating-point number with ONE decimal (e.g., 8.7).
            - No words, no JSON, no units, no symbols, no explanation.
            - Single line only (no leading/trailing spaces or extra lines).

            MODEL FOCUS SCORING - assign four subscores in [0,1]:
            - architecture_relevance (60%): discusses relevant model architectures, neural network designs, or ML approaches
            - performance_evidence (30%): provides performance benchmarks, comparisons, or evaluation results
            - implementation_details (10%): includes implementation specifics, hyperparameters, training procedures, or code
            - task_alignment (10%): addresses similar tasks, domains, or application requirements

            Compute:
            - Let a,p,i,t ∈ [0,1].
            - score = round((0.40*a + 0.30*p + 0.20*i + 0.10*t) * 10, 1).
            - If clearly unrelated to models/architectures (all four < 0.15), output 1.0.
            - Clip to [1.0, 10.0].
            - Prioritize papers with concrete model architectures and performance data.

            Research context:
            \"\"\"{query}\"\"\"

            Paper title:
            \"\"\"{title}\"\"\"

            Paper summary:
            \"\"\"{content}\"\"\"
        """.strip()
                    
    else:  # default prompt
        return None  # Use the original prompt in arxiv_paper_utils.py
