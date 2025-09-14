"""
Experiment-Focused Literature Search Workflow (search.py)
=========================================================

This workflow orchestrates literature search focusing on experiment extraction:
1. Keyword extraction from hypothesis
2. FAISS search for relevant experiment chunks
3. LLM-based validation of chunks
4. ArXiv search based on paper abstracts if insufficient chunks
5. Paper download and experiment design extraction via LLM
6. Storage in FAISS with source metadata

"""

import os
import faiss
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import urllib.request as libreq
from dotenv import load_dotenv
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from arxiv_paper_utils import ArxivPaperProcessor
from init_utils import get_llm_response
from langgraph.graph import StateGraph, END

# --- Initialize clients and ArXiv processor ---
load_dotenv()

primary_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://agents.aetherraid.dev")
)

PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")
arxiv_processor = None


# --- ArXiv Processor Initialization ---
def initialize_arxiv_processor():
    """Initialize ArXiv processor with LLM client"""
    global arxiv_processor
    if arxiv_processor is None:
        arxiv_processor = ArxivPaperProcessor(primary_client, PRIMARY_MODEL)
    return arxiv_processor

# --- Extract Keywords ---
async def extract_keywords_from_hypothesis(hypothesis):
    """Extract search keywords from hypothesis for experiment search"""
    prompt = f"""
    Extract 3-4 key research keywords from this hypothesis for searching experimental literature.
    Focus on methodology, variables, and research domains. Use simple, standard academic terms.
    
    Hypothesis: {hypothesis}
    
    Return only keywords separated by commas:
    """
    
    response = await get_llm_response([
        {"role": "system", "content": "Extract experimental research keywords for literature search."},
        {"role": "user", "content": prompt}
    ], temperature=0.1)
    
    keywords = [kw.strip() for kw in response.replace('*', '').split(',') if kw.strip()]
    return keywords[:4]

# --- Vectorization Helper Functions ---
async def vectorize(text):
    """Vectorize text using sentence-transformers"""
    global arxiv_processor
    if not arxiv_processor:
        arxiv_processor = initialize_arxiv_processor()
    
    model = arxiv_processor._get_embedding_model()
    if model is None:
        return None

    embedding = model.encode([text], convert_to_tensor=False, show_progress_bar=False)
    normalized_embedding = normalize_vector(embedding[0])
    return normalized_embedding

def normalize_vector(vector):
    """Normalize vector for cosine similarity with L2 distance"""
    if vector is None:
        return None
    
    vector = np.array(vector, dtype='float32')
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    normalized = vector / norms
    
    return normalized.flatten() if vector.shape[0] == 1 else normalized

# --- FAISS Search for Experiments ---
async def cosine_similarity_search_experiments(hypothesis_text, faiss_db_path='./Faiss/experiment_chunks_faiss.index',
                                               meta_db_path='./Faiss/experiment_chunks_meta.pkl', top_k=10, min_similarity=0.7):
    """Search FAISS database for experiment chunks with cosine similarity > min_similarity"""
    try:
        if not (os.path.exists(faiss_db_path) and os.path.exists(meta_db_path)):
            print(f"📂 No experiment FAISS database found")
            return []

        # Load FAISS index and metadata
        index = faiss.read_index(faiss_db_path)
        with open(meta_db_path, 'rb') as f:
            meta = pickle.load(f)

        # Flatten experiment chunks
        all_chunks = []
        if isinstance(meta, dict):
            for chunk_list in meta.values():
                if isinstance(chunk_list, list):
                    all_chunks.extend(chunk_list)
                else:
                    all_chunks.append(chunk_list)
        elif isinstance(meta, list):
            all_chunks = meta
        else:
            print(f"❌ Unexpected metadata format: {type(meta)}")
            return []

        if len(all_chunks) == 0:
            return []

        # Vectorize and normalize hypothesis
        hypothesis_vec = await vectorize(hypothesis_text)

        # Search FAISS
        search_k = min(top_k, index.ntotal, len(all_chunks))
        D, I = index.search(hypothesis_vec.reshape(1, -1), search_k)

        # Convert L2 distances to cosine similarity and filter
        relevant_chunks = []
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(all_chunks):
                cosine_sim = 1 - (dist * dist) / 2
                if cosine_sim >= min_similarity:
                    chunk = dict(all_chunks[idx])
                    chunk['cosine_similarity'] = float(cosine_sim)
                    chunk['faiss_index'] = int(idx)
                    relevant_chunks.append(chunk)

        relevant_chunks.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        print(f"🔍 Found {len(relevant_chunks)} relevant experiment chunks")
        
        for i, chunk in enumerate(relevant_chunks[:10]):
            title = chunk.get('paper_title', 'Unknown')[:50]
            print(f"   📄 {i+1}: {chunk['cosine_similarity']:.3f} - {title}...")
        
        return relevant_chunks

    except Exception as e:
        print(f"❌ FAISS experiment search failed: {e}")
        return []

# --- LLM Experiment Validation ---
async def llm_experiment_relevance_validation(chunk, hypothesis):
    """Use LLM to determine if experiment chunk is relevant to hypothesis"""
    try:
        chunk_text = chunk.get('text', '')
        paper_title = chunk.get('paper_title', 'Unknown')
        source_url = chunk.get('source_url', 'Unknown')
        
        prompt = f"""
        Is this experiment description relevant to the given hypothesis?
        
        Hypothesis: {hypothesis}
        
        Paper: {paper_title}
        Source: {source_url}
        Experiment: {chunk_text}
        
        Answer only "Yes" or "No" based on whether this experiment could provide insights, 
        methodology, or context relevant to testing or understanding the hypothesis.
        """
        
        response = await get_llm_response([
            {"role": "system", "content": "Determine experiment relevance. Answer only 'Yes' or 'No'."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        if response is None:
            return False
        
        return str(response).strip().lower().startswith('yes')
        
    except Exception as e:
        print(f"❌ LLM experiment validation failed: {e}")
        return False

# --- Filter Papers by Relevance ---
async def filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6):
    """Filter papers by title+abstract relevance to hypothesis using cosine similarity"""
    try:
        global arxiv_processor
        if not arxiv_processor:
            arxiv_processor = initialize_arxiv_processor()
        
        # Vectorize hypothesis
        hypothesis_embedding = await vectorize(hypothesis)
        
        relevant_papers = []
        for paper in papers:
            # Vectorize paper title + abstract
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            paper_text = f"{title} {abstract}"
            paper_embedding = await vectorize(paper_text)
            
            # Calculate cosine similarity between normalized vectors
            cos_sim = np.dot(hypothesis_embedding, paper_embedding)
            paper['relevance_score'] = float(cos_sim)
            
            if cos_sim >= min_relevance:
                relevant_papers.append(paper)
        
        # Sort by relevance score (highest first)
        relevant_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_papers
        
    except Exception as e:
        print(f"❌ Paper relevance filtering failed: {e}")
        return papers

# --- ArXiv Search by Abstracts ---
async def search_arxiv_by_abstracts(keywords, hypothesis, max_papers=20):
    """Search ArXiv and filter papers by abstract relevance to hypothesis"""
    print(f"🔍 Searching ArXiv for: {', '.join(keywords[:3])}")
    
    # Format ArXiv query
    search_terms = ' OR '.join(keywords[:3])
    formatted_query = search_terms.replace(' ', '+')
    url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results={max_papers}"

    # Fetch ArXiv results
    with libreq.urlopen(url) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    entries = root.findall('atom:entry', ns)
    
    print(f"📄 Found {len(entries)} papers from ArXiv")
    if len(entries) == 0:
        return []

    # Initialize processor
    global arxiv_processor
    if not arxiv_processor:
        arxiv_processor = initialize_arxiv_processor()

    # Extract basic paper info
    papers = []
    for i, entry in enumerate(entries):
        paper = arxiv_processor.extract_basic_paper_info(entry, ns, i+1)
        papers.append(paper)

    # Filter by abstract relevance to hypothesis
    relevant_papers = await filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6)
    print(f"📊 {len(relevant_papers)}/{len(papers)} papers relevant based on abstracts")
    
    # Return relevant papers
    return relevant_papers

# --- Download and Extract Experiments ---
async def download_and_extract_experiments(papers, hypothesis):
    """Download papers and extract experiment designs using LLM"""
    if not papers:
        return []
    
    global arxiv_processor
    if not arxiv_processor:
        arxiv_processor = initialize_arxiv_processor()
    
    print(f"📥 Downloading {len(papers)} papers and extracting experiments...")
    
    # Download paper contents
    downloaded_papers = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_paper = {executor.submit(arxiv_processor.download_paper_content, paper): paper for paper in papers}
        for future in future_to_paper:
            paper = future.result()
            if paper.get('content'):
                downloaded_papers.append(paper)
         

    # Extract experiments using LLM
    all_experiments = []
    for paper in downloaded_papers:
        experiments = await extract_experiments_from_paper(paper, hypothesis)
        all_experiments.extend(experiments)
    
    print(f"🧪 Extracted {len(all_experiments)} experiments from {len(downloaded_papers)} papers")
    return all_experiments

# --- Extract Experiments from Paper ---
async def extract_experiments_from_paper(paper, hypothesis):
    """Extract experiment designs from paper using LLM"""
    try:
        content = paper.get('content', '')
        title = paper.get('title', 'Unknown')
        source_url = paper.get('pdf_url', paper.get('id', 'Unknown'))
        
        prompt = f"""
        Extract all experiment designs from this research paper. For each experiment, identify:
        
        1. Research Goal: What the experiment aimed to achieve
        2. Variables: Independent and dependent variables
        3. Full Experiment Design: Complete methodology, approach, procedures, measurements, and analysis methods
        
        Extract the experiment details exactly as described in the paper without modification.
        Format each experiment as:
        
        EXPERIMENT N:
        Research Goal: [exact goal from paper]
        Variables: [exact variables from paper]
        Experiment Design: [complete experimental methodology as described]
        
        Paper Title: {title}
        Paper Content: {content[:15000]}  # Limit content to avoid token limits
        
        Focus on experiments that could be relevant to this hypothesis: {hypothesis}
        """
        
        response = await get_llm_response([
            {"role": "system", "content": "Extract experiment designs exactly as described in research papers."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        if not response:
            return []
        
        # Parse extracted experiments
        experiments = parse_extracted_experiments(response, paper, source_url)
        return experiments
        
    except Exception as e:
        print(f"❌ Experiment extraction failed for {title}: {e}")
        return []

# --- Parse Extracted Experiments ---
def parse_extracted_experiments(llm_response, paper, source_url):
    """Parse LLM response into structured experiment data"""
    experiments = []
    
    try:
        # Split by experiment markers
        sections = llm_response.split('EXPERIMENT ')
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if not section.strip():
                continue
                
            # Extract components
            lines = section.strip().split('\n')
            research_goal = ""
            variables = ""
            experiment_design = ""
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('Research Goal:'):
                    current_section = 'goal'
                    research_goal = line.replace('Research Goal:', '').strip()
                elif line.startswith('Variables:'):
                    current_section = 'variables'
                    variables = line.replace('Variables:', '').strip()
                elif line.startswith('Experiment Design:'):
                    current_section = 'design'
                    experiment_design = line.replace('Experiment Design:', '').strip()
                elif current_section == 'goal' and line:
                    research_goal += " " + line
                elif current_section == 'variables' and line:
                    variables += " " + line
                elif current_section == 'design' and line:
                    experiment_design += " " + line
            
            # Create experiment chunk
            if research_goal or experiment_design:
                full_text = f"Research Goal: {research_goal}\nVariables: {variables}\nExperiment Design: {experiment_design}"
                
                experiment = {
                    'text': full_text,
                    'research_goal': research_goal,
                    'variables': variables,
                    'experiment_design': experiment_design,
                    'paper_title': paper.get('title', 'Unknown'),
                    'paper_id': paper.get('id', 'Unknown'),
                    'source_url': source_url,
                    'experiment_number': i,
                    'type': 'experiment'
                }
                experiments.append(experiment)
    
    except Exception as e:
        print(f"❌ Failed to parse experiments: {e}")
    
    return experiments

# --- Store Experiments in FAISS ---
async def store_experiments_in_faiss(experiments, faiss_db_path='./Faiss/experiment_chunks_faiss.index',
                                     meta_db_path='./Faiss/experiment_chunks_meta.pkl'):
    """Store experiment chunks in FAISS database"""
    if not experiments:
        return False
    
    try:
        # Load or create FAISS database
        os.makedirs('Faiss', exist_ok=True)
        embedding_dim = 384
        
        if os.path.exists(faiss_db_path) and os.path.exists(meta_db_path):
            faiss_db = faiss.read_index(faiss_db_path)
            with open(meta_db_path, 'rb') as f:
                faiss_meta = pickle.load(f)
            if not isinstance(faiss_meta, dict):
                faiss_meta = {}
        else:
            faiss_db = faiss.IndexFlatL2(embedding_dim)
            faiss_meta = {}
        
        # Process each experiment
        stored_count = 0
        for experiment in experiments:
            # Generate embedding for experiment text
            embedding = await vectorize(experiment['text'])
            if embedding is None:
                continue
            
            # Add to FAISS
            faiss_db.add(embedding.reshape(1, -1))
            
            # Store metadata
            paper_id = experiment['paper_id']
            if paper_id not in faiss_meta:
                faiss_meta[paper_id] = []
            
            faiss_meta[paper_id].append({
                'text': experiment['text'],
                'research_goal': experiment['research_goal'],
                'variables': experiment['variables'],
                'experiment_design': experiment['experiment_design'],
                'paper_title': experiment['paper_title'],
                'paper_id': paper_id,
                'source_url': experiment['source_url'],
                'experiment_number': experiment['experiment_number'],
                'type': 'experiment'
            })
            stored_count += 1
        
        # Save FAISS index and metadata
        faiss.write_index(faiss_db, faiss_db_path)
        with open(meta_db_path, 'wb') as f:
            pickle.dump(faiss_meta, f)
        
        print(f"💾 Stored {stored_count} experiments in FAISS database")
        return True
        
    except Exception as e:
        print(f"❌ Failed to store experiments in FAISS: {e}")
        return False

# --- Node Functions for LangGraph ---

# Node 1: Extract keywords
async def node_extract_keywords(state):
    hypothesis = state['hypothesis']
    keywords = await extract_keywords_from_hypothesis(hypothesis)
    state['keywords'] = keywords
    print(f"🔑 Extracted keywords: {keywords}")
    return state

# Node 2: Search FAISS for relevant experiment chunks
async def node_faiss_retrieve(state):
    hypothesis = state['hypothesis']
    chunks = await cosine_similarity_search_experiments(hypothesis, min_similarity=0.7)
    state['retrieved_chunks'] = chunks
    print(f"📚 Retrieved {len(chunks)} chunks from FAISS")
    return state

# Node 3: LLM validation of experiment chunks
async def node_llm_validate(state):
    hypothesis = state['hypothesis']
    chunks = state['retrieved_chunks']
    validated = []
    print(f"🤖 Validating {len(chunks)} chunks...")
    for chunk in chunks:
        is_relevant = await llm_experiment_relevance_validation(chunk, hypothesis)
        if is_relevant:
            validated.append(chunk)
    state['validated_chunks'] = validated
    return state

# Node 4: Search ArXiv based on abstracts if insufficient chunks
async def node_search_arxiv(state):
    keywords = state['keywords']
    hypothesis = state['hypothesis']
    papers = await search_arxiv_by_abstracts(keywords, hypothesis)
    state['arxiv_papers'] = papers
    
    if papers:
        # Download and extract experiments
        experiments = await download_and_extract_experiments(papers, hypothesis)
        state['extracted_experiments'] = experiments
        
        # Store in FAISS
        if experiments:
            await store_experiments_in_faiss(experiments)
    else:
        state['extracted_experiments'] = []
    
    return state

# Node 5: Aggregate all results
async def node_aggregate(state):
    validated_chunks = state.get('validated_chunks', [])
    extracted_experiments = state.get('extracted_experiments', [])
    
    # Combine existing validated chunks with newly extracted experiments
    all_results = validated_chunks + extracted_experiments
    state['results'] = all_results
    return state

# --- Build LangGraph workflow ---
async def build_experiment_search_workflow():
    g = StateGraph(dict)
    g.add_node('extract_keywords', node_extract_keywords)
    g.add_node('faiss_retrieve', node_faiss_retrieve)
    g.add_node('llm_validate', node_llm_validate)
    g.add_node('search_arxiv', node_search_arxiv)
    g.add_node('aggregate', node_aggregate)

    # Flow: extract keywords -> search FAISS -> validate
    g.add_edge('extract_keywords', 'faiss_retrieve')
    g.add_edge('faiss_retrieve', 'llm_validate')

    # Decision: if enough validated chunks, aggregate; else search ArXiv
    async def check_sufficient_chunks(state):
        validated = state.get('validated_chunks', [])
        if len(validated) >= 5:  # Sufficient experiments found
            return 'aggregate'
        else:
            return 'search_arxiv'
    
    g.add_conditional_edges('llm_validate', check_sufficient_chunks)

    # ArXiv path: search -> aggregate
    g.add_edge('search_arxiv', 'aggregate')

    # End
    g.add_edge('aggregate', END)

    g.set_entry_point('extract_keywords')
    return g

if __name__ == "__main__":
    hypothesis = input("Enter your research hypothesis: ").strip()
    workflow = build_experiment_search_workflow()
    state = {'hypothesis': hypothesis}
    app = workflow.compile()
    results = app.invoke(state)
