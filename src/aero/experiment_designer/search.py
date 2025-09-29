import os
import faiss
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import urllib.request as libreq
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from aero.utils.arxiv_paper_utils import ArxivPaperProcessor
from aero.experiment_designer.utils import get_llm_response, stream_writer
from langgraph.graph import StateGraph, END
import asyncio
from sentence_transformers import SentenceTransformer
import requests

# --- Dataset Links Searcher ---
# List of known dataset repository domains
KNOWN_DATASET_DOMAINS = [
    "openneuro.org",
    "physionet.org",
    "kaggle.com",
    "zenodo.org",
    "figshare.com",
    "datadryad.org",
    "osf.io",
    "data.gov",
    "datahub.io",
    "uci.edu",
    "archive.ics.uci.edu",
    "bbci.de"  
]

def is_dataset_link(url: str) -> bool:
    return any(domain in url for domain in KNOWN_DATASET_DOMAINS)

def search_dataset_online(dataset_name: str, num_results: int = 5):
    """
    Search for dataset source/download links using Google Custom Search API.
    Only returns links from known dataset repositories.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("CX")
    if not api_key or not cx:
        raise Exception("GOOGLE_API_KEY and CX must be set in the .env file.")

    # Focus the query on dataset repositories and download pages
    query = f'"{dataset_name}" dataset site:(' + " OR ".join(KNOWN_DATASET_DOMAINS) + ')'
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "num": min(num_results * 2, 10)  # Get more results to filter
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Google API error: {response.status_code}, {response.text}")
    
    data = response.json()
    results = []
    for item in data.get("items", []):
        link = item.get("link", "")
        if is_dataset_link(link):
            results.append({
                "title": item.get("title"),
                "link": link,
                "snippet": item.get("snippet")
            })
        if len(results) >= num_results:
            break
    
    return results

# --- ArXiv Processor Initialization ---
def initialize_arxiv_processor():
    """Initialize ArXiv processor with LLM client"""
    
    primary_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

    PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")
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
    return keywords[:5]

def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name, device='cpu')

# --- Vectorization Helper Functions ---
async def vectorize(text, arxiv_processor=None, embedding_model=None):
    """Vectorize text using sentence-transformers"""
    if arxiv_processor is None:
        arxiv_processor = initialize_arxiv_processor()
    
    if embedding_model is None:
        embedding_model = get_embedding_model()
    embedding = embedding_model.encode([text], convert_to_tensor=False, show_progress_bar=False)

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
                                               meta_db_path='./Faiss/experiment_chunks_meta.pkl', top_k=10, min_similarity=0.7, writer=None):
    """Search FAISS database for experiment chunks with cosine similarity > min_similarity"""
    try:
        if not (os.path.exists(faiss_db_path) and os.path.exists(meta_db_path)):
            await stream_writer(f"No experiment FAISS database found", writer=writer, stream_mode="custom")
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
            await stream_writer(f"Unexpected metadata format: {type(meta)}", writer=writer, stream_mode="custom")
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
        await stream_writer(f"Found {len(relevant_chunks)} relevant experiment chunks", writer=writer, stream_mode="custom")
        
        for i, chunk in enumerate(relevant_chunks[:10]):
            title = chunk.get('paper_title', 'Unknown')
            await stream_writer(f"   📄 {i+1}: {chunk['cosine_similarity']:.3f} - {title}...", writer=writer, stream_mode="custom")

        return relevant_chunks

    except Exception as e:
        await stream_writer(f"FAISS experiment search failed: {e}", writer=writer, stream_mode="custom")
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
        return False

# --- Filter Papers by Relevance ---
async def filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6, arxiv_processor=None):
    """Filter papers by title+abstract relevance to hypothesis using cosine similarity"""
    try:
        if arxiv_processor is None:
            arxiv_processor = initialize_arxiv_processor()
        
        # Vectorize hypothesis
        hypothesis_embedding = await vectorize(hypothesis, arxiv_processor=arxiv_processor)

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
        return papers

# --- ArXiv Search by Abstracts ---
async def search_arxiv_by_abstracts(keywords, hypothesis, max_papers=20, arxiv_processor=None):
    """Search ArXiv and filter papers by abstract relevance to hypothesis"""
    #stream_writer(f"🔍 Searching ArXiv for: {', '.join(keywords[:3])}")
    await asyncio.sleep(0.5)  # Allow stream message to appear before LLM calls
    
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

    #stream_writer(f"📄 Found {len(entries)} papers from ArXiv")
    await asyncio.sleep(0.5)
    if len(entries) == 0:
        return []

    # Initialize processor
    if arxiv_processor is None:
        arxiv_processor = initialize_arxiv_processor()

    # Extract basic paper info
    papers = []
    for i, entry in enumerate(entries):
        paper = arxiv_processor.extract_basic_paper_info(entry, ns, i+1)
        papers.append(paper)

    # Filter by abstract relevance to hypothesis
    relevant_papers = await filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6, arxiv_processor=arxiv_processor)
    #stream_writer(f"📊 {len(relevant_papers)}/{len(papers)} papers relevant based on abstracts")
    await asyncio.sleep(0.5)
    
    # Return relevant papers
    return relevant_papers

# --- Download and Extract Experiments ---
async def download_and_extract_experiments(papers, hypothesis, arxiv_processor=None):
    """Download papers and extract experiment designs using LLM"""
    if not papers:
        return []
    if arxiv_processor is None:
        arxiv_processor = initialize_arxiv_processor()
    #stream_writer(f"📥 Downloading {len(papers)} papers and extracting experiments...")
    await asyncio.sleep(0.5)

    # Download paper contents (keep as is)
    downloaded_papers = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_paper = {executor.submit(arxiv_processor.download_paper_content, paper): paper for paper in papers}
        for future in future_to_paper:
            paper = future.result()
            if paper.get('content'):
                downloaded_papers.append(paper)

    # Parallelize experiment extraction
    tasks = [extract_experiments_from_paper(paper, hypothesis) for paper in downloaded_papers]
    all_experiments_nested = await asyncio.gather(*tasks)
    all_experiments = [exp for sublist in all_experiments_nested for exp in sublist]
    #stream_writer(f"🧪 Extracted {len(all_experiments)} experiments from {len(downloaded_papers)} papers")
    await asyncio.sleep(0.5)
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
        return []

# --- Parse Extracted Experiments ---
def parse_extracted_experiments(llm_response, paper, source_url):
    """Parse LLM response into structured experiment data"""
    experiments = []


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

        #stream_writer(f"💾 Stored {stored_count} experiments in FAISS database")
        await asyncio.sleep(0.5)
        return True
        
    except Exception as e:
        #stream_writer(f"❌ Failed to store experiments in FAISS: {e}")
        await asyncio.sleep(0.5)
        return False

# --- Node: Retrieve from FAISS ---
async def node_faiss_retrieve(state, writer=None):
    hypothesis = state['hypothesis']
    chunks = await cosine_similarity_search_experiments(hypothesis, min_similarity=0.7, writer=writer)
    state['retrieved_chunks'] = chunks
    await stream_writer(f"Retrieved {len(chunks)} chunks from FAISS", writer=writer, stream_mode="custom")
    return state

# --- Node: Extract keywords (for ArXiv search) ---
async def node_extract_keywords(state, writer=None):
    hypothesis = state['hypothesis']
    keywords = await extract_keywords_from_hypothesis(hypothesis)
    state['keywords'] = keywords
    #stream_writer(f"🔑 Extracted keywords: {keywords}")
    await asyncio.sleep(0.5)
    return state

# --- Node: Search ArXiv and add experiments to chunks ---
async def node_search_arxiv(state):
    keywords = state['keywords']
    hypothesis = state['hypothesis']
    papers = await search_arxiv_by_abstracts(keywords, hypothesis)
    state['arxiv_papers'] = papers

    if papers:
        experiments = await download_and_extract_experiments(papers, hypothesis)
        state['extracted_experiments'] = experiments
        # Add to retrieved_chunks for validation
        state['retrieved_chunks'] = state.get('retrieved_chunks', []) + experiments
        # Optionally store in FAISS
        if experiments:
            await store_experiments_in_faiss(experiments)
    else:
        state['extracted_experiments'] = []
    return state

# --- Node: LLM validation of all chunks ---
async def node_llm_validate(state):
    hypothesis = state['hypothesis']
    chunks = state.get('retrieved_chunks', [])
    #stream_writer(f"🤖 Validating {len(chunks)} chunks...")
    await asyncio.sleep(0.5)  # Allow stream message to appear before LLM calls
    tasks = [llm_experiment_relevance_validation(chunk, hypothesis) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    validated = [chunk for chunk, is_relevant in zip(chunks, results) if is_relevant]
    state['validated_chunks'] = validated
    return state

# --- Node: Aggregate ---
async def node_aggregate(state):
    state['results'] = state.get('validated_chunks', [])
    return state

# --- Build LangGraph workflow ---
async def build_experiment_search_workflow(writer=None):
    g = StateGraph(dict)
    async def faiss_retrieve_node(state):
        return await node_faiss_retrieve(state, writer=writer)

    g.add_node('faiss_retrieve', faiss_retrieve_node)
    g.add_node('extract_keywords', node_extract_keywords)
    g.add_node('search_arxiv', node_search_arxiv)
    g.add_node('llm_validate', node_llm_validate)
    g.add_node('aggregate', node_aggregate)

    # Start: faiss_retrieve
    # If enough chunks, skip arxiv; else, extract keywords and search arxiv
    async def check_faiss_chunks(state):
        chunks = state.get('retrieved_chunks', [])
        if len(chunks) >= 3:
            return 'llm_validate'
        else:
            return 'extract_keywords'

    g.add_conditional_edges('faiss_retrieve', check_faiss_chunks)
    g.add_edge('extract_keywords', 'search_arxiv')
    g.add_edge('search_arxiv', 'llm_validate')
    g.add_edge('llm_validate', 'aggregate')
    g.add_edge('aggregate', END)

    g.set_entry_point('faiss_retrieve')
    return g