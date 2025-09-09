#!/usr/bin/env python3
"""
Two-Level Tree-Based Experimental Design System (Enhanced)
=========================================================

This script implements a comprehensive workflow:
1. User input → breakdown into research goal, hypothesis and relevant information (OpenAI API)
2. For each hypothesis: generate parent nodes (general experimental strategies) 
3. Expand promising parents → child nodes (full implementation steps with details)
4. Score and prune at each level using AB-MCTS-A adaptive thresholds
5. Output complete tree structure to markdown for supervision

Configuration:
    Reads from environment variables or `.env` file:
        - OPENAI_API_KEY
        - BASE_URL (default: https://agents.aetherraid.dev)
        - DEFAULT_MODEL (default: gemini/gemini-2.5-flash)
"""

import os
import json
import re
import asyncio
import logging
import faiss
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import urllib.request as libreq
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from arxiv_paper_utils import ArxivPaperProcessor

# --- Load environment variables ---
load_dotenv()

# Setup logging - suppress noisy logs
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# Initialize clients and ArXiv processor
primary_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://agents.aetherraid.dev")
)
PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")

# Initialize ArXiv processor for literature grounding
arxiv_processor = None

# Global API cost tracking
total_api_cost = 0.0
api_call_count = 0

# Progress tracking system
class ProgressTracker:
    def __init__(self):
        self.tasks = {}
        self.completed = {}
        self.lock = asyncio.Lock()
    
    async def add_task(self, task_id, description):
        async with self.lock:
            self.tasks[task_id] = description
            self.completed[task_id] = False
    
    async def complete_task(self, task_id, result=None):
        async with self.lock:
            self.completed[task_id] = True
            if result:
                self.tasks[task_id] = f"{self.tasks[task_id]} ✅"
    
    async def get_status(self):
        async with self.lock:
            total = len(self.tasks)
            done = sum(1 for completed in self.completed.values() if completed)
            return done, total, dict(self.tasks), dict(self.completed)

# Global progress tracker
progress_tracker = ProgressTracker()

def calculate_api_cost(messages, response_text, model_name):
    """Calculate API cost using tiktoken"""
    global total_api_cost
    
    try:
        # Get appropriate encoding for the model
        if "gpt-4" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
            input_cost_per_1k = 0.03  # $0.03 per 1K input tokens
            output_cost_per_1k = 0.06  # $0.06 per 1K output tokens
        elif "gpt-3.5" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_cost_per_1k = 0.0015  # $0.0015 per 1K input tokens
            output_cost_per_1k = 0.002   # $0.002 per 1K output tokens
        elif "gemini" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as approximation
            input_cost_per_1k = 0.00075   # Gemini 2.0 Flash pricing (approximate)
            output_cost_per_1k = 0.003    # Gemini 2.0 Flash pricing (approximate)
        else:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Default fallback
            input_cost_per_1k = 0.002
            output_cost_per_1k = 0.002
        
        # Calculate input tokens
        input_text = ""
        for message in messages:
            input_text += message.get("content", "")
        
        input_tokens = len(encoding.encode(input_text))
        output_tokens = len(encoding.encode(response_text))
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        total_api_cost += total_cost
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': total_cost,
            'cumulative_cost': total_api_cost
        }
        
    except Exception as e:
        print(f"⚠️ Cost calculation error: {e}")
        return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'cumulative_cost': total_api_cost}

def initialize_arxiv_processor():
    """Initialize ArXiv processor with LLM client"""
    global arxiv_processor
    if arxiv_processor is None:
        arxiv_processor = ArxivPaperProcessor(primary_client, PRIMARY_MODEL)
    return arxiv_processor

def clean_json_string(text):
    """Clean JSON string by removing control characters and markdown"""
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

async def display_progress():
    """Display live progress updates"""
    while True:
        done, total, tasks, completed = await progress_tracker.get_status()
        if total == 0:
            await asyncio.sleep(0.5)
            continue
        
        # Clear screen and show progress
        print(f"\r🔄 Processing: {done}/{total} tasks completed", end="", flush=True)
        
        if done == total:
            print(f"\r🎉 All {total} tasks completed!           ")
            break
        
        await asyncio.sleep(0.5)

async def get_llm_response(messages, temperature=0.2, max_tokens=None):
    """Get LLM response using OpenAI API with cost tracking"""
    global total_api_cost, api_call_count
    
    api_call_count += 1
    
    # Add minimal delay to prevent rate limiting (reduced from 0.05 to 0.02)
    await asyncio.sleep(0.02)
    
    try:
        kwargs = {"model": PRIMARY_MODEL, "messages": messages, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        # Retry with exponential backoff for rate limits
        for attempt in range(3):
            try:
                response = await primary_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None:
                    content = "No response"
                
                # Calculate actual cost using tiktoken
                cost_info = calculate_api_cost(messages, content, PRIMARY_MODEL)
                
                return content.strip()
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    wait_time = (2 ** attempt) + 0.5
                    print(f"⏳ Rate limit hit, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif "budget" in str(e).lower():
                    print(f"💰 Budget exceeded: {e}")
                    raise e
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"OpenAI API failed: {e}")
        return "Error: API call failed"

class ExperimentNode:
    """Represents a node in the two-level experiment tree"""
    def __init__(self, description, depth=0, node_type="strategy"):
        self.description = description
        self.depth = depth
        self.node_type = node_type  # "strategy", "implementation"
        self.feasibility_score = 0.0
        self.impact_score = 0.0
        self.combined_score = 0.0
        self.children = []
        self.literature_context = ""  # Store literature context
        self.source_links = []  # Store source links
        
    def calculate_score(self, w_feasibility=0.6, w_impact=0.4):
        """Calculate weighted combination score"""
        self.combined_score = w_feasibility * self.feasibility_score + w_impact * self.impact_score
        return self.combined_score

# Literature Grounding Functions
async def vectorize_hypothesis(hypothesis):
    """Vectorize hypothesis using sentence-transformers"""
    global arxiv_processor
    if not arxiv_processor:
        arxiv_processor = initialize_arxiv_processor()
    
    # Wait for embedding model to be ready (it loads in background)
    model = arxiv_processor._get_embedding_model()
    if model is None:
        return None
    
    try:
        embedding = model.encode([hypothesis], convert_to_tensor=False, show_progress_bar=False)
        return embedding[0]  # Return first (and only) embedding
    except Exception as e:
        print(f"❌ Failed to vectorize hypothesis: {e}")
        return None

async def cosine_similarity_search(hypothesis_vector, faiss_db_path='Faiss/arxiv_chunks_faiss.index', 
                                 meta_db_path='Faiss/arxiv_chunks_meta.pkl', top_k=10, min_similarity=0.7):
    """Search FAISS database for chunks with cosine similarity > min_similarity"""
    try:
        if not (os.path.exists(faiss_db_path) and os.path.exists(meta_db_path)):
            print(f"📂 No FAISS database found")
            return []
        
        # Load FAISS index and metadata
        index = faiss.read_index(faiss_db_path)
        with open(meta_db_path, 'rb') as f:
            meta = pickle.load(f)
        
        print(f"📂 FAISS database info: {index.ntotal} vectors, metadata type: {type(meta)}")
        
        # Flatten all chunk metadata - handle both dict and list formats
        all_chunks = []
        if isinstance(meta, dict):
            for paper_id, chunk_list in meta.items():
                if isinstance(chunk_list, list):
                    all_chunks.extend(chunk_list)
                else:
                    all_chunks.append(chunk_list)
        elif isinstance(meta, list):
            all_chunks = meta
        else:
            print(f"❌ Unexpected metadata format: {type(meta)}")
            return []
        
        print(f"📊 Total chunks available: {len(all_chunks)}")
        
        # Check for index/metadata mismatch
        if index.ntotal != len(all_chunks):
            print(f"⚠️ Index/metadata mismatch: {index.ntotal} vectors vs {len(all_chunks)} chunks")
            print(f"🔧 Rebuilding FAISS database to fix inconsistency...")
            
            # Rebuild FAISS index to match metadata
            if all_chunks:
                # Get embedding model to rebuild embeddings
                global arxiv_processor
                if not arxiv_processor:
                    arxiv_processor = initialize_arxiv_processor()
                
                model = arxiv_processor._get_embedding_model()
                if model is not None:
                    try:
                        # Re-embed all chunks
                        texts = [chunk.get('text', '') for chunk in all_chunks]
                        embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
                        
                        # Create new index
                        new_index = faiss.IndexFlatL2(embeddings.shape[1])
                        new_index.add(embeddings.astype('float32'))
                        
                        # Save corrected index
                        faiss.write_index(new_index, faiss_db_path)
                        index = new_index
                        
                        print(f"✅ Rebuilt FAISS index with {index.ntotal} vectors matching {len(all_chunks)} chunks")
                    except Exception as e:
                        print(f"❌ Failed to rebuild index: {e}")
                        return []
                else:
                    print(f"❌ Cannot rebuild index - embedding model not available")
                    return []
        
        if len(all_chunks) == 0:
            return []
        
        # Prepare hypothesis vector for search
        query_vec = np.array(hypothesis_vector, dtype='float32')
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # Adjust dimension if needed
        if query_vec.shape[1] != index.d:
            if query_vec.shape[1] > index.d:
                query_vec = query_vec[:, :index.d]
            else:
                padded = np.zeros((1, index.d), dtype='float32')
                padded[:, :query_vec.shape[1]] = query_vec
                query_vec = padded
        
        # Search FAISS with proper error handling
        try:
            # Limit search to available chunks if there's a mismatch
            search_k = min(top_k, index.ntotal, len(all_chunks))
            D, I = index.search(query_vec, search_k)
        except Exception as e:
            print(f"❌ FAISS search operation failed: {e}")
            return []
        
        # Validate search results
        if D.shape[0] == 0 or I.shape[0] == 0:
            print("❌ FAISS search returned empty results")
            return []
        
        # Convert L2 distances to cosine similarities and filter
        relevant_chunks = []
        valid_indices_count = 0
        invalid_indices_count = 0
        
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if 0 <= idx < len(all_chunks):
                try:
                    # Convert L2 distance to cosine similarity
                    cosine_sim = 1 - (dist ** 2) / 2
                    
                    if cosine_sim >= min_similarity:
                        chunk = dict(all_chunks[idx])
                        chunk['cosine_similarity'] = float(cosine_sim)
                        chunk['faiss_index'] = int(idx)
                        relevant_chunks.append(chunk)
                        valid_indices_count += 1
                except Exception as e:
                    print(f"⚠️ Error processing chunk {idx}: {e}")
                    continue
            else:
                invalid_indices_count += 1
        
        if invalid_indices_count > 0:
            print(f"⚠️ Skipped {invalid_indices_count} invalid indices, processed {valid_indices_count} valid chunks")
        
        # Sort by similarity (highest first)
        relevant_chunks.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks[:3]):
                title = chunk.get('paper_title', 'Unknown')[:50]
                print(f"   📄 {i+1}: {chunk['cosine_similarity']:.3f} - {title}...")
        
        return relevant_chunks
        
    except Exception as e:
        print(f"❌ FAISS search failed: {e}")
        return []

async def llm_relevance_validation(chunk, hypothesis):
    """Use LLM to determine if chunk is relevant to hypothesis (Yes/No)"""
    try:
        chunk_text = chunk.get('text', '')[:500]  # First 500 chars
        paper_title = chunk.get('paper_title', 'Unknown')
        
        prompt = f"""
        Is this research paper chunk relevant to the given hypothesis?
        
        Hypothesis: {hypothesis}
        
        Paper Title: {paper_title}
        Chunk Text: {chunk_text}
        
        Answer only "Yes" or "No" based on whether the chunk contains information that could help validate, refute, or provide context for the hypothesis.
        """
        
        response = await get_llm_response([
            {"role": "system", "content": "Determine relevance. Answer only 'Yes' or 'No'."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        # Handle None response safely
        if response is None:
            return False
        
        return str(response).strip().lower().startswith('yes')
        
    except Exception as e:
        print(f"❌ LLM validation failed: {e}")
        return False

async def extract_keywords_from_hypothesis(hypothesis):
    """Extract search keywords from hypothesis"""
    try:
        prompt = f"""
        Extract 3-5 key research keywords from this hypothesis for academic literature search.
        Use simple, standard academic terms. Avoid special characters and long phrases.
        
        Hypothesis: {hypothesis}
        
        Return only keywords separated by commas:
        """
        
        response = await get_llm_response([
            {"role": "system", "content": "Extract simple academic keywords for literature search."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        keywords = [kw.strip() for kw in response.replace('*', '').split(',') if kw.strip()]
        return keywords[:5]  # Limit to 5 keywords
        
    except Exception as e:
        print(f"❌ Keyword extraction failed: {e}")
        return ["machine learning"]

async def filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6):
    """Filter papers by title+abstract relevance to hypothesis using cosine similarity"""
    try:
        global arxiv_processor
        if not arxiv_processor:
            arxiv_processor = initialize_arxiv_processor()
        
        model = arxiv_processor._get_embedding_model()
        if model is None:
            print("⚠️ Embedding model not available, returning all papers")
            return papers
        
        # Vectorize hypothesis
        hypothesis_embedding = model.encode([hypothesis], convert_to_tensor=False, show_progress_bar=False)[0]
        
        relevant_papers = []
        for paper in papers:
            try:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                paper_text = f"{title} {abstract}"
                
                if len(paper_text.strip()) < 10:  # Skip papers with minimal content
                    continue
                
                # Vectorize paper title + abstract
                paper_embedding = model.encode([paper_text], convert_to_tensor=False, show_progress_bar=False)[0]
                
                # Calculate cosine similarity
                import numpy as np
                cos_sim = np.dot(hypothesis_embedding, paper_embedding) / (
                    np.linalg.norm(hypothesis_embedding) * np.linalg.norm(paper_embedding)
                )
                
                paper['relevance_score'] = float(cos_sim)
                
                if cos_sim >= min_relevance:
                    relevant_papers.append(paper)
                    
            except Exception as e:
                print(f"⚠️ Error scoring paper relevance: {e}")
                continue
        
        # Sort by relevance score (highest first)
        relevant_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_papers
        
    except Exception as e:
        print(f"❌ Paper relevance filtering failed: {e}")
        return papers

async def search_arxiv_and_add_to_faiss(keywords, hypothesis, faiss_db_path, meta_db_path, max_papers=3):
    """Search ArXiv API and add papers to FAISS database"""
    global arxiv_processor
    
    try:
        print(f"   🔍 Searching ArXiv for: {', '.join(keywords[:3])}")
        
        # Format ArXiv query
        search_terms = ' OR '.join(keywords[:3])
        formatted_query = search_terms.replace(' ', '+')
        url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results=15"
        
        # Get ArXiv results
        with libreq.urlopen(url) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = root.findall('atom:entry', ns)
        print(f"   📄 Found {len(entries)} papers from ArXiv")
        
        if len(entries) == 0:
            return False
        
        # Extract and rank papers with initial relevance filtering
        papers = []
        for i, entry in enumerate(entries, 1):
            paper_info = arxiv_processor.extract_basic_paper_info(entry, ns, i)
            papers.append(paper_info)
        
        # Score papers by title + abstract relevance before PDF download
        print(f"   🔍 Scoring {len(papers)} papers for relevance...")
        relevant_papers = await filter_papers_by_relevance(papers, hypothesis, min_relevance=0.6)
        
        print(f"   📊 {len(relevant_papers)}/{len(papers)} papers above relevance threshold")
        
        # Download and process top relevant papers
        top_papers = relevant_papers[:max_papers]
        processed_papers = []
        
        print(f"   📥 Downloading top {len(top_papers)} papers...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_paper = {
                executor.submit(arxiv_processor.download_paper_content, paper): paper 
                for paper in top_papers
            }
            
            for future in future_to_paper:
                updated_paper = future.result()
                if updated_paper.get('content'):
                    processed_papers.append(updated_paper)
                    title = updated_paper.get('title', 'Unknown')[:40]
                    print(f"   ✅ Downloaded: {title}...")
        
        if processed_papers:
            await add_papers_to_faiss(processed_papers, faiss_db_path, meta_db_path)
            print(f"   💾 Added {len(processed_papers)} papers to FAISS database")
            return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ ArXiv search failed: {e}")
        return False

async def add_papers_to_faiss(papers, faiss_db_path, meta_db_path):
    """Add papers to FAISS database"""
    global arxiv_processor
    
    try:
        os.makedirs('Faiss', exist_ok=True)
        embedding_dim = 384
        
        # Load or create FAISS database
        if os.path.exists(faiss_db_path) and os.path.exists(meta_db_path):
            faiss_db = faiss.read_index(faiss_db_path)
            with open(meta_db_path, 'rb') as f:
                faiss_meta = pickle.load(f)
            
            # Ensure faiss_meta is a dictionary
            if not isinstance(faiss_meta, dict):
                print(f"⚠️ Converting metadata from {type(faiss_meta)} to dict")
                faiss_meta = {}
        else:
            faiss_db = faiss.IndexFlatL2(embedding_dim)
            faiss_meta = {}
        
        # Process each paper
        papers_processed = 0
        for paper in papers:
            paper_id = paper.get('id')
            if paper_id and paper_id not in faiss_meta:
                title = paper.get('title', 'Unknown')[:40]
                print(f"   🧩 Processing: {title}...")
                try:
                    chunk_meta = await arxiv_processor.chunk_and_embed(
                        paper, faiss_db=faiss_db, embedding_dim=embedding_dim
                    )
                    if chunk_meta and len(chunk_meta) > 0:
                        faiss_meta[paper_id] = chunk_meta
                        papers_processed += 1
                        print(f"   ✅ Added {len(chunk_meta)} chunks for paper {paper_id}")
                    else:
                        print(f"   ⚠️ No chunks generated for paper {paper_id}")
                except Exception as e:
                    print(f"   ❌ Failed to process paper {paper_id}: {e}")
                    continue
            else:
                if not paper_id:
                    print(f"   ⚠️ Skipping paper with no ID")
                else:
                    print(f"   ⚠️ Paper {paper_id} already exists in database")
        
        print(f"   📊 Successfully processed {papers_processed}/{len(papers)} papers")
        
        # Save updated database
        faiss.write_index(faiss_db, faiss_db_path)
        with open(meta_db_path, 'wb') as f:
            pickle.dump(faiss_meta, f)
        
    except Exception as e:
        print(f"   ❌ Failed to add papers to FAISS: {e}")

async def extract_context_and_links(chunks):
    """Extract literature context and source links from validated chunks"""
    if not chunks:
        return "", []
    
    contexts = []
    source_links = []
    
    for chunk in chunks:
        text = chunk.get('text', '')[:300]  # First 300 chars
        paper_title = chunk.get('paper_title', 'Unknown')
        
        # Priority order for citation URLs: abs_url > paper_url > url > construct from paper_id
        paper_url = chunk.get('abs_url', chunk.get('paper_url', chunk.get('url', '')))
        
        # If URL is missing or empty, try to construct it from paper_id
        if not paper_url or paper_url == 'error':
            paper_id = chunk.get('paper_id', '')
            if paper_id and paper_id != 'error':
                # Clean paper_id (remove version info if present)
                clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
                paper_url = f"https://arxiv.org/abs/{clean_id}"
            else:
                # Last resort: set empty
                paper_url = ""
        
        contexts.append(f"From '{paper_title}': {text}")
        source_links.append(f"[{paper_title}]({paper_url})")
    
    combined_context = " | ".join(contexts)
    return combined_context, source_links

async def literature_grounding_workflow(hypothesis, max_sources=3, min_similarity=0.7):
    """
    Complete literature grounding workflow:
    1. Vectorize hypothesis
    2. Cosine similarity search in FAISS (>0.7)
    3. LLM relevance validation (batch processing to reduce API calls)
    4. Limit to max 3 unique sources, prioritizing higher similarity
    5. Only if <3 chunks found, search ArXiv and retry once
    """
    global arxiv_processor
    
    if not arxiv_processor:
        arxiv_processor = initialize_arxiv_processor()
    
    print(f"📚 Starting literature grounding for hypothesis...")
    
    try:
        # Step 1: Vectorize hypothesis
        hypothesis_vector = await vectorize_hypothesis(hypothesis)
        if hypothesis_vector is None:
            return "", []
        
        faiss_db_path = 'Faiss/arxiv_chunks_faiss.index'
        meta_db_path = 'Faiss/arxiv_chunks_meta.pkl'
        
        # Step 2: Initial cosine similarity search
        similar_chunks = await cosine_similarity_search(
            hypothesis_vector, faiss_db_path, meta_db_path, top_k=30, min_similarity=min_similarity
        )
        
        # Check if we have enough chunks (minimum 3 with similarity > 0.7)
        if len(similar_chunks) < 3:
            print(f"⚠️ Only {len(similar_chunks)} chunks found with similarity > {min_similarity}")
            print(f"🌐 Searching ArXiv for additional sources...")
            
            # Extract keywords and search ArXiv
            keywords = await extract_keywords_from_hypothesis(hypothesis)
            success = await search_arxiv_and_add_to_faiss(
                keywords, hypothesis, faiss_db_path, meta_db_path, max_papers=4
            )
            
            if success:
                # Retry search after adding new papers
                similar_chunks = await cosine_similarity_search(
                    hypothesis_vector, faiss_db_path, meta_db_path, top_k=30, min_similarity=min_similarity
                )
                print(f"   🔄 After ArXiv search: {len(similar_chunks)} chunks found")
        else:
            print(f"🔍 Found {len(similar_chunks)} chunks with similarity > {min_similarity}")
        
        if not similar_chunks:
            print(f"❌ No relevant chunks found")
            return "", []
        
        # Step 3: LLM relevance validation with source limiting
        print(f"🤖 Validating {len(similar_chunks)} chunks with LLM...")
        
        # Group chunks by paper and sort by similarity
        chunks_by_paper = {}
        for chunk in similar_chunks:
            paper_title = chunk.get('paper_title', 'Unknown')
            if paper_title not in chunks_by_paper:
                chunks_by_paper[paper_title] = []
            chunks_by_paper[paper_title].append(chunk)
        
        # Sort papers by their best chunk similarity
        sorted_papers = sorted(
            chunks_by_paper.items(), 
            key=lambda x: max(c.get('cosine_similarity', 0) for c in x[1]), 
            reverse=True
        )
        
        validated_chunks = []
        sources_used = 0
        
        # Process validation tasks in batches to speed up
        validation_tasks = []
        for paper_title, paper_chunks in sorted_papers[:max_sources * 2]:  # Only validate top papers
            if sources_used >= max_sources:
                break
                
            # Sort chunks within paper by similarity
            paper_chunks.sort(key=lambda x: x.get('cosine_similarity', 0), reverse=True)
            best_chunk = paper_chunks[0]
            
            task = llm_relevance_validation(best_chunk, hypothesis)
            validation_tasks.append((paper_title, best_chunk, task))
        
        # Process validations in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent validations
        
        async def validate_with_semaphore(paper_title, chunk, task):
            async with semaphore:
                try:
                    is_relevant = await task
                    return paper_title, chunk, is_relevant
                except Exception as e:
                    # If validation fails, assume relevant to avoid losing good sources
                    return paper_title, chunk, True
        
        validation_results = await asyncio.gather(*[
            validate_with_semaphore(paper_title, chunk, task) 
            for paper_title, chunk, task in validation_tasks
        ])
        
        # Process results
        for paper_title, chunk, is_relevant in validation_results:
            if sources_used >= max_sources:
                break
                
            if is_relevant:
                validated_chunks.append(chunk)
                sources_used += 1
                print(f"  ✅ Added chunk from '{paper_title[:40]}...'")
            else:
                print(f"  ❌ Rejected '{paper_title[:40]}...'")
        
        print(f"✅ Final selection: {len(validated_chunks)} chunks from {sources_used} unique sources")
        
        if validated_chunks:
            # Sort final chunks by similarity and extract context
            validated_chunks.sort(key=lambda x: x.get('cosine_similarity', 0), reverse=True)
            context, links = await extract_context_and_links(validated_chunks)
            return context, links
        
        return "", []
        
    except Exception as e:
        print(f"❌ Literature grounding failed: {e}")
        return "", []

# Global cache for literature context to avoid redundant API calls
literature_cache = {}

async def get_literature_for_strategy_or_hypothesis(search_target):
    """Get literature context for strategy or hypothesis with intelligent caching"""
    cache_key = search_target[:100]  # Use first 100 chars as cache key
    
    if cache_key in literature_cache:
        print(f"📚 Using cached literature...")
        return literature_cache[cache_key]
    
    # Get literature context using the existing workflow
    context, links = await literature_grounding_workflow(search_target, max_sources=3, min_similarity=0.7)
    
    # Cache result
    literature_cache[cache_key] = (context, links)
    return context, links

async def get_literature_for_hypothesis(hypothesis):
    """Get literature context for hypothesis with caching to reduce API calls"""
    return await get_literature_for_strategy_or_hypothesis(hypothesis)

async def extract_research_components(user_input):
    """Extract research goal, hypotheses, and relevant information"""
    prompt = f"""
    Extract and structure the following from the research plan:
    - research_goal: Main research objective
    - hypotheses: List of testable hypotheses (as strings)
    - relevant_info: Supporting information, constraints, variables
    
    Return only JSON format.
    
    Research Plan: {user_input}
    """
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": "Extract research components. Return only valid JSON with hypotheses as string array."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        
        cleaned_content = clean_json_string(content)
        result = json.loads(cleaned_content)
        
        # Ensure all values are strings
        if "hypotheses" in result:
            hypotheses = result["hypotheses"]
            if isinstance(hypotheses, list):
                result["hypotheses"] = [str(h) for h in hypotheses]
            else:
                result["hypotheses"] = [str(hypotheses)]
        
        return result
    except Exception:
        return {"error": "Failed to parse", "hypotheses": [user_input]}

async def generate_nodes(parent_description, node_type, hypothesis, relevant_info, user_input, count=3):
    """Generate child nodes with dynamic literature context based on search target"""
    
    # Determine search target for literature context
    if node_type == "strategy":
        # For strategies, search based on hypothesis
        search_target = hypothesis
        print(f"  📚 Getting literature context for hypothesis...")
    else:
        # For implementations, search based on strategy (parent node)
        search_target = parent_description
        print(f"    📚 Getting literature context for strategy...")
    
    # Get dynamic literature context
    literature_context, source_links = await get_literature_for_strategy_or_hypothesis(search_target)
    
    if node_type == "strategy":
        prompt = f"""Generate {count} DISTINCT high-level experimental strategies to test: {hypothesis}

Requirements:
- 1-2 sentences each, concise and clear
- Fundamentally different approaches (controlled experiments, observational studies, simulations, surveys, meta-analysis)
- Focus on overall approach, not implementation details
- MUST consider constraints and opportunities from the original research context below
- Tailor strategies to match the apparent research setting, resources, and objectives

Original Research Context: {user_input}
Additional Context/Constraints: {relevant_info}

IMPORTANT: Design strategies that are realistic and appropriate for the research context described above.

Format your response as plain text with each strategy separated by "---":

Strategy 1 description here

---

Strategy 2 description here

---

Strategy 3 description here"""

    else:  # implementation
        prompt = f"""Generate {count} COMPLETELY DIFFERENT experimental designs that all follow this general strategy approach: 

STRATEGY APPROACH: {parent_description}

Each experimental design should be a STANDALONE, COMPLETE experiment that uses the same general strategy type but with entirely different approaches.

Requirements for each experimental design:
- Each must be a COMPLETE, INDEPENDENT experimental study 
- Focus PRIMARY on detailed step-by-step experimental procedures
- Keep background information concise
- Include specific methodologies, tools, datasets, and evaluation approaches
- MUST consider practical constraints and available resources from the original research context
- Tailor complexity and resource requirements to match the research setting

Structure each experimental design as follows:

**Experimental Design Title:** [Clear, descriptive title]

**Brief Background:** [1-2 sentences on objective and hypothesis]

**Step-by-Step Experimental Protocol:**
Step 1: [Detailed description]
Step 2: [Detailed description]
Step 3: [Detailed description]
[Continue with as many steps as needed]

**Key Tools:** [Brief list of essential tools/software]
**Success Criteria:** [Brief measurable outcomes]

Original Research Context: {user_input}
Additional Context/Constraints: {relevant_info}
Hypothesis: {hypothesis}

IMPORTANT: Design experiments that are feasible given the research context and constraints described above.

Format your response as plain text with each experimental design separated by "---":

Experimental Design 1: [Title]
[Complete experimental design with focus on steps]

---

Experimental Design 2: [Title]  
[Complete experimental design with focus on steps]

---

Experimental Design 3: [Title]
[Complete experimental design with focus on steps]"""
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": f"Generate {count} COMPLETELY INDEPENDENT experimental {node_type}. For implementations, each should be a STANDALONE experimental study that uses the same strategy category but with entirely different approaches, data sources, tools, and methodologies. Think of each as a separate research paper. Return plain text format with sections separated by '---'. "},
            {"role": "user", "content": prompt}
        ], temperature=0.8)
        
        # Parse plain text sections separated by "---"
        sections = content.split("---")
        
        # Clean and validate each section
        valid_nodes = []
        for section in sections:
            section = section.strip()
            
            # Enhanced validation to filter out JSON fragments and short content
            if len(section) < 100:  # Too short to be a complete experimental design
                continue
                        
            # Add source links at the end only if literature context was used and chunk info appears in content
            if literature_context and source_links and node_type == "implementation":
                # Check if any chunk information is actually referenced in the generated content
                chunk_used = False
                
                # Extract key terms from literature chunks to check usage
                for chunk in literature_context.split('\n\n'):
                    if chunk.strip():
                        # Extract key technical terms and concepts (longer than 4 chars, not common words)
                        chunk_words = chunk.lower().split()
                        key_terms = [word for word in chunk_words 
                                   if len(word) > 4 and 
                                   word not in ['methods', 'results', 'study', 'research', 'analysis', 'approach', 'using', 'based', 'findings', 'showed', 'demonstrate', 'conducted', 'performed', 'investigated']]
                        
                        # Check if any key terms appear in the generated content
                        section_lower = section.lower()
                        for term in key_terms[:10]:  # Check top 10 terms to avoid false positives
                            if term in section_lower:
                                chunk_used = True
                                break
                        
                        if chunk_used:
                            break
                
                # Only add source links if chunk information was actually used
                if chunk_used:
                    source_text = "\n\n**Literature References:** " + " | ".join(source_links)
                    section += source_text
                
            # Create node
            node = ExperimentNode(section, depth=(1 if node_type == "implementation" else 0), node_type=node_type)
            node.source_links = source_links  # Keep source links for reference
            valid_nodes.append(node)
        
        if node_type == "strategy":
            print(f"  📝 Generated {len(valid_nodes)} strategy nodes")
        else:
            print(f"    📝 Generated {len(valid_nodes)} implementation nodes")
        return valid_nodes[:count] if valid_nodes else []
        
    except Exception as e:
        logger.warning(f"Failed to generate {node_type} nodes: {e}")
        return []

async def score_node(description, hypothesis, depth, user_input, relevant_info):
    """Score node for feasibility and impact, with special consideration for implementation detail and user context"""
    depth_context = {0: "experimental strategy", 1: "implementation plan"}
    
    # Enhanced prompts that consider user context and constraints
    if depth == 1:  # Implementation node
        prompt = f"""
        Score this implementation plan for testing: "{hypothesis}"
        
        Original Research Context: {user_input[:300]}
        Additional Constraints/Info: {relevant_info[:200]}
        
        Implementation: {description[:600]}  
        
        Rate 0-10 considering the specific research context:
        1. FEASIBILITY: How practical is this given the user's apparent resources, timeline, and constraints?
        2. IMPACT: How well does this address the hypothesis and align with the research goals?
        
        Consider factors like:
        - Available resources/equipment mentioned in context
        - Timeline constraints
        - Technical complexity vs. apparent expertise level
        - Alignment with stated research objectives
        
        Return ONLY: {{"feasibility": X, "impact": Y}}
        """
    else:  # Strategy node
        prompt = f"""
        Score this strategy for testing: "{hypothesis}"
        
        Original Research Context: {user_input[:300]}
        Additional Constraints/Info: {relevant_info[:200]}
        
        Strategy: {description[:500]}
        
        Rate 0-10 considering the specific research context:
        1. FEASIBILITY: How realistic is this approach given the user's context and constraints?
        2. IMPACT: How effectively would this strategy test the hypothesis and meet research goals?
        
        Consider factors like:
        - Research setting (academic, industry, etc.)
        - Available resources and timeline
        - Methodological rigor vs. practical constraints
        - Alignment with stated objectives
        
        Return ONLY: {{"feasibility": X, "impact": Y}}
        """
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": "Provide objective scores. Return ONLY valid JSON with integer scores 0-10."},
            {"role": "user", "content": prompt}
        ], temperature=0.1, max_tokens=50)  # Limit tokens for faster response
        
        cleaned_content = clean_json_string(content)
        scores = json.loads(cleaned_content)
        
        feasibility = max(0, min(10, int(scores.get("feasibility", 5))))
        impact = max(0, min(10, int(scores.get("impact", 5))))
        
        return feasibility, impact
        
    except Exception:
        return 6, 6  # Slightly higher fallback scores

def apply_adaptive_threshold(nodes, base_threshold=0.6, max_nodes=None):
    """Apply AB-MCTS-A adaptive thresholding (Sakana-style)"""
    if not nodes:
        return []
    
    scores = [node.combined_score for node in nodes]
    mean_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    if len(scores) > 1:
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        adaptive_threshold = max(
            base_threshold * max_score,
            mean_score + 0.5 * std_dev
        )
    else:
        adaptive_threshold = base_threshold * max_score
    
    promising_nodes = [node for node in nodes if node.combined_score >= adaptive_threshold]
    
    # Ensure at least top 2 nodes if any exist
    if not promising_nodes and nodes:
        sorted_nodes = sorted(nodes, key=lambda x: x.combined_score, reverse=True)
        promising_nodes = sorted_nodes[:2]
    
    # Apply maximum limit if specified
    if max_nodes and len(promising_nodes) > max_nodes:
        sorted_nodes = sorted(promising_nodes, key=lambda x: x.combined_score, reverse=True)
        promising_nodes = sorted_nodes[:max_nodes]
    
    return promising_nodes

async def build_experiment_tree(hypothesis, relevant_info, user_input, hypothesis_id):
    """Build two-level experiment tree with parallel processing and progress tracking"""
    hypothesis_str = str(hypothesis)
    
    # Add progress tracking for this hypothesis
    await progress_tracker.add_task(f"hyp_{hypothesis_id}", f"Hypothesis {hypothesis_id}: {hypothesis_str[:40]}...")
    
    try:
        # Level 1: Generate strategies in parallel with literature context
        strategy_task_id = f"hyp_{hypothesis_id}_strategies"
        await progress_tracker.add_task(strategy_task_id, f"  Generating strategies for hypothesis {hypothesis_id}")
        
        strategy_nodes = await generate_nodes("", "strategy", hypothesis_str, relevant_info, user_input, count=3)
        if not strategy_nodes:
            await progress_tracker.complete_task(strategy_task_id)
            await progress_tracker.complete_task(f"hyp_{hypothesis_id}")
            return []
        
        await progress_tracker.complete_task(strategy_task_id)
        
        # Score all strategies in parallel
        scoring_task_id = f"hyp_{hypothesis_id}_scoring"
        await progress_tracker.add_task(scoring_task_id, f"  Scoring strategies for hypothesis {hypothesis_id}")
        
        semaphore = asyncio.Semaphore(3)  # Increased concurrency
        
        async def score_with_semaphore(node):
            async with semaphore:
                feasibility, impact = await score_node(node.description, hypothesis_str, 0, user_input, relevant_info)
                node.feasibility_score = feasibility
                node.impact_score = impact
                node.calculate_score()
                return node
        
        scored_strategies = await asyncio.gather(*[score_with_semaphore(node) for node in strategy_nodes])
        await progress_tracker.complete_task(scoring_task_id)
        
        # Apply adaptive threshold but limit to maximum 3 strategies
        promising_strategies = apply_adaptive_threshold(scored_strategies, base_threshold=0.5, max_nodes=3)
        
        # Level 2: Generate implementations in parallel for all strategies
        impl_tasks = []
        total_budget = 6  # Maximum total implementations across all strategies
        remaining_budget = total_budget
        
        for i, strategy_node in enumerate(promising_strategies):
            strategies_remaining = len(promising_strategies) - i
            
            if strategies_remaining == 1:
                child_count = min(remaining_budget, 6)
            else:
                base_allocation = remaining_budget // strategies_remaining
                bonus = 1 if strategy_node.combined_score > 7.0 else 0
                child_count = min(base_allocation + bonus, 3)
            
            if child_count > 0:
                impl_task_id = f"hyp_{hypothesis_id}_impl_{i}"
                await progress_tracker.add_task(impl_task_id, f"  Generating implementations for strategy {i+1}")
                
                task = generate_and_score_implementations_parallel(
                    strategy_node, hypothesis_str, relevant_info, user_input, child_count, impl_task_id
                )
                impl_tasks.append(task)
                remaining_budget -= child_count
        
        # Execute all implementation tasks in parallel
        if impl_tasks:
            await asyncio.gather(*impl_tasks)
        
        await progress_tracker.complete_task(f"hyp_{hypothesis_id}")
        return promising_strategies
        
    except Exception as e:
        logger.error(f"Failed to build tree for hypothesis {hypothesis_id}: {e}")
        await progress_tracker.complete_task(f"hyp_{hypothesis_id}")
        return []

async def generate_and_score_implementations(strategy_node, hypothesis_str, relevant_info, user_input, child_count=2):
    """Generate and score implementation nodes for a strategy with dynamic child count"""
    print(f"    ⚙️ Generating {child_count} implementations...")
    
    impl_nodes = await generate_nodes(
        strategy_node.description, "implementation", hypothesis_str, relevant_info, user_input, count=child_count
    )
    
    if not impl_nodes:
        return
    
    # Score implementations in parallel with controlled concurrency
    semaphore = asyncio.Semaphore(2)  # Limit concurrent scoring
    
    async def score_impl_with_semaphore(impl_node):
        async with semaphore:
            try:
                feasibility, impact = await score_node(impl_node.description, hypothesis_str, 1, user_input, relevant_info)
                impl_node.feasibility_score = feasibility
                impl_node.impact_score = impact
                impl_node.calculate_score()
                return impl_node
            except Exception as e:
                logger.error(f"Failed to score implementation: {e}")
                return None
    
    # Score all implementations
    scored_impls = await asyncio.gather(*[score_impl_with_semaphore(node) for node in impl_nodes])
    
    # Filter out failed scorings and add to strategy
    for impl_node in scored_impls:
        if impl_node is not None:
            strategy_node.children.append(impl_node)
    
    # Apply adaptive threshold with max limit
    promising_impls = apply_adaptive_threshold(strategy_node.children, base_threshold=0.5, max_nodes=child_count)
    strategy_node.children = promising_impls
    print(f"    📊 Selected {len(promising_impls)}/{len(impl_nodes)} implementations")

async def generate_and_score_implementations_parallel(strategy_node, hypothesis_str, relevant_info, user_input, child_count, task_id):
    """Parallel version of generate_and_score_implementations with progress tracking"""
    try:
        impl_nodes = await generate_nodes(
            strategy_node.description, "implementation", hypothesis_str, relevant_info, user_input, count=child_count
        )
        
        if not impl_nodes:
            await progress_tracker.complete_task(task_id)
            return
        
        # Score implementations in parallel with increased concurrency
        semaphore = asyncio.Semaphore(4)  # Higher concurrency for faster processing
        
        async def score_impl_with_semaphore(impl_node):
            async with semaphore:
                try:
                    feasibility, impact = await score_node(impl_node.description, hypothesis_str, 1, user_input, relevant_info)
                    impl_node.feasibility_score = feasibility
                    impl_node.impact_score = impact
                    impl_node.calculate_score()
                    return impl_node
                except Exception as e:
                    logger.error(f"Failed to score implementation: {e}")
                    return None
        
        # Score all implementations in parallel
        scored_impls = await asyncio.gather(*[score_impl_with_semaphore(node) for node in impl_nodes])
        
        # Filter out failed scorings and add to strategy
        for impl_node in scored_impls:
            if impl_node is not None:
                strategy_node.children.append(impl_node)
        
        # Apply adaptive threshold with max limit
        promising_impls = apply_adaptive_threshold(strategy_node.children, base_threshold=0.5, max_nodes=child_count)
        strategy_node.children = promising_impls
        
        await progress_tracker.complete_task(task_id)
        
    except Exception as e:
        logger.error(f"Failed to generate implementations: {e}")
        await progress_tracker.complete_task(task_id)

def serialize_tree(nodes):
    """Convert tree to serializable format including literature context"""
    def serialize_node(node):
        return {
            "description": node.description,
            "type": node.node_type,
            "depth": node.depth,
            "feasibility_score": node.feasibility_score,
            "impact_score": node.impact_score,
            "combined_score": node.combined_score,
            "literature_context": getattr(node, 'literature_context', ''),
            "source_links": getattr(node, 'source_links', []),
            "children": [serialize_node(child) for child in node.children]
        }
    return [serialize_node(node) for node in nodes]

def create_tree_markdown(hypothesis_results, processing_time):
    """Create comprehensive markdown output with literature grounding"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_tree_{timestamp}.md"
    
    md_content = f"""# Two-Level Experiment Design Tree with Literature Grounding
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Processing Time: {processing_time:.2f} seconds*
*System: AB-MCTS-A Two-Level Tree Expansion with Literature Context*

## Overview
Complete experimental design tree with two levels and literature grounding:
- **Level 1 (Strategies)**: High-level experimental approaches
- **Level 2 (Implementations)**: Complete step-by-step execution plans with literature context
- **Literature Integration**: Cosine similarity search (>0.7) + LLM validation + ArXiv fallback

**Scoring:** Feasibility (F) and Impact (I) from 0-10, Combined = 0.6×F + 0.4×I

---

"""
    
    for hyp_idx, hyp_result in enumerate(hypothesis_results, 1):
        md_content += f"""## Hypothesis {hyp_idx}
**{hyp_result['hypothesis']}**

"""
        
        tree_structure = hyp_result.get('tree_structure', [])
        
        for strategy_idx, strategy_node in enumerate(tree_structure, 1):
            md_content += f"""### {strategy_idx}. Strategy (Level 1)
**Description:** {strategy_node['description']}
**Scores:** F={strategy_node['feasibility_score']}, I={strategy_node['impact_score']}, Combined={strategy_node['combined_score']:.2f}

"""
            
            for impl_idx, impl_node in enumerate(strategy_node.get('children', []), 1):
                impl_description = impl_node['description']
                
                md_content += f"""#### {strategy_idx}.{impl_idx}. Implementation (Level 2)
**Description:** {impl_description}
**Scores:** F={impl_node['feasibility_score']}, I={impl_node['impact_score']}, Combined={impl_node['combined_score']:.2f}

"""
                
                # Add literature context if available
                if impl_node.get('literature_context'):
                    md_content += f"""**Literature Context:** {impl_node['literature_context'][:300]}...

"""
                
                # Add source links if available (these should already be in the description)
                if impl_node.get('source_links') and not any("Literature References:" in impl_description for impl_description in [impl_description]):
                    md_content += f"""**Literature References:** {' | '.join(impl_node['source_links'])}

"""
        
        md_content += "---\n\n"
    
    # Summary statistics
    total_strategies = sum(len(hyp_result.get('tree_structure', [])) for hyp_result in hypothesis_results)
    total_implementations = sum(
        len(strategy.get('children', [])) 
        for hyp_result in hypothesis_results 
        for strategy in hyp_result.get('tree_structure', [])
    )
    
    # Count literature-grounded implementations
    literature_grounded = sum(
        1 for hyp_result in hypothesis_results 
        for strategy in hyp_result.get('tree_structure', [])
        for impl in strategy.get('children', [])
        if impl.get('literature_context')
    )
    
    md_content += f"""## Summary Statistics
- **Total Hypotheses:** {len(hypothesis_results)}
- **Total Strategies (Level 1):** {total_strategies}
- **Total Implementations (Level 2):** {total_implementations}
- **Literature-Grounded Implementations:** {literature_grounded}
- **Processing Time:** {processing_time:.2f} seconds

## Literature Grounding Workflow
1. **Hypothesis Vectorization:** Used sentence-transformers for semantic embeddings
2. **Similarity Search:** Cosine similarity > 0.3 threshold in FAISS database
3. **LLM Validation:** Each chunk validated for relevance to hypothesis
4. **ArXiv Fallback:** If <3 validated chunks, search ArXiv and add to database
5. **Context Integration:** Literature context included in experimental designs

## Supervision Notes
1. **Strategy Diversity:** Review Level 1 nodes for comprehensive coverage
2. **Implementation Completeness:** Verify Level 2 nodes contain detailed execution steps
3. **Literature Integration:** Check that literature context enhances experimental design
4. **Score Validation:** Assess if scores align with domain expertise
5. **Source Verification:** Validate literature sources and links

---
*Generated by Enhanced Two-Level Tree-Based Experiment Design System with Literature Grounding*
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return filename

async def process_single_hypothesis(hypothesis, relevant_info, user_input, hypothesis_id):
    """Process a single hypothesis and return results with progress tracking"""
    try:
        tree_nodes = await build_experiment_tree(hypothesis, relevant_info, user_input, hypothesis_id)
        
        return {
            "hypothesis": hypothesis,
            "hypothesis_id": hypothesis_id,
            "tree_structure": serialize_tree(tree_nodes)
        }
        
    except Exception as e:
        logger.error(f"Error processing hypothesis {hypothesis_id} '{hypothesis}': {e}")
        return {
            "hypothesis": hypothesis,
            "hypothesis_id": hypothesis_id,
            "tree_structure": [],
            "error": str(e)
        }

async def main():
    """Main workflow entry point with parallel processing and organized output"""
    print("🧪 Enhanced Two-Level Tree-Based Experiment Design System with Literature Grounding")
    print("=" * 80)    
    user_input = input("\n📝 Enter your research plan: ").strip()
    if not user_input:
        print("❌ No input provided. Exiting.")
        return
    
    start_time = datetime.now()
    
    # Initialize ArXiv processor in background
    print("🔧 Initializing literature processing system...")
    global arxiv_processor
    arxiv_processor = initialize_arxiv_processor()
    
    # Step 1: Extract research components
    print("📋 Extracting research components...")
    await progress_tracker.add_task("extract_components", "Extracting research components")
    
    components = await extract_research_components(user_input)
    await progress_tracker.complete_task("extract_components")
    
    if "error" in components:
        hypotheses = [user_input]
        relevant_info = ""
        research_goal = "Research goal extraction failed"
    else:
        hypotheses = components.get("hypotheses", [user_input])
        relevant_info = str(components.get("relevant_info", ""))
        research_goal = str(components.get("research_goal", "No specific goal identified"))
    
    # Display extracted components
    print(f"\n📊 RESEARCH COMPONENTS EXTRACTED")
    print(f"  ✅ Research Goal: {research_goal}")
    print(f"  ✅ Found {len(hypotheses)} hypothesis(es)")
    print(f"  ✅ Context: {relevant_info}")
    
    # Step 2: Start parallel processing with progress display
    print(f"\n🚀 Starting parallel processing of {len(hypotheses)} hypotheses...")

    # Start progress display task
    progress_task = asyncio.create_task(display_progress())
    
    # Process all hypotheses in parallel
    hypothesis_tasks = []
    for i, hypothesis in enumerate(hypotheses, 1):
        task = process_single_hypothesis(hypothesis, relevant_info, user_input, i)
        hypothesis_tasks.append(task)
    
    # Execute all hypothesis processing in parallel
    hypothesis_results = await asyncio.gather(*hypothesis_tasks)
    
    # Stop progress display
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Step 3: Generate markdown output
    print(f"\n📝 Creating comprehensive markdown output...")
    markdown_file = create_tree_markdown(hypothesis_results, processing_time)
    
    # Step 4: Display organized results
    print(f"\n" + "=" * 80)
    print("🎯 LITERATURE-GROUNDED EXPERIMENT DESIGN SUMMARY")
    print("=" * 80)
    
    total_strategies = sum(len(result.get('tree_structure', [])) for result in hypothesis_results)
    total_implementations = sum(
        len(strategy.get('children', [])) 
        for result in hypothesis_results 
        for strategy in result.get('tree_structure', [])
    )
    
    # Count literature-grounded implementations
    literature_grounded = sum(
        1 for result in hypothesis_results 
        for strategy in result.get('tree_structure', [])
        for impl in strategy.get('children', [])
        if impl.get('literature_context')
    )
    
    for i, result in enumerate(hypothesis_results, 1):
        print(f"\n📋 Hypothesis {i}: {result['hypothesis']}")
        tree_structure = result.get('tree_structure', [])
        if tree_structure:
            strategies = len(tree_structure)
            implementations = sum(len(s.get('children', [])) for s in tree_structure)
            grounded_impls = sum(
                1 for s in tree_structure 
                for impl in s.get('children', [])
                if impl.get('literature_context')
            )
            print(f"  🌳 Generated {strategies} strategies → {implementations} implementations")
        else:
            print(f"  ❌ No experiments generated")
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  • Total Hypotheses: {len(hypothesis_results)}")
    print(f"  • Total Strategies (Level 1): {total_strategies}")
    print(f"  • Total Implementations (Level 2): {total_implementations}")
    print(f"  • Processing Time: {processing_time:.2f} seconds")
    print(f"  • Total API Calls: {api_call_count}")
    print(f"  • Estimated API Cost: ${total_api_cost:.4f}")
    
    print(f"\n✅ WORKFLOW COMPLETE")
    print(f"📄 Complete literature-grounded tree structure saved to: {markdown_file}")

# --- Main workflow ---
if __name__ == "__main__":
    asyncio.run(main())
