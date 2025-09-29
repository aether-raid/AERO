"""
ArXiv Paper Processing Utilities

This module contains utilities for downloading, extracting, and ranking arXiv papers
for relevance to research queries. Separated from the main MLResearcherLangGraph
class for better code organization.
"""

# import os
# # Disable TensorFlow oneDNN optimization messages and other warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)  # Suppress BeautifulSoup warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)  # Suppress TensorFlow deprecation warnings

import re
import math
import asyncio
from typing import Dict, List, Any, Optional
import requests
import feedparser
from .report_to_txt import extract_pdf_text


class ArxivPaperProcessor:

    def get_top_n_chunks(self, query: str, n: int = 5, faiss_db_path: str = 'Faiss/arxiv_chunks_faiss.index', meta_db_path: str = 'Faiss/arxiv_chunks_meta.pkl', embedding_dim: int = 768) -> list:
        """
        Retrieve the top n most relevant chunks for a query from the FAISS DB.
        Returns a list of chunk metadata dicts (with similarity scores).
        """
        import faiss
        import numpy as np
        import pickle
        import os
        
        try:
            # Load FAISS DB and metadata
            if not (os.path.exists(faiss_db_path) and os.path.exists(meta_db_path)):
                print(f"No FAISS DB or metadata found at {faiss_db_path} or {meta_db_path}")
                return []
            
            print(f"Loading FAISS index from {faiss_db_path}")
            index = faiss.read_index(faiss_db_path)
            print(f"FAISS index loaded with {index.ntotal} vectors, dimension: {index.d}")
            
            print(f"Loading metadata from {meta_db_path}")
            with open(meta_db_path, 'rb') as f:
                meta = pickle.load(f)
            print(f"Metadata loaded for {len(meta)} papers")
            
            # Flatten all chunk metadata
            all_chunks = []
            for paper_id, chunk_list in meta.items():
                all_chunks.extend(chunk_list)
            print(f"Total chunks available: {len(all_chunks)}")
            
            if len(all_chunks) == 0:
                print("No chunks found in metadata")
                return []
            
            # Use the actual FAISS index dimension, not the parameter
            actual_faiss_dim = index.d
            print(f"FAISS index dimension: {actual_faiss_dim}, requested embedding_dim: {embedding_dim}")
            
            # Embed the query
            print(f"Embedding query: '{query[:50]}...'")
            if hasattr(self, 'create_embedding') and callable(self.create_embedding):
                query_vec = self.create_embedding(query)
                if query_vec is None:
                    raise Exception("create_embedding returned None - embedding model not ready")
                print(f"Query embedded successfully, shape: {np.array(query_vec).shape}")
            else:
                print(f"Warning: Using random vector for query embedding with dimension {actual_faiss_dim}")
                query_vec = np.random.randn(actual_faiss_dim).astype('float32')
            
            # Ensure query vector is the right shape and type
            query_vec = np.array(query_vec, dtype='float32')
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            
            print(f"Query vector shape: {query_vec.shape}, FAISS expects: (1, {actual_faiss_dim})")
            
            # Check dimension compatibility with actual FAISS index
            if query_vec.shape[1] != actual_faiss_dim:
                print(f"Adjusting query vector dimension from {query_vec.shape[1]} to {actual_faiss_dim}")
                if query_vec.shape[1] > actual_faiss_dim:
                    query_vec = query_vec[:, :actual_faiss_dim]
                else:
                    # Pad with zeros
                    padded = np.zeros((1, actual_faiss_dim), dtype='float32')
                    padded[:, :query_vec.shape[1]] = query_vec
                    query_vec = padded
                print(f"Adjusted query vector shape: {query_vec.shape}")
            
            # Search FAISS
            print(f"Searching FAISS for top {n} chunks...")
            D, I = index.search(query_vec, min(n, index.ntotal))
            print(f"Search completed, found {len(I[0])} results")
            
            # Map indices to chunk metadata
            top_chunks = []
            for idx, dist in zip(I[0], D[0]):
                if 0 <= idx < len(all_chunks):
                    chunk = dict(all_chunks[idx])
                    chunk['faiss_index'] = int(idx)
                    chunk['distance'] = float(dist)
                    top_chunks.append(chunk)
                else:
                    print(f"Warning: Index {idx} out of range for {len(all_chunks)} chunks")
            
            print(f"Successfully mapped {len(top_chunks)} chunks")
            return top_chunks
            
        except Exception as e:
            print(f"Error in get_top_n_chunks: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    """Utility class for processing arXiv papers."""
    
    def __init__(self, llm_client, model_name: str):
        """Initialize with LLM client for relevance scoring. Embedding model loads asynchronously."""
        self.client = llm_client
        self.model = model_name
        # Async loading: start loading embedding model in background immediately
        self.embedding_model = None
        self._embedding_model_loading = False
        self._embedding_load_task = None
        self._start_async_model_loading()
    
    def _start_async_model_loading(self):
        """Start loading the embedding model asynchronously in the background."""
        import threading
        
        def load_model():
            if not self._embedding_model_loading:
                self._embedding_model_loading = True
                try:
                    # First check if torch is working properly with a simple tensor operation
                    try:
                        import torch
                        # Test torch functionality with a simple operation instead of version check
                        test_tensor = torch.tensor([1.0, 2.0])
                        _ = test_tensor.sum()  # Simple operation to verify torch works
                        # print("✅ PyTorch is working correctly")  # Suppressed to avoid cluttering output
                    except Exception as torch_err:
                        print(f"⚠️ PyTorch functionality issue detected: {torch_err}")
                        raise Exception(f"PyTorch not working properly: {torch_err}")
                    
                    # Try importing SentenceTransformers with better error handling
                    try:
                        from sentence_transformers import SentenceTransformer
                        # print("✅ SentenceTransformers imported successfully")  # Suppressed to avoid cluttering output
                    except ImportError as st_err:
                        if "cannot import name 'Tensor'" in str(st_err):
                            print("⚠️ Known PyTorch-SentenceTransformers compatibility issue detected")
                            print("💡 Try: pip install --upgrade torch sentence-transformers")
                        raise st_err
                    
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    # print("✅ Embedding model loaded and ready!")  # Suppressed to avoid cluttering output
                except Exception as e:
                    print(f"❌ Background embedding model loading failed: {e}")
                    print("🔧 Embedding features will be disabled. The system will work without semantic search.")
                    self.embedding_model = None
                finally:
                    self._embedding_model_loading = False
        
        # Start loading in background thread
        self._embedding_load_task = threading.Thread(target=load_model, daemon=True)
        self._embedding_load_task.start()
    
    def _get_embedding_model(self):
        """Get the embedding model, waiting for background loading if needed."""
        if self.embedding_model is None and self._embedding_model_loading:
            print("⏳ Waiting for embedding model to finish loading...")
            if self._embedding_load_task and self._embedding_load_task.is_alive():
                self._embedding_load_task.join()  # Wait for background loading to complete
        
        if self.embedding_model is None and not self._embedding_model_loading:
            # Fallback: try loading synchronously if background loading failed
            try:
                # First check if torch is working properly with a simple tensor operation
                try:
                    import torch
                    # Test torch functionality with a simple operation instead of version check
                    test_tensor = torch.tensor([1.0, 2.0])
                    _ = test_tensor.sum()  # Simple operation to verify torch works
                    # print("✅ PyTorch is working correctly")  # Suppressed to avoid cluttering output
                except Exception as torch_err:
                    print(f"⚠️ PyTorch functionality issue detected: {torch_err}")
                    raise Exception(f"PyTorch not working properly: {torch_err}")
                
                # Try importing SentenceTransformers with better error handling
                try:
                    from sentence_transformers import SentenceTransformer
                    # print("✅ SentenceTransformers imported successfully")  # Suppressed to avoid cluttering output
                except ImportError as st_err:
                    if "cannot import name 'Tensor'" in str(st_err):
                        print("⚠️ Known PyTorch-SentenceTransformers compatibility issue detected")
                        print("💡 Try: pip install --upgrade torch sentence-transformers")
                    raise st_err
                
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # print("✅ Embedding model loaded successfully!")  # Suppressed to avoid cluttering output
            except Exception as e:
                print(f"❌ Embedding model failed to load: {e}")
                print("🔧 Embedding features will be disabled. The system will work without semantic search.")
                self.embedding_model = None
        
        return self.embedding_model
    
    def create_embedding(self, text):
        """Create embedding for text using sentence-transformers."""
        # Validate input text
        if not isinstance(text, str):
            print(f"⚠️ Warning: Expected string input, got {type(text)}: {repr(text)}")
            if text is None:
                text = ""
            else:
                text = str(text)
        
        # Ensure text is not empty
        if not text.strip():
            text = "empty text"
            
        model = self._get_embedding_model()
        if model is not None:
            try:
                return model.encode(text, show_progress_bar=False, normalize_embeddings=True)
            except Exception as e:
                print(f"❌ Error creating embedding for text: {e}")
                print(f"Text type: {type(text)}, Text preview: {repr(text[:100])}")
                # Return a random embedding as fallback
                import numpy as np
                return np.random.randn(384).astype('float32')
        return None
    
    def extract_basic_paper_info(self, entry, ns, index):
        """Extract basic paper info without downloading PDF content."""
        try:
            # Extract basic info
            title = entry.find('atom:title', ns).text.strip()
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            
            # Get published date
            published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown"
            
            # Get abstract/summary
            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Get arXiv URL
            arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            # Store paper info without content
            paper_info = {
                "title": title,
                "id": paper_id,
                "published": published,
                "summary": summary,
                "content": None,  # Will be filled later for top papers
                "url": arxiv_url,
                "index": index,
                "pdf_downloaded": False
            }
            
            return paper_info
            
        except Exception as e:
            print(f"❌ Error extracting basic info for paper #{index}: {e}")
            return {
                "title": f"Error processing paper #{index}",
                "id": "error",
                "published": "Unknown",
                "summary": "",
                "content": None,
                "url": "error",
                "index": index,
                "pdf_downloaded": False,
                "error": str(e)
            }
    
    def download_paper_content(self, paper_info):
        """Download and extract PDF content for a specific paper."""
        try:
            paper_id = paper_info['id']
            arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            response = requests.get(arxiv_url)
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                return paper_info
                
            entry_data = feed.entries[0]
            
            # Find PDF link
            pdf_link = None
            for link in entry_data.links:
                if link.type == 'application/pdf':
                    pdf_link = link.href
                    break
            
            # Extract text from PDF
            if pdf_link:
               # print(f"Fetching PDF from: {pdf_link}")
                # Pass paper metadata for proper file saving
                pdf_txt = extract_pdf_text(
                    pdf_link, 
                    paper_title=paper_info.get('title', 'Unknown Title'),
                    paper_id=paper_id,
                    save_files=True
                )
                paper_info['content'] = pdf_txt
                paper_info['pdf_downloaded'] = True
              #  print(f"✅ Downloaded PDF content for: {paper_info['title'][:50]}...")
            else:
               print(f"⚠️ No PDF link found for: {paper_info['title'][:50]}...")
            
            return paper_info
            
        except Exception as e:
            print(f"❌ Error downloading PDF for {paper_info['title'][:50]}...: {e}")
            return paper_info
    
    async def score_paper_relevance(self, paper_title: str, paper_content: str, original_query: str, custom_prompt: Optional[str] = None) -> float:
        """LLM relevance score in [1.0, 10.0]. Returns a float only."""
        # Keep prompts lean; truncate huge inputs to control tokens
        MAX_CHARS = 8000
        title = (paper_title or "").strip()[:512] or "<untitled>"
        content = (paper_content or "").strip()[:MAX_CHARS]
        #print(content)
        query = (original_query or "").strip()[:2000]

        # Use custom prompt if provided, otherwise use default prompt
        if custom_prompt:
            user_prompt = custom_prompt.format(
                query=query,
                title=title,
                content=content
            )
        else:
            # Default prompt (original logic)
            user_prompt = f"""
You are an expert ML librarian. Score how relevant the paper is to the user's research query on a 1–10 scale.

OUTPUT FORMAT (STRICT):
- Return EXACTLY one floating-point number with ONE decimal (e.g., 8.7).
- No words, no JSON, no units, no symbols, no explanation.
- Single line only (no leading/trailing spaces or extra lines).

Use ONLY the text provided below (title + content). Do not browse or assume unstated results. If the content is partial, rely on what's given (title/abstract first).

Scoring rubric — assign four subscores in [0,1]:
- task_match (40%): directly addresses the research task(s).
- method_match (30%): overlap with the architectures/approaches in the query or close variants.
- constraint_match (20%): aligns with constraints/tooling/datasets/hardware (e.g., real-time, FPS/latency, edge/mobile, TensorRT/ONNX, INT8/FP16).
- evidence_match (10%): concrete signals (benchmarks like nuScenes/Waymo/KITTI, metrics, ablations, deployment notes).

Compute:
- Let t,m,c,e ∈ [0,1].
- score = round((0.40*t + 0.30*m + 0.20*c + 0.10*e) * 10, 1).
- If clearly unrelated (all four < 0.15), output 1.0.
- Clip to [1.0, 10.0].
- Be conservative if uncertain.

Research query:
\"\"\"{query}\"\"\"

Paper title:
\"\"\"{title}\"\"\"

Paper content:
\"\"\"{content}\"\"\"
""".strip()

        async def _call_llm(prompt: str) -> str:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a strict numeric scorer. Reply with ONLY a number between 1.0 and 10.0."},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
            return (resp.choices[0].message.content or "").strip()

        def _to_score(txt: str) -> float:
            # Pull the first numeric token; tolerate minor deviations
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
            if not m:
                return 1.0
            val = float(m.group())
            # clip to [1.0, 10.0]
            if not math.isfinite(val):
                return 1.0
            return max(1.0, min(10.0, val))

        # Retries with backoff for transient failures
        backoff = 0.6
        for attempt in range(3):
            try:
                raw = await _call_llm(user_prompt)
                score = _to_score(raw)
                return score
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    return 1.0
        return 1.0  # Fallback (should not reach here)
    
    async def rank_papers_by_relevance(self, papers: list[dict], original_query: str, custom_prompt: Optional[str] = None) -> list[dict]:
        """Score and rank papers by relevance using cosine similarity (fast, deterministic)."""
        #print("\n🎯 Scoring papers for relevance using cosine similarity...")
        
        # Create scoring tasks for all papers using the new cosine similarity method
        async def score_paper(i, paper):
            #print(f"⏳ Scoring paper {i}/{len(papers)}: {paper['title'][:50]}...")
            
            # Use the new cosine similarity scoring method with optional custom prompt
            relevance_score = await self.score_paper_relevance(
                paper['title'], 
                paper.get('summary', ''),  # Use summary instead of content for initial ranking
                original_query,
                custom_prompt  # Pass through the custom prompt
            )
            
            paper['relevance_score'] = relevance_score
            #print(f"Relevance Score for paper {i:03d}: {relevance_score:.1f}/10.0")
            return paper

        
        # Run all scoring tasks concurrently
        scoring_tasks = [score_paper(i, paper) for i, paper in enumerate(papers, 1)]
        scored_papers = await asyncio.gather(*scoring_tasks)
        
        # Sort by relevance score (highest first) and return top 5
        ranked_papers = sorted(scored_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        #print(f"\n Papers ranked by cosine similarity to: '{original_query}'")
        return ranked_papers[:5]  # Return only top 5

    async def chunk_and_embed(self, paper: dict, faiss_db=None, embedding_dim: int = 768) -> list[dict]:
        """
        Chunk the paper content by sections, then into 500-token chunks with 150-token overlap (within sections, do not split sections),
        embed each chunk, and save to a FAISS database. If faiss_db is not provided, create one. Returns a list of chunk dicts.
        """
        import faiss
        import numpy as np
        print(f"🔍 Chunking and embedding paper: {paper['title'][:50]}...")
        content = paper.get('content', '')
        if not content:
            print("⚠️ No content to chunk and embed.")
            return []

        # Helper: Split content into sections (simple heuristic: look for lines with ALL CAPS or numbered headings)
        def split_into_sections(text):
            import re
            # First, try to split by page markers (--- PAGE X ---)
            page_pattern = r'--- PAGE \d+ ---'
        
            pages = re.split(page_pattern, text)
            
            # If we found page markers, use those as sections
            if len(pages) > 1:
                sections = []
                for i, page_content in enumerate(pages):
                    if page_content.strip():  # Skip empty pages
                        # Clean up the page content
                        cleaned_content = page_content.strip()
                        if cleaned_content:
                            sections.append(cleaned_content)
                print(f"Split into {len(sections)} sections using page markers")
                return sections
            
            # Fallback: Original section splitting logic for papers without page markers
            lines = text.split('\n')
            sections = []
            current_section = []
            for line in lines:
                # Look for section headers (numbered sections, all caps headers, etc.)
                if re.match(r'^(\d+\.|[A-Z][A-Z\s\-:]{5,}|Abstract|Introduction|Conclusion|References|Methodology|Results|Discussion)$', line.strip()):
                    if current_section:
                        sections.append('\n'.join(current_section).strip())
                        current_section = []
                current_section.append(line)
            if current_section:
                sections.append('\n'.join(current_section).strip())
            
            # Filter out very short sections (less than 100 characters)
            sections = [s for s in sections if len(s.strip()) > 100]
            print(f"Split into {len(sections)} sections using header detection")
            return sections

        # Helper: Tokenize (use simple whitespace split, or replace with tokenizer if available)
        def tokenize(text):
            # If you have a tokenizer, use it here. Otherwise, fallback to whitespace split.
            return text.split()

        # Helper: Detokenize
        def detokenize(tokens):
            return ' '.join(tokens)

        # Helper: Chunk a section into 500-token chunks with 150-token overlap
        def chunk_section(section, chunk_size=500, overlap=150):
            tokens = tokenize(section)
            chunks = []
            i = 0
            while i < len(tokens):
                chunk_tokens = tokens[i:i+chunk_size]
                if not chunk_tokens:
                    break
                chunks.append(detokenize(chunk_tokens))
                if i + chunk_size >= len(tokens):
                    break
                i += chunk_size - overlap
            return chunks

        # Helper: Embed a chunk using sentence-transformers
        def embed_chunk(chunk):
            # Validate chunk input
            if not isinstance(chunk, str):
                print(f"⚠️ Warning: Chunk is not a string, got {type(chunk)}: {repr(chunk)}")
                if chunk is None:
                    chunk = ""
                else:
                    chunk = str(chunk)
            
            if not chunk.strip():
                print("⚠️ Warning: Empty chunk detected, using placeholder text")
                chunk = "empty chunk"
            
            if hasattr(self, 'create_embedding') and callable(self.create_embedding):
                emb = self.create_embedding(chunk)
                import numpy as np
                emb = np.array(emb, dtype='float32')
                if emb.shape[0] != embedding_dim:
                    # Pad or truncate if needed
                    if emb.shape[0] > embedding_dim:
                        emb = emb[:embedding_dim]
                    else:
                        emb = np.pad(emb, (0, embedding_dim - emb.shape[0]), 'constant')
                return emb
            # Fallback: random vector
            import numpy as np
            return np.random.randn(embedding_dim).astype('float32')

        # Split into sections
        body, refs = self.split_body_and_references(content)
        sections = split_into_sections(body)


        all_chunks = []
        chunk_metadata = []
        for sec_idx, section in enumerate(sections):
            # Validate section is a string
            if not isinstance(section, str):
                print(f"⚠️ Warning: Section {sec_idx} is not a string, got {type(section)}: {repr(section)}")
                if section is None:
                    continue  # Skip None sections
                else:
                    section = str(section)
            
            if not section.strip():
                print(f"⚠️ Warning: Empty section {sec_idx}, skipping")
                continue
            
            section_chunks = chunk_section(section)
            for chunk_idx, chunk in enumerate(section_chunks):
                # Double-check chunk validity
                if isinstance(chunk, str) and chunk.strip():
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        'section_index': sec_idx,
                        'chunk_index': chunk_idx,
                        'section_title': section.split('\n', 1)[0][:80],
                        'text': chunk,
                        'paper_id': paper.get('id', ''),
                        'paper_title': paper.get('title', ''),
                    })

        # Embed all chunks
        embeddings = [embed_chunk(chunk) for chunk in all_chunks]
        embeddings_np = np.stack(embeddings)

        # Setup or use FAISS DB
        if faiss_db is None:
            index = faiss.IndexFlatL2(embedding_dim)
            faiss_db = index
            print(f"Created new FAISS index (dim={embedding_dim})")
        else:
            index = faiss_db

        # Add embeddings to FAISS DB
        index.add(embeddings_np) # type: ignore
        # TODO: figure out this issue ^
        print(f"Added {len(embeddings)} chunks to FAISS DB.")

        # Optionally, you could store metadata elsewhere (e.g., in a parallel list or DB)
        # Here, we return the chunk metadata with their vector index
        for i, meta in enumerate(chunk_metadata):
            meta['faiss_index'] = index.ntotal - len(chunk_metadata) + i

        return chunk_metadata
    
    def split_body_and_references(self, text: str):
        lines = text.splitlines()
        n = len(lines)
        
        # Step 1: find candidate "References" heading near the end
        candidates = [
            i for i, line in enumerate(lines)
            if re.match(r'^\s*(\d+\s+)?(references|bibliography)\b', line.strip(), re.I)
        ]

        if not candidates:
            return text, ""
        
        # Pick the LAST occurrence, since refs are usually last
        start_idx = candidates[-1]
        
        # Step 2: check if it's in last ~20% of the doc (to avoid false hits)
        if start_idx < 0.8 * n:
            return text, ""  # probably inline mention, not the section
        
        # Step 3: separate body vs refs
        body = "\n".join(lines[:start_idx]).strip()
        refs_block = "\n".join(lines[start_idx:]).strip()
        
        return body, refs_block
