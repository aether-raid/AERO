from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
create_embedding = lambda text: embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
