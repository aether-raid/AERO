import unittest
from arxiv_paper_utils import ArxivPaperProcessor

class DummyLLMClient:
    pass

def get_dummy_paper():
    return {
        'id': 'test123',
        'title': 'Test Paper',
        'content': (
            'INTRODUCTION\nThis is the introduction section. It explains the background.\n'
            'METHODS\nThis section describes the methods used.\n'
            'RESULTS\nHere are the results.\n'
            'CONCLUSION\nThis is the conclusion.'
        )
    }

class TestChunkAndEmbed(unittest.TestCase):
    def setUp(self):
        self.processor = ArxivPaperProcessor(DummyLLMClient(), 'gpt-3.5-turbo')

    def test_chunking(self):
        import asyncio
        paper = get_dummy_paper()
        # Run the async chunk_and_embed method
        chunks = asyncio.run(self.processor.chunk_and_embed(paper))
        print(f"Chunks returned: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: Section {chunk['section_index']} - {chunk['section_title']}")
            print(f"Text: {chunk['text'][:60]}...")
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all('text' in c for c in chunks))
        if getattr(self.processor, 'embedding_model', None) is None:
            print("[Warning] Embedding model is not available. Install torch and sentence-transformers for real embeddings.")

if __name__ == '__main__':
    unittest.main()
