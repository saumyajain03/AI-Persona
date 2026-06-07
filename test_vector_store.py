import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from vector_store import RAGVectorStore

class TestRAGVectorStore(unittest.TestCase):
    def setUp(self):
        self.persist_dir = "./test_chroma_db"
        # Remove it if it already exists from a crashed run
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

    def tearDown(self):
        # Clean up temporary database directory
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

    def test_chunking_and_chroma_ingestion_mocked(self):
        """Tests document ingestion, chunking, embedding calling, and retrieval using Chroma."""
        # Mock embedding of correct dimension for the provider (768 for gemini default when no key)
        mock_embedding = [0.1] * 768
        
        # Patch the _get_embeddings and _get_query_embedding methods directly
        with patch.object(RAGVectorStore, '_get_embeddings') as mock_get_emb, \
             patch.object(RAGVectorStore, '_get_query_embedding') as mock_get_query_emb:
            
            # Setup mock returns
            mock_get_emb.return_value = [mock_embedding, mock_embedding]  # 2 docs -> 2 chunks
            mock_get_query_emb.return_value = mock_embedding
            
            # Initialize store (no API key needed since we mock the embedding methods)
            store = RAGVectorStore(
                db_type="chroma",
                index_name="test_collection",
                persist_directory=self.persist_dir,
                embedding_provider="gemini"  # Set explicitly to avoid key detection warnings
            )
            
            # Standard documents to ingest
            docs = [
                {
                    "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
                    "metadata": {"source": "python_wiki", "category": "coding"}
                },
                {
                    "content": "GitHub is a developer platform that allows developers to create, store, manage and share their code.",
                    "metadata": {"source": "github_wiki", "category": "platform"}
                }
            ]
            
            # Run ingestion
            store.ingest_documents(docs)
            
            # Verify the embedding method was called
            self.assertTrue(mock_get_emb.called)
            
            # Run a semantic query
            results = store.query("coding language", top_k=2)
            
            # Assertions
            self.assertEqual(len(results), 2)
            self.assertIn("content", results[0])
            self.assertIn("metadata", results[0])
            self.assertIn("score", results[0])
            self.assertEqual(results[0]["metadata"]["category"], "coding")
            self.assertEqual(results[1]["metadata"]["category"], "platform")
            
            print("[Test Log] Mocked ingestion and query tests passed successfully.")

    @patch('pdfplumber.open')
    def test_parse_resume_pdf(self, mock_pdf_open):
        """Tests that pdfplumber successfully parses PDF pages and returns consolidated text."""
        from ingest_all import parse_resume
        # Setup mock PDF structure
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Resume Page 1: John Doe, software engineer."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Resume Page 2: Experience at Google and DeepMind."
        
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        # Run function
        text = parse_resume("mock_resume.pdf")
        
        # Verify
        mock_pdf_open.assert_called_once_with("mock_resume.pdf")
        self.assertIn("John Doe", text)
        self.assertIn("Page Break", text)
        self.assertIn("DeepMind", text)
        print("[Test Log] pdfplumber mock test passed successfully.")

if __name__ == '__main__':
    unittest.main()
