"""Run: python -m unittest discover -s tests -v (no API key needed)."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from rag import chunk_pages, embed_texts, find_relevant_chunks, answer_with_context, extract_pdf_pages


class TestRag(unittest.TestCase):
    def test_extract_sample_pdf(self):
        with open("sample_document.pdf", "rb") as f:
            pages = extract_pdf_pages(f.read())
        self.assertGreaterEqual(len(pages), 2)
        self.assertTrue(any("robotics" in p["text"].lower() for p in pages))

    def test_chunk_overlap_and_page(self):
        pages = [{"page": 4, "text": "one two three four five six seven eight nine ten"}]
        result = chunk_pages(pages, chunk_size=6, overlap=2)
        self.assertEqual(result[0]["text"], "one two three four five six")
        self.assertEqual(result[1]["text"], "five six seven eight nine ten")
        self.assertEqual([x["page"] for x in result], [4, 4])

    def test_gemini_embedding_task_and_output(self):
        client = MagicMock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])
        result = embed_texts(client, ["hello"], task_type="RETRIEVAL_QUERY")
        self.assertEqual(result.tolist(), [[1.0, 0.0]])
        self.assertEqual(client.models.embed_content.call_args.kwargs["config"]["task_type"], "RETRIEVAL_QUERY")

    def test_retrieval_selects_closest_page(self):
        client = MagicMock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])
        chunks = [{"page": 1, "id": "p1-c1", "text": "Apples"}, {"page": 2, "id": "p2-c1", "text": "Robots"}]
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        found = find_relevant_chunks(client, "robotics", chunks, vectors, top_k=1)
        self.assertEqual(found[0]["page"], 2)
        self.assertAlmostEqual(found[0]["similarity"], 1.0)

    def test_answer_receives_page_markers(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text="Tuesday, 4 PM (Page 1).")
        answer = answer_with_context(client, "When?", [{"page": 1, "id": "p1-c1", "text": "Meet Tuesday, 4 PM."}])
        self.assertEqual(answer, "Tuesday, 4 PM (Page 1).")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertIn("Page 1", kwargs["contents"])
        self.assertIn("system_instruction", kwargs["config"])


if __name__ == "__main__":
    unittest.main()
