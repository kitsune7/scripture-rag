"""Tests for scripture reranker functionality."""

import pytest

from scripture_rag.reranker import ScriptureReranker


class TestScriptureReranker:
    """Tests for ScriptureReranker class."""

    def test_reranker_initialization(self):
        """Test reranker initialization with default model."""
        reranker = ScriptureReranker()

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # Model should not be loaded yet (lazy loading)
        assert reranker._model is None

    def test_reranker_initialization_custom_model(self):
        """Test reranker initialization with custom model."""
        custom_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        reranker = ScriptureReranker(model_name=custom_model)

        assert reranker.model_name == custom_model
        assert reranker._model is None

    def test_reranker_lazy_loading(self, mocker):
        """Test that model is lazy loaded on first access."""
        # Mock the CrossEncoder class
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance

        reranker = ScriptureReranker()
        assert reranker._model is None

        # Access the model property
        model = reranker.model

        # Model should now be loaded
        assert model is mock_model_instance
        mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_rerank_basic(self, mocker):
        """Test basic reranking functionality."""
        # Mock the CrossEncoder
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance

        # Mock scores (higher = more relevant)
        mock_model_instance.predict.return_value = [0.8, 0.3, 0.9, 0.5]

        reranker = ScriptureReranker()
        query = "faith and prayer"
        documents = [
            "Have faith in God",
            "The weather is nice",
            "Prayer brings peace",
            "Faith without works",
        ]

        results = reranker.rerank(query, documents)

        # Should return list of (index, score) tuples
        assert len(results) == 4
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

        # Should be sorted by score (highest first)
        # Expected order: doc 2 (0.9), doc 0 (0.8), doc 3 (0.5), doc 1 (0.3)
        assert results[0] == (2, 0.9)
        assert results[1] == (0, 0.8)
        assert results[2] == (3, 0.5)
        assert results[3] == (1, 0.3)

        # Verify the correct pairs were passed to the model
        expected_pairs = [
            [query, "Have faith in God"],
            [query, "The weather is nice"],
            [query, "Prayer brings peace"],
            [query, "Faith without works"],
        ]
        mock_model_instance.predict.assert_called_once_with(expected_pairs)

    def test_rerank_with_top_k(self, mocker):
        """Test reranking with top_k limit."""
        # Mock the CrossEncoder
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance

        mock_model_instance.predict.return_value = [0.8, 0.3, 0.9, 0.5, 0.7]

        reranker = ScriptureReranker()
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        results = reranker.rerank("query", documents, top_k=3)

        # Should only return top 3 results
        assert len(results) == 3
        # Should be the highest scoring ones
        assert results[0] == (2, 0.9)
        assert results[1] == (0, 0.8)
        assert results[2] == (4, 0.7)

    def test_rerank_empty_documents(self, mocker):
        """Test reranking with empty document list."""
        reranker = ScriptureReranker()
        results = reranker.rerank("query", [])

        assert results == []

    def test_rerank_single_document(self, mocker):
        """Test reranking with single document."""
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance
        mock_model_instance.predict.return_value = [0.75]

        reranker = ScriptureReranker()
        results = reranker.rerank("query", ["single doc"])

        assert len(results) == 1
        assert results[0] == (0, 0.75)

    def test_rerank_top_k_none(self, mocker):
        """Test reranking with top_k=None returns all results."""
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance
        mock_model_instance.predict.return_value = [0.5, 0.8, 0.3]

        reranker = ScriptureReranker()
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=None)

        # Should return all documents
        assert len(results) == 3

    def test_rerank_top_k_larger_than_documents(self, mocker):
        """Test reranking when top_k is larger than number of documents."""
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance
        mock_model_instance.predict.return_value = [0.5, 0.8]

        reranker = ScriptureReranker()
        results = reranker.rerank("query", ["doc1", "doc2"], top_k=10)

        # Should return all available documents
        assert len(results) == 2

    def test_rerank_preserves_indices(self, mocker):
        """Test that reranking preserves original document indices."""
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance

        # Scores that will reverse the order
        mock_model_instance.predict.return_value = [0.1, 0.2, 0.3]

        reranker = ScriptureReranker()
        documents = ["first", "second", "third"]
        results = reranker.rerank("query", documents)

        # Results should reference original indices
        assert results[0][0] == 2  # "third" had highest score
        assert results[1][0] == 1  # "second" had middle score
        assert results[2][0] == 0  # "first" had lowest score

        # Indices should allow us to retrieve original documents
        for idx, score in results:
            assert documents[idx] is not None

    def test_rerank_scores_are_floats(self, mocker):
        """Test that scores are converted to Python floats."""
        mock_cross_encoder = mocker.patch("scripture_rag.reranker.CrossEncoder")
        mock_model_instance = mocker.Mock()
        mock_cross_encoder.return_value = mock_model_instance

        # Return numpy-like array elements
        import numpy as np

        mock_model_instance.predict.return_value = np.array([0.5, 0.8])

        reranker = ScriptureReranker()
        results = reranker.rerank("query", ["doc1", "doc2"])

        # Scores should be Python floats
        for idx, score in results:
            assert isinstance(score, float)
            assert not isinstance(score, np.floating)
