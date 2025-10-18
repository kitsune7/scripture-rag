"""Tests for query engine functionality."""

import pytest

from scripture_rag.query import QueryResult, RAGResponse, ScriptureQueryEngine


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating a QueryResult with all fields."""
        result = QueryResult(
            reference="Genesis 1:1",
            text="In the beginning God created the heaven and the earth.",
            section_heading="The Creation",
            book="Genesis",
            chapter=1,
            verse=1,
            distance=0.25,
            reranker_score=0.95,
        )

        assert result.reference == "Genesis 1:1"
        assert result.text == "In the beginning God created the heaven and the earth."
        assert result.section_heading == "The Creation"
        assert result.book == "Genesis"
        assert result.chapter == 1
        assert result.verse == 1
        assert result.distance == 0.25
        assert result.reranker_score == 0.95

    def test_query_result_without_reranker_score(self):
        """Test creating a QueryResult without reranker score."""
        result = QueryResult(
            reference="Genesis 1:1",
            text="Test text",
            section_heading="",
            book="Genesis",
            chapter=1,
            verse=1,
            distance=0.5,
        )

        assert result.reranker_score is None


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_rag_response_creation(self):
        """Test creating a RAGResponse with all fields."""
        results = [
            QueryResult(
                reference="Genesis 1:1",
                text="Test",
                section_heading="",
                book="Genesis",
                chapter=1,
                verse=1,
                distance=0.5,
            )
        ]

        response = RAGResponse(
            query="What is the creation story?", results=results, answer="This is the answer."
        )

        assert response.query == "What is the creation story?"
        assert len(response.results) == 1
        assert response.answer == "This is the answer."

    def test_rag_response_without_answer(self):
        """Test creating a RAGResponse without LLM answer."""
        results = []
        response = RAGResponse(query="test query", results=results)

        assert response.query == "test query"
        assert response.results == []
        assert response.answer is None


class TestScriptureQueryEngineInit:
    """Tests for ScriptureQueryEngine initialization."""

    def test_initialization_without_api_key(self, mocker, tmp_path):
        """Test initialization without Gemini API key."""
        # Mock environment to not have GEMINI_API_KEY
        mocker.patch("os.getenv", return_value=None)

        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        assert engine.vector_store is not None
        assert engine.reranker is not None
        assert engine.llm_available is False

    def test_initialization_with_api_key(self, mocker, tmp_path):
        """Test initialization with Gemini API key."""
        # Mock the genai module
        mock_genai = mocker.Mock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})
        mocker.patch("scripture_rag.query.genai", mock_genai)

        api_key = "test_api_key"
        engine = ScriptureQueryEngine(persist_directory=tmp_path, api_key=api_key)

        # Should have configured Gemini
        mock_genai.configure.assert_called_once_with(api_key=api_key)
        assert engine.llm_available is True

    def test_initialization_with_env_api_key(self, mocker, tmp_path):
        """Test initialization with API key from environment."""
        # Mock environment variable
        mocker.patch("os.getenv", return_value="env_api_key")
        mock_genai = mocker.Mock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})
        mocker.patch("scripture_rag.query.genai", mock_genai)

        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_genai.configure.assert_called_once_with(api_key="env_api_key")
        assert engine.llm_available is True


class TestScriptureQueryEngineSearch:
    """Tests for ScriptureQueryEngine.search method."""

    def test_search_basic(self, mocker, tmp_path):
        """Test basic search without filters."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        # Mock the vector store query method
        mock_results = {
            "documents": ["Text 1", "Text 2"],
            "metadatas": [
                {
                    "reference": "Genesis 1:1",
                    "book": "Genesis",
                    "chapter": 1,
                    "verse": 1,
                    "section_heading": "",
                },
                {
                    "reference": "Genesis 1:2",
                    "book": "Genesis",
                    "chapter": 1,
                    "verse": 2,
                    "section_heading": "",
                },
            ],
            "distances": [0.2, 0.4],
        }
        mocker.patch.object(engine.vector_store, "query", return_value=mock_results)

        results = engine.search("creation", top_k=2, use_reranker=False)

        assert len(results) == 2
        assert results[0].reference == "Genesis 1:1"
        assert results[0].distance == 0.2
        assert results[1].reference == "Genesis 1:2"

    def test_search_with_single_book_filter(self, mocker, tmp_path):
        """Test search with single book filter."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_query = mocker.patch.object(engine.vector_store, "query")
        mock_query.return_value = {"documents": [], "metadatas": [], "distances": []}

        engine.search("faith", top_k=5, books="Alma", use_reranker=False)

        # Verify the where filter was passed correctly
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["where"] == {"book": "Alma"}

    def test_search_with_multiple_book_filter(self, mocker, tmp_path):
        """Test search with multiple book filter."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_query = mocker.patch.object(engine.vector_store, "query")
        mock_query.return_value = {"documents": [], "metadatas": [], "distances": []}

        engine.search("faith", top_k=5, books=["Alma", "Moroni"], use_reranker=False)

        # Verify the where filter was passed correctly
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["where"] == {"book": {"$in": ["Alma", "Moroni"]}}

    def test_search_with_empty_book_list(self, mocker, tmp_path):
        """Test search with empty book list."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_query = mocker.patch.object(engine.vector_store, "query")
        mock_query.return_value = {"documents": [], "metadatas": [], "distances": []}

        engine.search("faith", top_k=5, books=[], use_reranker=False)

        # Empty list should result in no filter
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["where"] is None

    def test_search_retrieval_factor_without_reranker(self, mocker, tmp_path):
        """Test that retrieval_factor is not used when reranker is disabled."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_query = mocker.patch.object(engine.vector_store, "query")
        mock_query.return_value = {"documents": [], "metadatas": [], "distances": []}

        engine.search("test", top_k=5, use_reranker=False, retrieval_factor=3.0)

        # Should request exactly top_k results
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["n_results"] == 5

    def test_search_retrieval_factor_with_reranker(self, mocker, tmp_path):
        """Test that retrieval_factor is applied when reranker is enabled."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_metadata = {
            "reference": "Gen 1:1",
            "book": "Genesis",
            "chapter": 1,
            "verse": 1,
            "section_heading": "",
        }
        mock_query = mocker.patch.object(engine.vector_store, "query")
        mock_query.return_value = {
            "documents": ["doc"] * 15,
            "metadatas": [mock_metadata] * 15,
            "distances": [0.5] * 15,
        }

        # Mock the reranker
        mocker.patch.object(
            engine.reranker, "rerank", return_value=[(i, 0.5) for i in range(5)]
        )

        engine.search("test", top_k=5, use_reranker=True, retrieval_factor=3.0)

        # Should request top_k * retrieval_factor results
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["n_results"] == 15  # 5 * 3.0

    def test_search_with_reranker(self, mocker, tmp_path):
        """Test search with reranker enabled."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        # Mock vector store results
        mock_results = {
            "documents": ["Text 1", "Text 2", "Text 3"],
            "metadatas": [
                {"reference": "Gen 1:1", "book": "Genesis", "chapter": 1, "verse": 1, "section_heading": ""},
                {"reference": "Gen 1:2", "book": "Genesis", "chapter": 1, "verse": 2, "section_heading": ""},
                {"reference": "Gen 1:3", "book": "Genesis", "chapter": 1, "verse": 3, "section_heading": ""},
            ],
            "distances": [0.5, 0.4, 0.6],
        }
        mocker.patch.object(engine.vector_store, "query", return_value=mock_results)

        # Mock reranker to reverse the order and return top 2
        mocker.patch.object(engine.reranker, "rerank", return_value=[(2, 0.95), (0, 0.85)])

        results = engine.search("test", top_k=2, use_reranker=True)

        # Should return reranked results
        assert len(results) == 2
        assert results[0].reference == "Gen 1:3"
        assert results[0].reranker_score == 0.95
        assert results[1].reference == "Gen 1:1"
        assert results[1].reranker_score == 0.85


class TestScriptureQueryEngineQuery:
    """Tests for ScriptureQueryEngine.query method."""

    def test_query_without_llm(self, mocker, tmp_path):
        """Test query method without LLM."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)
        engine.llm_available = False

        # Mock search method
        mock_results = [
            QueryResult(
                reference="Genesis 1:1",
                text="Test",
                section_heading="",
                book="Genesis",
                chapter=1,
                verse=1,
                distance=0.5,
            )
        ]
        mocker.patch.object(engine, "search", return_value=mock_results)

        response = engine.query("creation", use_llm=True)

        assert response.query == "creation"
        assert response.results == mock_results
        assert response.answer is None

    def test_query_with_llm_disabled(self, mocker, tmp_path):
        """Test query method with use_llm=False."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)
        engine.llm_available = True

        mock_results = [
            QueryResult(
                reference="Genesis 1:1",
                text="Test",
                section_heading="",
                book="Genesis",
                chapter=1,
                verse=1,
                distance=0.5,
            )
        ]
        mocker.patch.object(engine, "search", return_value=mock_results)

        response = engine.query("creation", use_llm=False)

        assert response.query == "creation"
        assert response.results == mock_results
        assert response.answer is None

    def test_query_passes_parameters_to_search(self, mocker, tmp_path):
        """Test that query method passes parameters to search."""
        engine = ScriptureQueryEngine(persist_directory=tmp_path)

        mock_search = mocker.patch.object(engine, "search", return_value=[])

        engine.query("test", top_k=10, books="Alma", use_llm=False, use_reranker=True, retrieval_factor=2.5)

        mock_search.assert_called_once_with(
            "test", top_k=10, books="Alma", use_reranker=True, retrieval_factor=2.5
        )
