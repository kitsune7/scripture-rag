"""RAG query engine with Gemini LLM integration."""

import os
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai

from .downloader import ensure_assets_downloaded
from .reranker import ScriptureReranker
from .vector_store import ScriptureVectorStore


@dataclass
class QueryResult:
    """Result of a scripture query."""

    reference: str
    text: str
    section_heading: str
    book: str
    chapter: int
    verse: int
    distance: float
    reranker_score: float | None = None


@dataclass
class RAGResponse:
    """Response from the RAG engine including LLM-generated answer."""

    query: str
    results: list[QueryResult]
    answer: str | None = None


class ScriptureQueryEngine:
    """RAG query engine for scripture search with LLM integration."""

    def __init__(self, persist_directory: str | Path | None = None, api_key: str | None = None):
        """
        Initialize the query engine.

        Args:
            persist_directory: Directory where ChromaDB data is persisted
            api_key: Gemini API key. If not provided, uses GEMINI_API_KEY env var
        """
        # Ensure scripture assets are available
        ensure_assets_downloaded()

        self.vector_store = ScriptureVectorStore(persist_directory=persist_directory)
        self.reranker = ScriptureReranker()

        # Configure Gemini
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            self.llm_available = True
        else:
            self.llm_available = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        books: str | list[str] | None = None,
        use_reranker: bool = True,
        retrieval_factor: float = 3.0,
    ) -> list[QueryResult]:
        """
        Search for relevant scripture passages.

        Args:
            query: Search query
            top_k: Number of results to return
            books: Optional book name(s) to filter by. Can be a single book name
                   or a list of book names (e.g., "Alma" or ["Alma", "Moroni"])
            use_reranker: Whether to use the cross-encoder reranker (default: True)
            retrieval_factor: Multiplier for initial retrieval when reranking
                            (default: 3.0, retrieves top_k * 3.0 candidates)

        Returns:
            List of QueryResult objects
        """
        # Build the where filter for book filtering
        where = None
        if books is not None:
            if isinstance(books, str):
                where = {"book": books}
            elif isinstance(books, list) and books:
                where = {"book": {"$in": books}}

        # Determine how many results to retrieve
        n_results = top_k
        if use_reranker:
            n_results = max(top_k, int(top_k * retrieval_factor))

        results_dict = self.vector_store.query(query, n_results=n_results, where=where)

        # Build initial query results
        query_results = []
        for i in range(len(results_dict["documents"])):
            metadata = results_dict["metadatas"][i]
            query_results.append(
                QueryResult(
                    reference=metadata["reference"],
                    text=results_dict["documents"][i],
                    section_heading=metadata["section_heading"],
                    book=metadata["book"],
                    chapter=metadata["chapter"],
                    verse=metadata["verse"],
                    distance=results_dict["distances"][i],
                )
            )

        # Apply reranking if enabled
        if use_reranker and query_results:
            documents = [result.text for result in query_results]
            reranked = self.reranker.rerank(query, documents, top_k=top_k)

            # Reorder results based on reranker scores and add scores
            reranked_results = []
            for original_idx, score in reranked:
                result = query_results[original_idx]
                result.reranker_score = score
                reranked_results.append(result)

            return reranked_results

        return query_results[:top_k]

    def query_with_llm(
        self,
        query: str,
        top_k: int = 5,
        books: str | list[str] | None = None,
        use_reranker: bool = True,
        retrieval_factor: float = 3.0,
    ) -> RAGResponse:
        """
        Query scriptures and generate an answer using the LLM.

        Args:
            query: User's question
            top_k: Number of scripture passages to retrieve for context
            books: Optional book name(s) to filter by
            use_reranker: Whether to use the cross-encoder reranker (default: True)
            retrieval_factor: Multiplier for initial retrieval when reranking

        Returns:
            RAGResponse with search results and LLM-generated answer
        """
        # Get relevant scripture passages
        results = self.search(
            query,
            top_k=top_k,
            books=books,
            use_reranker=use_reranker,
            retrieval_factor=retrieval_factor,
        )

        # Build context from retrieved passages
        context_parts = []
        for result in results:
            context_parts.append(f"[{result.reference}] {result.text}")

        context = "\n\n".join(context_parts)

        # Generate answer with LLM if available
        answer = None
        if self.llm_available and context:
            prompt = f"""You are a helpful assistant that answers questions about \
scripture passages.

Question: {query}

Relevant scripture passages:
{context}

Please provide a helpful answer based on the scripture passages above. Include \
citations in the format [Book Chapter:Verse] when referencing specific passages. \
Keep your answer concise and accurate."""

            try:
                response = self.model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                print(f"Warning: Failed to generate LLM response: {e}")

        return RAGResponse(query=query, results=results, answer=answer)

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_llm: bool = True,
        books: str | list[str] | None = None,
        use_reranker: bool = True,
        retrieval_factor: float = 3.0,
    ) -> RAGResponse:
        """
        Query the scripture database.

        Args:
            query: Search query or question
            top_k: Number of results to return
            use_llm: Whether to use the LLM for answer generation
            books: Optional book name(s) to filter by
            use_reranker: Whether to use the cross-encoder reranker (default: True)
            retrieval_factor: Multiplier for initial retrieval when reranking

        Returns:
            RAGResponse with results and optional LLM answer
        """
        if use_llm and self.llm_available:
            return self.query_with_llm(
                query,
                top_k=top_k,
                books=books,
                use_reranker=use_reranker,
                retrieval_factor=retrieval_factor,
            )
        else:
            results = self.search(
                query,
                top_k=top_k,
                books=books,
                use_reranker=use_reranker,
                retrieval_factor=retrieval_factor,
            )
            return RAGResponse(query=query, results=results, answer=None)
