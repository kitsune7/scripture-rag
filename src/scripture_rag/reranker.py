"""Cross-encoder reranker for improving scripture search relevance."""

from sentence_transformers import CrossEncoder


class ScriptureReranker:
    """Reranks scripture search results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Default is ms-marco-MiniLM-L-6-v2 (~80MB, fast and accurate)
        """
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of (index, score) tuples sorted by relevance score (highest first).
            The index refers to the position in the input documents list.
        """
        if not documents:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Create (index, score) tuples and sort by score descending
        ranked_results = [(i, float(score)) for i, score in enumerate(scores)]
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        return ranked_results
