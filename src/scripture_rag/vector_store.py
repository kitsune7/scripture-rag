"""ChromaDB vector store for scripture embeddings."""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .parser import ScriptureChunk


class ScriptureVectorStore:
    """Manages the ChromaDB vector store for scripture embeddings."""

    def __init__(self, persist_directory: str | Path | None = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the ChromaDB data.
                             Defaults to ~/.scripture-rag/chroma
        """
        if persist_directory is None:
            persist_directory = Path.home() / ".scripture-rag" / "chroma"
        else:
            persist_directory = Path(persist_directory)

        persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        # Configure the embedding function (sentence-transformers)
        # Using all-mpnet-base-v2 for superior semantic understanding (768 dimensions)
        # Ideal for cross-reference queries and theological concept matching
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Collection name
        self.collection_name = "scriptures"

    def get_or_create_collection(self):
        """Get or create the scriptures collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )

    def clear_collection(self):
        """Delete and recreate the collection (useful for re-indexing)."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection doesn't exist, that's fine

        return self.get_or_create_collection()

    def add_chunks(self, chunks: list[ScriptureChunk], batch_size: int = 100):
        """
        Add scripture chunks to the vector store.

        Args:
            chunks: List of ScriptureChunk objects to add
            batch_size: Number of chunks to process in each batch
        """
        collection = self.get_or_create_collection()

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Prepare data for ChromaDB
            ids = [f"{chunk.prefix}_{chunk.chapter}:{chunk.verse}" for chunk in batch]
            documents = [chunk.text for chunk in batch]
            metadatas = [
                {
                    "book": chunk.book,
                    "prefix": chunk.prefix,
                    "chapter": chunk.chapter,
                    "verse": chunk.verse,
                    "reference": chunk.reference,
                    "section_heading": chunk.section_heading,
                    "source_file": chunk.source_file,
                }
                for chunk in batch
            ]

            # Add to collection
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self, query_text: str, n_results: int = 5, where: dict | None = None
    ) -> dict[str, list[str | dict | float]]:
        """
        Query the vector store for relevant scripture passages.

        Args:
            query_text: The search query
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"book": "Alma"})

        Returns:
            Dictionary containing:
            - documents: List of matching text passages
            - metadatas: List of metadata for each result
            - distances: List of distance scores (lower = more similar)
        """
        collection = self.get_or_create_collection()

        results = collection.query(query_texts=[query_text], n_results=n_results, where=where)

        # Flatten the results (ChromaDB returns nested lists for batch queries)
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def count(self) -> int:
        """Get the number of chunks in the collection."""
        collection = self.get_or_create_collection()
        return collection.count()
