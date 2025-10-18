"""Indexing pipeline for building the scripture vector database."""

from pathlib import Path

from .book_mapping import get_default_mapping
from .downloader import ensure_assets_downloaded
from .parser import parse_all_scripture_files
from .vector_store import ScriptureVectorStore


def index_scriptures(
    assets_dir: str | Path | None = None,
    persist_directory: str | Path | None = None,
    clear_existing: bool = True,
) -> tuple[int, int]:
    """
    Index all scripture files into the vector database.

    Args:
        assets_dir: Path to the assets directory. Defaults to ../../../assets relative to this file
        persist_directory: Directory to persist ChromaDB data. Defaults to ~/.scripture-rag/chroma
        clear_existing: Whether to clear the existing collection before indexing

    Returns:
        Tuple of (number of files processed, number of chunks indexed)
    """
    # Ensure assets are downloaded
    print("Ensuring scripture assets are available...")
    assets_dir = ensure_assets_downloaded(assets_dir)
    print()

    print(f"Loading book mappings from {assets_dir / 'Contents.txt'}...")
    book_mapping = get_default_mapping()
    print(f"Loaded {len(book_mapping)} book mappings")

    print(f"\nParsing scripture files from {assets_dir}...")
    all_chunks = parse_all_scripture_files(assets_dir, book_mapping)
    print(f"Parsed {len(all_chunks)} scripture chunks")

    # Count files (subdirectories with .txt files)
    file_count = sum(
        1 for subdir in assets_dir.iterdir() if subdir.is_dir() for _ in subdir.glob("*.txt")
    )

    print("\nInitializing vector store...")
    vector_store = ScriptureVectorStore(persist_directory=persist_directory)

    if clear_existing:
        print("Clearing existing collection...")
        vector_store.clear_collection()

    print(f"Adding {len(all_chunks)} chunks to vector store...")
    vector_store.add_chunks(all_chunks)

    final_count = vector_store.count()
    print("\nIndexing complete!")
    print(f"  Files processed: {file_count}")
    print(f"  Chunks indexed: {final_count}")

    return file_count, final_count
