"""Command line interface for Scripture RAG application."""

import argparse
import sys

from .indexer import index_scriptures
from .query import ScriptureQueryEngine


def cmd_index(args):
    """Handle the index command."""
    try:
        file_count, chunk_count = index_scriptures(
            assets_dir=args.assets_dir, clear_existing=not args.append
        )
        print(f"\n✓ Successfully indexed {chunk_count} verses from {file_count} files")
        return 0
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        return 1


def cmd_query(args):
    """Handle the query command."""
    try:
        engine = ScriptureQueryEngine()

        # Check if database is empty
        count = engine.vector_store.count()
        if count == 0:
            print("Error: Vector database is empty. Run 'scripture-rag index' first.")
            return 1

        # Parse book filter if provided
        books = None
        if args.book:
            books = args.book[0] if len(args.book) == 1 else args.book

        # Display filter info if applicable
        if books:
            filter_msg = books if isinstance(books, str) else ", ".join(books)
            print(f"Filtering results to: {filter_msg}")

        # Perform query
        response = engine.query(
            query=args.query,
            top_k=args.top_k,
            use_llm=args.answer,
            books=books,
            use_reranker=args.reranker,
            retrieval_factor=args.retrieval_factor,
        )

        # Display LLM answer if available
        if response.answer:
            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(response.answer)
            print()

        # Display search results
        print("\n" + "=" * 80)
        print("RELEVANT SCRIPTURE PASSAGES")
        print("=" * 80)

        for i, result in enumerate(response.results, 1):
            print(f"\n[{i}] {result.reference}")
            if result.section_heading:
                print(f"    Context: {result.section_heading}")
            print(f"    {result.text}")
            # Show reranker score if available, otherwise show distance-based score
            if result.reranker_score is not None:
                print(
                    f"    (reranker score: {result.reranker_score:.3f}, "
                    f"vector distance: {result.distance:.3f})"
                )
            else:
                print(f"    (relevance score: {1 - result.distance:.3f})")

        return 0

    except Exception as e:
        print(f"Error during query: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scripture RAG - Semantic search and Q&A over scripture texts"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index scripture files into vector database")
    index_parser.add_argument(
        "--assets-dir", type=str, help="Path to assets directory (default: auto-detect)"
    )
    index_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of clearing it",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the scripture database")
    query_parser.add_argument("query", type=str, help="Search query or question")
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return (default: 5)"
    )
    query_parser.add_argument(
        "--no-answer",
        dest="answer",
        action="store_false",
        help="Skip LLM answer generation, only show search results",
    )
    query_parser.add_argument(
        "--book",
        type=str,
        action="append",
        help="Filter by book name (can be used multiple times, e.g., --book Alma --book Moroni)",
    )
    query_parser.add_argument(
        "--no-reranker",
        dest="reranker",
        action="store_false",
        help="Disable cross-encoder reranking (faster but less accurate)",
    )
    query_parser.add_argument(
        "--retrieval-factor",
        type=float,
        default=3.0,
        help="Multiplier for initial retrieval when reranking (default: 3.0)",
    )

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "query":
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
