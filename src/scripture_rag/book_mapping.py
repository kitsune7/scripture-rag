"""Parse Contents.txt to create a mapping from abbreviations to full book names."""

import re
from pathlib import Path


def load_book_mapping(contents_path: str | Path) -> dict[str, str]:
    """
    Parse the Contents.txt file to extract abbreviation to book name mappings.

    Args:
        contents_path: Path to the Contents.txt file

    Returns:
        Dictionary mapping abbreviations (e.g., "GEN") to full book names (e.g., "Genesis")
    """
    mapping = {}
    contents_path = Path(contents_path)

    if not contents_path.exists():
        raise FileNotFoundError(f"Contents file not found: {contents_path}")

    with open(contents_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match lines like: "Genesis   . . . . . . . . . . . . . . . . .   GEN "
            # or: "Doctrine-and-Covenants  . . . . . . . . . .   D&C"
            # or: "1-Nephi   . . . . . . . . . . . . . . . . .   NE1"
            # Pattern accounts for spaces between dots and numbers in abbreviations
            match = re.search(r"^([A-Za-z0-9\-&]+)\s+\.\s+\.\s+.*\s+([A-Z0-9&]+)\s*$", line)
            if match:
                book_name = match.group(1)
                abbreviation = match.group(2)
                mapping[abbreviation] = book_name

    return mapping


def get_default_mapping() -> dict[str, str]:
    """
    Get the book mapping using the default Contents.txt location.

    Returns:
        Dictionary mapping abbreviations to full book names
    """
    # Determine the path to Contents.txt relative to this file
    this_file = Path(__file__)
    project_root = this_file.parent.parent.parent
    contents_path = project_root / "assets" / "Contents.txt"

    return load_book_mapping(contents_path)
