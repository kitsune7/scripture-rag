"""Parse scripture text files into structured chunks with metadata."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScriptureChunk:
    """Represents a single verse chunk with its metadata."""

    text: str
    book: str
    prefix: str
    chapter: int
    verse: int
    reference: str
    section_heading: str
    source_file: str


def parse_scripture_file(
    file_path: str | Path, book_mapping: dict[str, str]
) -> list[ScriptureChunk]:
    """
    Parse a scripture text file into chunks.

    Args:
        file_path: Path to the scripture text file
        book_mapping: Dictionary mapping abbreviations to full book names

    Returns:
        List of ScriptureChunk objects (excluding section headers with verse 0)
    """
    file_path = Path(file_path)
    chunks = []
    current_section_heading = ""

    # Pattern to match lines like: "JON 1:1 Now the word of the LORD..."
    # or: "NE1 1:1 I, Nephi, having been born of goodly parents..."
    pattern = re.compile(r"^([A-Z0-9&]+)\s+(\d+):(\d+)\s+(.+)$")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                continue

            prefix = match.group(1)
            chapter = int(match.group(2))
            verse = int(match.group(3))
            text = match.group(4)

            # Get the full book name from the mapping
            book = book_mapping.get(prefix, prefix)

            # If verse is 0, it's a section heading
            if verse == 0:
                current_section_heading = text
                continue

            # Create a formatted reference
            reference = f"{book} {chapter}:{verse}"

            # Create the chunk
            chunk = ScriptureChunk(
                text=text,
                book=book,
                prefix=prefix,
                chapter=chapter,
                verse=verse,
                reference=reference,
                section_heading=current_section_heading,
                source_file=str(file_path),
            )
            chunks.append(chunk)

    return chunks


def parse_all_scripture_files(
    assets_dir: str | Path, book_mapping: dict[str, str]
) -> list[ScriptureChunk]:
    """
    Parse all scripture text files in the assets directory.

    Args:
        assets_dir: Path to the assets directory
        book_mapping: Dictionary mapping abbreviations to full book names

    Returns:
        List of all ScriptureChunk objects from all files
    """
    assets_dir = Path(assets_dir)
    all_chunks = []

    # Walk through all subdirectories
    for subdir in assets_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Process all .txt files in each subdirectory
        for txt_file in subdir.glob("*.txt"):
            try:
                chunks = parse_scripture_file(txt_file, book_mapping)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to parse {txt_file}: {e}")

    return all_chunks
