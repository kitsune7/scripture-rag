"""Tests for scripture parsing functionality."""

from pathlib import Path

import pytest

from scripture_rag.parser import ScriptureChunk, parse_all_scripture_files, parse_scripture_file


class TestScriptureChunk:
    """Tests for ScriptureChunk dataclass."""

    def test_scripture_chunk_creation(self):
        """Test creating a ScriptureChunk with all fields."""
        chunk = ScriptureChunk(
            text="Test verse text",
            book="Genesis",
            prefix="GEN",
            chapter=1,
            verse=1,
            reference="Genesis 1:1",
            section_heading="In the beginning",
            source_file="/path/to/file.txt",
        )

        assert chunk.text == "Test verse text"
        assert chunk.book == "Genesis"
        assert chunk.prefix == "GEN"
        assert chunk.chapter == 1
        assert chunk.verse == 1
        assert chunk.reference == "Genesis 1:1"
        assert chunk.section_heading == "In the beginning"
        assert chunk.source_file == "/path/to/file.txt"


class TestParseScriptureFile:
    """Tests for parse_scripture_file function."""

    def test_parse_basic_scripture_file(self, temp_scripture_file, sample_book_mapping):
        """Test parsing a basic scripture file."""
        chunks = parse_scripture_file(temp_scripture_file, sample_book_mapping)

        # Should have verses but not section headings (verse 0)
        assert len(chunks) > 0

        # Check that all chunks are ScriptureChunk objects
        assert all(isinstance(chunk, ScriptureChunk) for chunk in chunks)

    def test_parse_scripture_file_excludes_verse_zero(
        self, temp_scripture_file, sample_book_mapping
    ):
        """Test that verse 0 (section headings) are excluded from chunks."""
        chunks = parse_scripture_file(temp_scripture_file, sample_book_mapping)

        # No chunk should have verse 0
        assert all(chunk.verse != 0 for chunk in chunks)

    def test_parse_scripture_file_section_heading_carried_forward(
        self, temp_scripture_file, sample_book_mapping
    ):
        """Test that section headings are carried forward to subsequent verses."""
        chunks = parse_scripture_file(temp_scripture_file, sample_book_mapping)

        # Find Ruth verses
        ruth_chunks = [c for c in chunks if c.prefix == "RTH"]
        assert len(ruth_chunks) > 0

        # All Ruth chunks should have the section heading
        for chunk in ruth_chunks:
            assert chunk.section_heading == "Ruth and Naomi"

        # Find Jonah verses
        jonah_chunks = [c for c in chunks if c.prefix == "JON"]
        assert len(jonah_chunks) > 0

        # All Jonah chunks should have their section heading
        for chunk in jonah_chunks:
            assert chunk.section_heading == "Jonah Sent to Nineveh"

    def test_parse_scripture_file_correct_metadata(
        self, temp_scripture_file, sample_book_mapping
    ):
        """Test that chunks have correct metadata."""
        chunks = parse_scripture_file(temp_scripture_file, sample_book_mapping)

        # Get first Ruth verse
        ruth_verse = next(c for c in chunks if c.prefix == "RTH" and c.verse == 1)

        assert ruth_verse.book == "Ruth"
        assert ruth_verse.prefix == "RTH"
        assert ruth_verse.chapter == 1
        assert ruth_verse.verse == 1
        assert ruth_verse.reference == "Ruth 1:1"
        assert "famine in the land" in ruth_verse.text
        assert ruth_verse.source_file == str(temp_scripture_file)

    def test_parse_scripture_file_reference_format(
        self, temp_scripture_file, sample_book_mapping
    ):
        """Test that references are formatted correctly."""
        chunks = parse_scripture_file(temp_scripture_file, sample_book_mapping)

        for chunk in chunks:
            # Reference should be in format "Book Chapter:Verse"
            expected_ref = f"{chunk.book} {chunk.chapter}:{chunk.verse}"
            assert chunk.reference == expected_ref

    def test_parse_scripture_file_accepts_path_object(self, tmp_path, sample_book_mapping):
        """Test that function accepts Path objects."""
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text("GEN 1:1 In the beginning God created the heaven and the earth.")

        chunks = parse_scripture_file(Path(scripture_file), sample_book_mapping)
        assert len(chunks) == 1

    def test_parse_scripture_file_accepts_string(self, tmp_path, sample_book_mapping):
        """Test that function accepts string paths."""
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text("GEN 1:1 In the beginning God created the heaven and the earth.")

        chunks = parse_scripture_file(str(scripture_file), sample_book_mapping)
        assert len(chunks) == 1

    def test_parse_scripture_file_empty_lines_ignored(self, tmp_path, sample_book_mapping):
        """Test that empty lines are ignored."""
        content = """GEN 1:1 In the beginning God created the heaven and the earth.

GEN 1:2 And the earth was without form, and void.


"""
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text(content)

        chunks = parse_scripture_file(scripture_file, sample_book_mapping)
        assert len(chunks) == 2

    def test_parse_scripture_file_malformed_lines_ignored(self, tmp_path, sample_book_mapping):
        """Test that malformed lines are skipped."""
        content = """GEN 1:1 In the beginning God created the heaven and the earth.
This is not a valid scripture line
Another invalid line without prefix
GEN 1:2 And the earth was without form, and void.
"""
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text(content)

        chunks = parse_scripture_file(scripture_file, sample_book_mapping)
        assert len(chunks) == 2
        assert chunks[0].verse == 1
        assert chunks[1].verse == 2

    def test_parse_scripture_file_unknown_abbreviation(self, tmp_path):
        """Test handling of unknown book abbreviations."""
        content = "UNK 1:1 This book abbreviation is not in the mapping."
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text(content)

        # Empty mapping
        chunks = parse_scripture_file(scripture_file, {})

        # Should use the abbreviation as the book name if not in mapping
        assert len(chunks) == 1
        assert chunks[0].book == "UNK"
        assert chunks[0].prefix == "UNK"

    def test_parse_scripture_file_empty_file(self, tmp_path, sample_book_mapping):
        """Test parsing an empty file."""
        scripture_file = tmp_path / "empty.txt"
        scripture_file.write_text("")

        chunks = parse_scripture_file(scripture_file, sample_book_mapping)
        assert len(chunks) == 0

    def test_parse_scripture_file_multiple_chapters(self, tmp_path, sample_book_mapping):
        """Test parsing file with multiple chapters."""
        content = """GEN 1:1 First chapter, first verse.
GEN 1:2 First chapter, second verse.
GEN 2:1 Second chapter, first verse.
GEN 2:2 Second chapter, second verse.
"""
        scripture_file = tmp_path / "test.txt"
        scripture_file.write_text(content)

        chunks = parse_scripture_file(scripture_file, sample_book_mapping)
        assert len(chunks) == 4

        # Check chapter numbers
        assert chunks[0].chapter == 1
        assert chunks[1].chapter == 1
        assert chunks[2].chapter == 2
        assert chunks[3].chapter == 2


class TestParseAllScriptureFiles:
    """Tests for parse_all_scripture_files function."""

    def test_parse_all_scripture_files_basic(self, temp_assets_directory, sample_book_mapping):
        """Test parsing all files in assets directory."""
        chunks = parse_all_scripture_files(temp_assets_directory, sample_book_mapping)

        # Should have verses from multiple files
        assert len(chunks) > 0

        # Should have chunks from different books
        books = {chunk.book for chunk in chunks}
        assert len(books) > 1

    def test_parse_all_scripture_files_includes_subdirectories(
        self, temp_assets_directory, sample_book_mapping
    ):
        """Test that all subdirectories are processed."""
        chunks = parse_all_scripture_files(temp_assets_directory, sample_book_mapping)

        # Should have chunks from bible and book-of-mormon subdirectories
        prefixes = {chunk.prefix for chunk in chunks}
        assert "RTH" in prefixes or "JON" in prefixes  # From bible
        assert "NE1" in prefixes  # From book-of-mormon

    def test_parse_all_scripture_files_accepts_path_object(
        self, temp_assets_directory, sample_book_mapping
    ):
        """Test that function accepts Path objects."""
        chunks = parse_all_scripture_files(Path(temp_assets_directory), sample_book_mapping)
        assert len(chunks) > 0

    def test_parse_all_scripture_files_accepts_string(
        self, temp_assets_directory, sample_book_mapping
    ):
        """Test that function accepts string paths."""
        chunks = parse_all_scripture_files(str(temp_assets_directory), sample_book_mapping)
        assert len(chunks) > 0

    def test_parse_all_scripture_files_warning_on_error(
        self, temp_assets_directory, sample_book_mapping, capsys
    ):
        """Test that errors are caught and warnings are printed."""
        # Create a file that will cause an error (e.g., binary file)
        bible_dir = temp_assets_directory / "bible"
        bad_file = bible_dir / "bad.txt"
        bad_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

        chunks = parse_all_scripture_files(temp_assets_directory, sample_book_mapping)

        # Should still return chunks from valid files
        assert len(chunks) > 0

        # Check that a warning was printed
        captured = capsys.readouterr()
        assert "Warning: Failed to parse" in captured.out

    def test_parse_all_scripture_files_empty_directory(self, tmp_path, sample_book_mapping):
        """Test parsing an empty assets directory."""
        empty_assets = tmp_path / "empty_assets"
        empty_assets.mkdir()

        chunks = parse_all_scripture_files(empty_assets, sample_book_mapping)
        assert len(chunks) == 0

    def test_parse_all_scripture_files_no_txt_files(self, tmp_path, sample_book_mapping):
        """Test directory with subdirectories but no .txt files."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        subdir = assets_dir / "bible"
        subdir.mkdir()
        # Create a non-.txt file
        (subdir / "readme.md").write_text("# README")

        chunks = parse_all_scripture_files(assets_dir, sample_book_mapping)
        assert len(chunks) == 0

    def test_parse_all_scripture_files_preserves_source_file(
        self, temp_assets_directory, sample_book_mapping
    ):
        """Test that source_file is correctly set for each chunk."""
        chunks = parse_all_scripture_files(temp_assets_directory, sample_book_mapping)

        # All chunks should have a source_file
        assert all(chunk.source_file for chunk in chunks)

        # Source files should be actual file paths
        for chunk in chunks:
            assert Path(chunk.source_file).suffix == ".txt"
