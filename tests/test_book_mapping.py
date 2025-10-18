"""Tests for book mapping functionality."""

from pathlib import Path

import pytest

from scripture_rag.book_mapping import get_default_mapping, load_book_mapping


class TestLoadBookMapping:
    """Tests for load_book_mapping function."""

    def test_load_book_mapping_basic(self, temp_contents_file, sample_book_mapping):
        """Test loading a valid Contents.txt file."""
        mapping = load_book_mapping(temp_contents_file)

        # Verify expected mappings are present
        assert mapping["GEN"] == "Genesis"
        assert mapping["EXO"] == "Exodus"
        assert mapping["RTH"] == "Ruth"
        assert mapping["JON"] == "Jonah"

    def test_load_book_mapping_with_numbers(self, temp_contents_file):
        """Test loading book names with numbers."""
        mapping = load_book_mapping(temp_contents_file)
        assert mapping["SA1"] == "1-Samuel"

    def test_load_book_mapping_with_ampersand(self, temp_contents_file):
        """Test loading book names with ampersands."""
        mapping = load_book_mapping(temp_contents_file)
        assert mapping["D&C"] == "Doctrine-and-Covenants"

    def test_load_book_mapping_accepts_path_object(self, tmp_path, sample_contents_txt):
        """Test that function accepts Path objects."""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text(sample_contents_txt)

        # Pass as Path object
        mapping = load_book_mapping(Path(contents_file))
        assert "RTH" in mapping

    def test_load_book_mapping_accepts_string(self, tmp_path, sample_contents_txt):
        """Test that function accepts string paths."""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text(sample_contents_txt)

        # Pass as string
        mapping = load_book_mapping(str(contents_file))
        assert "RTH" in mapping

    def test_load_book_mapping_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        non_existent_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_book_mapping(non_existent_file)

        assert "Contents file not found" in str(exc_info.value)

    def test_load_book_mapping_empty_lines_ignored(self, tmp_path):
        """Test that empty lines are ignored."""
        contents = """Genesis   . . . . . . . . . . . . . . . . .   GEN

Ruth  . . . . . . . . . . . . . . . . . . .   RTH

"""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text(contents)

        mapping = load_book_mapping(contents_file)
        assert mapping["GEN"] == "Genesis"
        assert mapping["RTH"] == "Ruth"
        assert len(mapping) == 2

    def test_load_book_mapping_malformed_lines_ignored(self, tmp_path):
        """Test that malformed lines are skipped."""
        contents = """Genesis   . . . . . . . . . . . . . . . . .   GEN
This is a malformed line without proper format
Ruth  . . . . . . . . . . . . . . . . . . .   RTH
Another bad line
"""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text(contents)

        mapping = load_book_mapping(contents_file)
        assert mapping["GEN"] == "Genesis"
        assert mapping["RTH"] == "Ruth"
        # Should only have the two valid mappings
        assert len(mapping) == 2

    def test_load_book_mapping_header_lines_ignored(self, tmp_path):
        """Test that header lines are ignored."""
        contents = """               TABLE OF CONTENTS I
             In order of appearance

                      BIBLE

Genesis   . . . . . . . . . . . . . . . . .   GEN
"""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text(contents)

        mapping = load_book_mapping(contents_file)
        # Should only have Genesis, not the header lines
        assert len(mapping) == 1
        assert mapping["GEN"] == "Genesis"

    def test_load_book_mapping_empty_file(self, tmp_path):
        """Test loading an empty Contents.txt file."""
        contents_file = tmp_path / "Contents.txt"
        contents_file.write_text("")

        mapping = load_book_mapping(contents_file)
        assert mapping == {}


class TestGetDefaultMapping:
    """Tests for get_default_mapping function."""

    def test_get_default_mapping_returns_dict(self):
        """Test that get_default_mapping returns a dictionary."""
        mapping = get_default_mapping()
        assert isinstance(mapping, dict)

    def test_get_default_mapping_has_common_books(self):
        """Test that default mapping contains expected common books."""
        mapping = get_default_mapping()

        # Check for some common books that should be in the standard works
        expected_books = ["GEN", "EXO", "MAT", "ROM", "NE1", "ALM"]

        # At least some of these should be present (depending on actual Contents.txt)
        found_books = [book for book in expected_books if book in mapping]
        assert len(found_books) > 0, "Default mapping should contain some common books"

    def test_get_default_mapping_consistent(self):
        """Test that get_default_mapping returns consistent results."""
        mapping1 = get_default_mapping()
        mapping2 = get_default_mapping()

        assert mapping1 == mapping2
