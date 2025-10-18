"""Shared pytest fixtures for scripture-rag tests."""

from pathlib import Path

import pytest

from scripture_rag.parser import ScriptureChunk


@pytest.fixture
def sample_scripture_content():
    """Sample scripture text content for testing."""
    return """RTH 1:0 Ruth and Naomi

RTH 1:1 Now it came to pass in the days when the judges ruled, that there was a famine in the land. And a certain man of Bethlehemjudah went to sojourn in the country of Moab, he, and his wife, and his two sons.

RTH 1:2 And the name of the man was Elimelech, and the name of his wife Naomi, and the name of his two sons Mahlon and Chilion, Ephrathites of Bethlehemjudah. And they came into the country of Moab, and continued there.

RTH 1:3 And Elimelech Naomi's husband died; and she was left, and her two sons.

JON 1:0 Jonah Sent to Nineveh

JON 1:1 Now the word of the LORD came unto Jonah the son of Amittai, saying,

JON 1:2 Arise, go to Nineveh, that great city, and cry against it; for their wickedness is come up before me.
"""


@pytest.fixture
def sample_contents_txt():
    """Sample Contents.txt content for testing."""
    return """               TABLE OF CONTENTS I
             In order of appearance

                      BIBLE

Preface   . . . . . . . . . . . . . . . . .   PRE
Genesis   . . . . . . . . . . . . . . . . .   GEN
Exodus    . . . . . . . . . . . . . . . . .   EXO
Ruth  . . . . . . . . . . . . . . . . . . .   RTH
1-Samuel  . . . . . . . . . . . . . . . . .   SA1
Doctrine-and-Covenants  . . . . . . . . . .   D&C
Jonah . . . . . . . . . . . . . . . . . . .   JON
"""


@pytest.fixture
def temp_scripture_file(tmp_path, sample_scripture_content):
    """Create a temporary scripture file for testing."""
    scripture_file = tmp_path / "test_scripture.txt"
    scripture_file.write_text(sample_scripture_content)
    return scripture_file


@pytest.fixture
def temp_contents_file(tmp_path, sample_contents_txt):
    """Create a temporary Contents.txt file for testing."""
    contents_file = tmp_path / "Contents.txt"
    contents_file.write_text(sample_contents_txt)
    return contents_file


@pytest.fixture
def temp_assets_directory(tmp_path, sample_scripture_content):
    """Create a temporary assets directory structure for testing."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    # Create Contents.txt
    contents = """Genesis   . . . . . . . . . . . . . . . . .   GEN
Ruth  . . . . . . . . . . . . . . . . . . .   RTH
Jonah . . . . . . . . . . . . . . . . . . .   JON
"""
    (assets_dir / "Contents.txt").write_text(contents)

    # Create bible subdirectory with files
    bible_dir = assets_dir / "bible"
    bible_dir.mkdir()
    (bible_dir / "08.ruth.txt").write_text(sample_scripture_content)
    (bible_dir / "32.jonah.txt").write_text(
        "JON 1:1 Now the word of the LORD came unto Jonah the son of Amittai, saying,\n"
    )

    # Create book-of-mormon subdirectory with a file
    bom_dir = assets_dir / "book-of-mormon"
    bom_dir.mkdir()
    (bom_dir / "01.1-nephi.txt").write_text(
        "NE1 1:1 I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days.\n"
    )

    return assets_dir


@pytest.fixture
def sample_book_mapping():
    """Sample book abbreviation to full name mapping."""
    return {
        "GEN": "Genesis",
        "EXO": "Exodus",
        "RTH": "Ruth",
        "SA1": "1-Samuel",
        "D&C": "Doctrine-and-Covenants",
        "JON": "Jonah",
        "NE1": "1-Nephi",
    }


@pytest.fixture
def sample_chunks():
    """Sample ScriptureChunk objects for testing."""
    return [
        ScriptureChunk(
            text="Now it came to pass in the days when the judges ruled, that there was a famine in the land.",
            book="Ruth",
            prefix="RTH",
            chapter=1,
            verse=1,
            reference="Ruth 1:1",
            section_heading="Ruth and Naomi",
            source_file="/path/to/ruth.txt",
        ),
        ScriptureChunk(
            text="And the name of the man was Elimelech, and the name of his wife Naomi.",
            book="Ruth",
            prefix="RTH",
            chapter=1,
            verse=2,
            reference="Ruth 1:2",
            section_heading="Ruth and Naomi",
            source_file="/path/to/ruth.txt",
        ),
        ScriptureChunk(
            text="Now the word of the LORD came unto Jonah the son of Amittai, saying,",
            book="Jonah",
            prefix="JON",
            chapter=1,
            verse=1,
            reference="Jonah 1:1",
            section_heading="Jonah Sent to Nineveh",
            source_file="/path/to/jonah.txt",
        ),
    ]
