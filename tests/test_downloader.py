"""Tests for the scripture downloader module."""

import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripture_rag.downloader import (
    SCRIPTURE_URLS,
    UNWANTED_FILES,
    add_txt_extension,
    download_file,
    ensure_assets_downloaded,
    extract_zip,
    process_scripture_directory,
    remove_unwanted_files,
)


class TestDownloadFile:
    """Tests for download_file function."""

    @patch("scripture_rag.downloader.create_session_with_retries")
    def test_download_file_success(self, mock_create_session, tmp_path):
        """Test successful file download."""
        # Create mock response with streaming support
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"test content"])

        # Create mock session
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_create_session.return_value = mock_session

        dest_path = tmp_path / "test.zip"
        download_file("https://example.com/test.zip", dest_path)

        assert dest_path.exists()
        assert dest_path.read_bytes() == b"test content"
        mock_session.get.assert_called_once_with(
            "https://example.com/test.zip", timeout=60, stream=True, verify=False
        )

    @patch("scripture_rag.downloader.create_session_with_retries")
    def test_download_file_http_error(self, mock_create_session, tmp_path):
        """Test download failure with HTTP error."""
        # Create mock session that raises HTTPError
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.HTTPError("404 Not Found")
        mock_create_session.return_value = mock_session

        dest_path = tmp_path / "test.zip"
        with pytest.raises(requests.HTTPError):
            download_file("https://example.com/test.zip", dest_path)


class TestExtractZip:
    """Tests for extract_zip function."""

    def test_extract_zip_success(self, tmp_path):
        """Test successful zip extraction."""
        # Create a test zip file
        zip_path = tmp_path / "test.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        # Create a zip with a single directory containing a file
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test_dir/file.txt", "test content")

        result = extract_zip(zip_path, extract_to)

        assert result.exists()
        assert result.is_dir()
        assert (result / "file.txt").exists()
        assert (result / "file.txt").read_text() == "test content"

    def test_extract_zip_multiple_dirs_error(self, tmp_path):
        """Test error when zip contains multiple directories."""
        zip_path = tmp_path / "test.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        # Create a zip with multiple top-level directories
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dir1/file.txt", "content1")
            zf.writestr("dir2/file.txt", "content2")

        with pytest.raises(ValueError, match="Expected 1 directory in zip"):
            extract_zip(zip_path, extract_to)

    def test_extract_zip_corrupt(self, tmp_path):
        """Test extraction of corrupt zip file."""
        zip_path = tmp_path / "corrupt.zip"
        zip_path.write_text("not a zip file")
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        with pytest.raises(zipfile.BadZipFile):
            extract_zip(zip_path, extract_to)


class TestAddTxtExtension:
    """Tests for add_txt_extension function."""

    def test_add_txt_extension_to_files_without_extension(self, tmp_path):
        """Test adding .txt extension to files without extensions."""
        # Create test files
        (tmp_path / "file1").write_text("content1")
        (tmp_path / "file2").write_text("content2")
        (tmp_path / "file3.txt").write_text("content3")  # Already has extension

        add_txt_extension(tmp_path)

        assert (tmp_path / "file1.txt").exists()
        assert (tmp_path / "file2.txt").exists()
        assert (tmp_path / "file3.txt").exists()
        assert not (tmp_path / "file1").exists()
        assert not (tmp_path / "file2").exists()

    def test_add_txt_extension_empty_directory(self, tmp_path):
        """Test adding extensions in empty directory."""
        add_txt_extension(tmp_path)
        # Should not raise any errors


class TestRemoveUnwantedFiles:
    """Tests for remove_unwanted_files function."""

    def test_remove_unwanted_files_success(self, tmp_path):
        """Test removing unwanted files."""
        # Create unwanted files
        for filename in UNWANTED_FILES:
            (tmp_path / filename).write_text("unwanted")

        # Create a wanted file
        (tmp_path / "wanted.txt").write_text("wanted")

        remove_unwanted_files(tmp_path)

        # Verify unwanted files are removed
        for filename in UNWANTED_FILES:
            assert not (tmp_path / filename).exists()

        # Verify wanted file still exists
        assert (tmp_path / "wanted.txt").exists()

    def test_remove_unwanted_files_when_not_present(self, tmp_path):
        """Test removing unwanted files when they don't exist."""
        remove_unwanted_files(tmp_path)
        # Should not raise any errors


class TestProcessScriptureDirectory:
    """Tests for process_scripture_directory function."""

    def test_process_scripture_directory_success(self, tmp_path):
        """Test processing a scripture directory."""
        # Create source directory with test files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file1").write_text("content1")
        (source_dir / "00.index1").write_text("unwanted")
        (source_dir / "wanted").write_text("content2")

        # Create assets directory
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        process_scripture_directory(source_dir, "test-scripture", assets_dir)

        target_dir = assets_dir / "test-scripture"
        assert target_dir.exists()
        assert (target_dir / "file1.txt").exists()
        assert (target_dir / "wanted.txt").exists()
        assert not (target_dir / "00.index1").exists()

    def test_process_scripture_directory_overwrites_existing(self, tmp_path):
        """Test that processing overwrites existing directory."""
        # Create source directory
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "new_file").write_text("new content")

        # Create assets directory with existing target
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        target_dir = assets_dir / "test-scripture"
        target_dir.mkdir()
        (target_dir / "old_file.txt").write_text("old content")

        process_scripture_directory(source_dir, "test-scripture", assets_dir)

        assert (target_dir / "new_file.txt").exists()
        assert not (target_dir / "old_file.txt").exists()


class TestEnsureAssetsDownloaded:
    """Tests for ensure_assets_downloaded function."""

    def test_ensure_assets_downloaded_skip_when_exists(self, tmp_path):
        """Test that download is skipped when assets already exist."""
        # Create complete assets structure
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "bible").mkdir()
        (assets_dir / "book-of-mormon").mkdir()
        (assets_dir / "doctrine-and-covenants").mkdir()
        (assets_dir / "pearl-of-great-price").mkdir()

        with patch("scripture_rag.downloader.download_file") as mock_download:
            result = ensure_assets_downloaded(assets_dir)

            assert result == assets_dir
            mock_download.assert_not_called()

    def test_ensure_assets_downloaded_incomplete_assets(self, tmp_path):
        """Test re-download when assets directory is incomplete."""
        # Create incomplete assets structure (missing some directories)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "bible").mkdir()

        with patch("scripture_rag.downloader.download_file") as mock_download:
            with patch("scripture_rag.downloader.extract_zip") as mock_extract:
                with patch("scripture_rag.downloader.shutil.copy") as mock_copy:
                    # Create mock extracted directories
                    for name in SCRIPTURE_URLS.keys():
                        mock_dir = tmp_path / "temp" / name / "extracted"
                        mock_dir.mkdir(parents=True)
                        (mock_dir / "00.Contents").write_text("contents")
                        (mock_dir / "file").write_text("content")

                    mock_extract.side_effect = [
                        tmp_path / "temp" / "bible" / "extracted",
                        tmp_path / "temp" / "book-of-mormon" / "extracted",
                        tmp_path / "temp" / "doctrine-and-covenants" / "extracted",
                        tmp_path / "temp" / "pearl-of-great-price" / "extracted",
                    ]

                    result = ensure_assets_downloaded(assets_dir)

                    assert mock_download.call_count == 4

    @patch("scripture_rag.downloader.download_file")
    @patch("scripture_rag.downloader.extract_zip")
    def test_ensure_assets_downloaded_full_download(
        self, mock_extract, mock_download, tmp_path
    ):
        """Test complete download and processing flow."""
        assets_dir = tmp_path / "assets"

        # Create mock extracted directories
        mock_extracted_dirs = {}
        for name in SCRIPTURE_URLS.keys():
            mock_dir = tmp_path / "mock_extracted" / name
            mock_dir.mkdir(parents=True)
            (mock_dir / "00.Contents").write_text("Contents file")
            (mock_dir / "00.index1").write_text("unwanted")
            (mock_dir / "file1").write_text("content1")
            (mock_dir / "file2").write_text("content2")
            mock_extracted_dirs[name] = mock_dir

        # Configure mock to return appropriate extracted directories
        mock_extract.side_effect = [
            mock_extracted_dirs["bible"],
            mock_extracted_dirs["book-of-mormon"],
            mock_extracted_dirs["doctrine-and-covenants"],
            mock_extracted_dirs["pearl-of-great-price"],
        ]

        result = ensure_assets_downloaded(assets_dir)

        # Verify downloads were called
        assert mock_download.call_count == 4

        # Verify all expected directories exist
        assert result == assets_dir
        assert (assets_dir / "bible").exists()
        assert (assets_dir / "book-of-mormon").exists()
        assert (assets_dir / "doctrine-and-covenants").exists()
        assert (assets_dir / "pearl-of-great-price").exists()

        # Verify Contents.txt was copied
        assert (assets_dir / "Contents.txt").exists()

        # Verify files were renamed with .txt extension
        assert (assets_dir / "bible" / "file1.txt").exists()
        assert (assets_dir / "bible" / "file2.txt").exists()

        # Verify unwanted files were removed
        assert not (assets_dir / "bible" / "00.index1").exists()

    @patch("scripture_rag.downloader.download_file")
    def test_ensure_assets_downloaded_cleanup_on_error(self, mock_download, tmp_path):
        """Test that partial assets are cleaned up on error."""
        assets_dir = tmp_path / "assets"

        # Make download fail
        mock_download.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException):
            ensure_assets_downloaded(assets_dir)

        # Verify assets directory was cleaned up
        assert not assets_dir.exists()

    def test_ensure_assets_downloaded_default_path(self):
        """Test that default assets path is used when none provided."""
        with patch("scripture_rag.downloader.download_file"):
            with patch("scripture_rag.downloader.extract_zip"):
                with patch("scripture_rag.downloader.Path.exists") as mock_exists:
                    # Mock that all expected directories exist
                    mock_exists.return_value = True

                    result = ensure_assets_downloaded()

                    # Should use default path
                    expected_path = Path(__file__).parent.parent / "src" / "scripture_rag"
                    expected_assets = expected_path.parent.parent / "assets"
                    assert result.name == "assets"


class TestScriptureUrls:
    """Tests for SCRIPTURE_URLS constant."""

    def test_scripture_urls_defined(self):
        """Test that all required scripture URLs are defined."""
        assert "bible" in SCRIPTURE_URLS
        assert "book-of-mormon" in SCRIPTURE_URLS
        assert "doctrine-and-covenants" in SCRIPTURE_URLS
        assert "pearl-of-great-price" in SCRIPTURE_URLS

    def test_scripture_urls_are_valid(self):
        """Test that all URLs are properly formatted."""
        for name, url in SCRIPTURE_URLS.items():
            assert url.startswith("https://") or url.startswith("http://")
            assert url.endswith(".zip")


class TestUnwantedFiles:
    """Tests for UNWANTED_FILES constant."""

    def test_unwanted_files_defined(self):
        """Test that unwanted files list is defined."""
        assert "00.index1" in UNWANTED_FILES
        assert "00.index2" in UNWANTED_FILES
        assert "00.Readme" in UNWANTED_FILES
