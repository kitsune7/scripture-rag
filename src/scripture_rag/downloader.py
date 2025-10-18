"""Download and prepare scripture assets from remote sources."""

import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# Disable SSL warnings since we're intentionally disabling verification for old sites
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Hard-coded URLs for scripture zip files
SCRIPTURE_URLS = {
    "bible": "https://ldsguy.tripod.com/Iron-rod/kjv-lds.zip",
    "book-of-mormon": "https://ldsguy.tripod.com/Iron-rod/bom.zip",
    "doctrine-and-covenants": "https://ldsguy.tripod.com/Iron-rod/dnc.zip",
    "pearl-of-great-price": "https://ldsguy.tripod.com/Iron-rod/pofgp.zip",
}

# Files to remove from each scripture directory
UNWANTED_FILES = ["00.index1", "00.index2", "00.Readme"]


def create_session_with_retries() -> requests.Session:
    """
    Create a requests session with retry logic and proper headers.

    Returns:
        Configured requests session with retries and SSL handling
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET", "HEAD"],  # Only retry safe methods
    )

    # Mount the retry strategy to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set a user-agent to avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    return session


def download_file(url: str, dest_path: Path, session: requests.Session | None = None) -> None:
    """
    Download a file from a URL to a destination path.

    Args:
        url: URL to download from
        dest_path: Path to save the downloaded file
        session: Optional requests session to use (with retries configured)

    Raises:
        requests.RequestException: If download fails
    """
    print(f"Downloading {url}...")

    # Create session if not provided
    if session is None:
        session = create_session_with_retries()

    try:
        # Use streaming download for better performance and memory efficiency
        # Disable SSL verification for old sites with certificate issues
        response = session.get(url, timeout=60, stream=True, verify=False)
        response.raise_for_status()

        # Write file in chunks
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"  Downloaded to {dest_path}")

    except requests.exceptions.SSLError as e:
        print(f"  SSL Error: {e}")
        print(f"  Retrying without SSL verification...")
        # Already disabled SSL verification above, but this catches any other SSL issues
        raise

    except requests.exceptions.RequestException as e:
        print(f"  Download failed: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extract a zip file to a directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to

    Returns:
        Path to the extracted directory

    Raises:
        zipfile.BadZipFile: If the zip file is corrupt
    """
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Find the extracted directory (should be only one)
    extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
    if len(extracted_dirs) != 1:
        raise ValueError(f"Expected 1 directory in zip, found {len(extracted_dirs)}")

    return extracted_dirs[0]


def add_txt_extension(directory: Path) -> None:
    """
    Add .txt extension to all files in directory that don't have .txt extension.

    Args:
        directory: Directory containing files to rename
    """
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix != ".txt":
            # Add .txt to the end of the filename, preserving any existing extension
            new_path = Path(str(file_path) + ".txt")
            file_path.rename(new_path)


def remove_unwanted_files(directory: Path) -> None:
    """
    Remove unwanted files from a directory.

    Args:
        directory: Directory to clean up
    """
    for filename in UNWANTED_FILES:
        file_path = directory / filename
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed {filename}")


def process_scripture_directory(source_dir: Path, target_name: str, assets_dir: Path) -> None:
    """
    Process a scripture directory: rename, clean up, and move to assets.

    Args:
        source_dir: Source directory with extracted files
        target_name: Target name for the scripture collection
        assets_dir: Assets directory where the collection should be placed
    """
    target_dir = assets_dir / target_name

    # Move directory to assets
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(source_dir), str(target_dir))
    print(f"  Moved to {target_dir}")

    # Remove unwanted files
    remove_unwanted_files(target_dir)

    # Add .txt extension to all files
    print(f"  Adding .txt extensions...")
    add_txt_extension(target_dir)


def ensure_assets_downloaded(assets_dir: str | Path | None = None) -> Path:
    """
    Ensure scripture assets are downloaded and prepared.

    This function checks if assets already exist. If not, it downloads all
    scripture zip files, extracts them, renames folders, removes unwanted files,
    and adds .txt extensions.

    Args:
        assets_dir: Path to assets directory. Defaults to ../../../assets relative to this file

    Returns:
        Path to the assets directory

    Raises:
        requests.RequestException: If download fails
        zipfile.BadZipFile: If any zip file is corrupt
        ValueError: If zip structure is unexpected
    """
    # Determine assets directory
    if assets_dir is None:
        this_file = Path(__file__)
        assets_dir = this_file.parent.parent.parent / "assets"
    else:
        assets_dir = Path(assets_dir)

    # Check if assets already exist
    if assets_dir.exists():
        # Check if it has expected subdirectories
        expected_dirs = ["bible", "book-of-mormon", "doctrine-and-covenants", "pearl-of-great-price"]
        if all((assets_dir / subdir).exists() for subdir in expected_dirs):
            print(f"Assets already exist at {assets_dir}, skipping download")
            return assets_dir
        else:
            print(f"Assets directory exists but is incomplete, re-downloading...")

    # Create assets directory
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading scripture assets to {assets_dir}...")

    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        contents_copied = False

        # Create a session with retries for all downloads
        session = create_session_with_retries()

        # Download and process each scripture collection
        for i, (scripture_name, url) in enumerate(SCRIPTURE_URLS.items()):
            try:
                # Add delay between downloads to avoid rate limiting (except for first download)
                if i > 0:
                    print(f"Waiting 2 seconds to avoid rate limiting...")
                    time.sleep(2)

                # Download zip file
                zip_path = temp_path / f"{scripture_name}.zip"
                download_file(url, zip_path, session=session)

                # Extract zip file
                extract_path = temp_path / scripture_name
                extract_path.mkdir()
                extracted_dir = extract_zip(zip_path, extract_path)

                # Copy Contents file (only once)
                if not contents_copied:
                    contents_file = extracted_dir / "00.Contents"
                    if contents_file.exists():
                        shutil.copy(contents_file, assets_dir / "Contents.txt")
                        print(f"  Copied Contents.txt to {assets_dir}")
                        contents_copied = True

                # Process the directory
                process_scripture_directory(extracted_dir, scripture_name, assets_dir)

                print(f"Successfully processed {scripture_name}")

            except Exception as e:
                print(f"Error processing {scripture_name}: {e}")
                # Clean up partial assets on error
                if assets_dir.exists():
                    shutil.rmtree(assets_dir)
                raise

    print(f"\nAll scripture assets downloaded and prepared successfully!")
    return assets_dir


if __name__ == "__main__":
    # Allow running as a standalone script
    ensure_assets_downloaded()
