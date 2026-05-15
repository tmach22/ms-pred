"""
MassIVE Asynchronous FTP Fetcher
================================
Reads the v2 master index for required mzML files and orchestrates
parallel `wget` subprocesses to download them from MassIVE's FTP server.
Automatically routes to positive/negative subdirectories based on filename.
"""

import asyncio
import pandas as pd
from pathlib import Path
import logging
import re

# Configuration
MASTER_INDEX_PATH = "data/MSnLib/master_metadata_index_v2.parquet"
BASE_TARGET_DIR = Path("/data/nas-gpu/wang/tmach007/data/MSnLib/raw_scans")
MASSIVE_FTP_BASE = "ftp://massive.ucsd.edu/MSV000094528/peak"  # Converted mzMLs usually live in 'peak'
CONCURRENT_DOWNLOADS = 3  # Strict limit for MassIVE FTP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("massive_download.log"),
        logging.StreamHandler()
    ]
)

async def download_file(semaphore: asyncio.Semaphore, filename: str):
    """Executes a wget subprocess for a single file."""

    # Routing logic (matching the bash script)
    if re.search(r'(pos|positive)', filename, re.IGNORECASE):
        dest_dir = BASE_TARGET_DIR / "positive"
    elif re.search(r'(neg|negative)', filename, re.IGNORECASE):
        dest_dir = BASE_TARGET_DIR / "negative"
    else:
        dest_dir = BASE_TARGET_DIR / "unclassified"

    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / filename

    # We rely on wget's built-in -c (continue) flag.
    # If the file exists and is fully downloaded, wget -c will exit instantly with code 0.
    ftp_url = f"{MASSIVE_FTP_BASE}/{filename}"

    async with semaphore:
        await asyncio.sleep(1)  # stagger connection handshakes
        # wget arguments:
        # -c (continue)
        # -q (quiet, we don't need the progress bar in the log for 13k files)
        # -O (output document)
        # -t 5 (retry up to 5 times on network errors)
        process = await asyncio.create_subprocess_exec(
            'wget', '-c', '-q', '-t', '5', '-O', str(output_path), ftp_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        if process.returncode == 0:
            logging.info(f"✅ Success/Verified: {filename}")
            return True
        else:
            logging.error(f"❌ Failed: {filename} (Return code: {process.returncode})")
            # Clean up the 0-byte or corrupted stub if it failed completely
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            return False

async def main():
    logging.info(f"Loading master index from {MASTER_INDEX_PATH}...")
    df = pd.read_parquet(MASTER_INDEX_PATH, columns=['mzml_file'])
    unique_files = df['mzml_file'].unique()
    logging.info(f"Found {len(unique_files)} required .mzML files.")

    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

    tasks = [download_file(semaphore, filename) for filename in unique_files]
    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r)
    logging.info(f"--- Download Complete ---")
    logging.info(f"Successful transfers: {successful} / {len(unique_files)}")

if __name__ == "__main__":
    asyncio.run(main())
