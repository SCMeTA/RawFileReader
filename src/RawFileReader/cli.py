import click
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import logging
from threading import Lock
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .reader import RawFileReader

# Thread-safe error list with lock
_error_lock = Lock()
_error_list = []


def _record_error(file_name: str, error: Exception):
    """Thread-safe error recording."""
    with _error_lock:
        _error_list.append(file_name)
    logging.error(f"Error converting {file_name}")
    with open('error.log', 'a') as f:
        f.write(f"Error converting {file_name}: {error}\n")


def convert_raw_to_mzml(input_path: str, output_path: str, include_ms2: bool = False, filter_threshold: int | None = None) -> bool:
    """Convert a single RAW file to mzML.

    Returns True on success, False on failure.
    """
    try:
        raw_file_reader = RawFileReader(input_path)
        raw_file_reader.to_mzml(output_path, include_ms2, filter_threshold)
        return True
    except Exception as e:
        file_name = Path(input_path).name
        _record_error(file_name, e)
        # Remove partial mzml file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False


def convert_folder_to_mzml(
    input_folder: str,
    output_folder: str,
    include_ms2: bool = False,
    filter_threshold: int | None = None,
    include_blank: bool = False,
    max_workers: int | None = None
):
    """Convert all RAW files in a folder to mzML using thread pool.

    Args:
        input_folder: Path to folder containing RAW files
        output_folder: Path to output folder for mzML files
        include_ms2: Include MS2 spectra in output
        filter_threshold: Filter peaks below this intensity
        include_blank: Include files with 'blank' in name
        max_workers: Maximum number of parallel workers (default: min(8, cpu_count))
    """
    global _error_list
    _error_list = []  # Reset error list

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_files = list(Path(input_folder).rglob('*.raw'))
    if not include_blank:
        raw_files = [f for f in raw_files if "blank" not in f.stem.lower()]

    if not raw_files:
        logging.warning(f"No RAW files found in {input_folder}")
        return

    output_files = [Path(output_folder) / f'{f.stem}.mzML' for f in raw_files]

    # Default to min(8, cpu_count) workers for optimal performance
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    logging.info(f"Converting {len(raw_files)} files with {max_workers} workers")

    # Use ThreadPoolExecutor for efficient thread management
    # This uses a work queue instead of creating all threads upfront
    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            future_to_file = {
                executor.submit(
                    convert_raw_to_mzml,
                    str(input_path),
                    str(output_path),
                    include_ms2,
                    filter_threshold
                ): input_path.name
                for input_path, output_path in zip(raw_files, output_files)
            }

            # Track progress with tqdm
            with tqdm(total=len(raw_files), desc="Converting files") as pbar:
                for future in as_completed(future_to_file):
                    file_name = future_to_file[future]
                    try:
                        success = future.result()
                        if not success:
                            logging.debug(f"Failed: {file_name}")
                    except Exception as e:
                        logging.error(f"Unexpected error with {file_name}: {e}")
                    pbar.update(1)

    # Report results
    success_count = len(raw_files) - len(_error_list)
    logging.info(f"Conversion complete: {success_count}/{len(raw_files)} files successful")
    if _error_list:
        logging.warning(f"Failed files: {', '.join(_error_list)}")


@click.command(name='convert folder')
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--include-ms2', is_flag=True, help='Include MS2 spectra in the mzML file')
@click.option('--filter-threshold', type=int, help='Filter out peaks with intensity below this threshold')
@click.option('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
def cli(input_folder, output_folder, include_ms2, filter_threshold, workers):
    """Convert RAW files in INPUT_FOLDER to mzML format in OUTPUT_FOLDER."""
    convert_folder_to_mzml(input_folder, output_folder, include_ms2, filter_threshold, max_workers=workers)
