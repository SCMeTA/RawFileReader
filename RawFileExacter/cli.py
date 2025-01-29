import click

from pathlib import Path
# multi processing
import threading
import os
import sys
import logging

from .reader import RawFileReader

error_list = []


def convert_raw_to_mzml(input_path: str, output_path: str, include_ms2: bool = False, filter_threshold: int | None = None):
    global error_list
    try:
        raw_file_reader = RawFileReader(input_path)
        raw_file_reader.to_mzml(output_path, include_ms2, filter_threshold)
    except Exception as e:
        file_name = Path(input_path).name
        error_list.append(file_name)
        logging.error(f"Error converting {file_name}")
        # remove the mzml file
        os.remove(output_path)
        with open('error.log', 'a') as f:
            f.write(f"Error converting {file_name}: {e}\n")


def convert_folder_to_mzml(input_folder: str, output_folder: str, include_ms2: bool = False, filter_threshold: int | None = None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    raw_files = list(Path(input_folder).rglob('*.raw'))
    output_files = [Path(output_folder) / f'{raw_file.stem}.mzML' for raw_file in raw_files]
    # RawFileReader(file_path).write_mzml(output_path, include_ms2, filter_threshold)
    threads = []
    max_threads = 8
    semaphore = threading.Semaphore(max_threads)

    def worker(input_path, output_path, include_ms2, filter_threshold):
        with semaphore:  # Acquire the semaphore
            convert_raw_to_mzml(input_path, output_path, include_ms2, filter_threshold)
        # Semaphore is automatically released at the end of this block
    threads = []
    for input_path, output_path in zip(raw_files, output_files):
        thread = threading.Thread(target=worker,
                                  args=(str(input_path), str(output_path), include_ms2, filter_threshold))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    logging.info("Conversion complete")


@click.command(name='convert folder')
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--include-ms2', is_flag=True, help='Include MS2 spectra in the mzML file')
@click.option('--filter-threshold', type=int, help='Filter out peaks with intensity below this threshold')
def cli(input_folder, output_folder, include_ms2, filter_threshold):
    convert_folder_to_mzml(input_folder, output_folder, include_ms2, filter_threshold)