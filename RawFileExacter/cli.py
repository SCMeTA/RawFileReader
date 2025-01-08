import click

from pathlib import Path
# multi processing
from multiprocessing import Pool
import os
import sys
import logging

from .reader import RawFileReader


@click.command(name='convert file')
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--include-ms2', is_flag=True, help='Include MS2 spectra in the mzML file')
@click.option('--filter-threshold', type=int, help='Filter out peaks with intensity below this threshold')
def convert_raw_to_mzml(input_path: str, output_path: str, include_ms2: bool = False, filter_threshold: int | None = None):
    raw_file_reader = RawFileReader(input_path)
    raw_file_reader.write_mzml(output_path, include_ms2, filter_threshold)


@click.command(name='convert folder')
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--include-ms2', is_flag=True, help='Include MS2 spectra in the mzML file')
@click.option('--filter-threshold', type=int, help='Filter out peaks with intensity below this threshold')
def convert_folder_to_mzml(input_folder: str, output_folder: str, include_ms2: bool = False, filter_threshold: int | None = None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    raw_files = list(Path(input_folder).rglob('*.raw'))
    output_files = [Path(output_folder) / f'{raw_file.stem}.mzML' for raw_file in raw_files]
    with Pool() as pool:
        pool.starmap(RawFileReader, zip(raw_files, output_files, [include_ms2]*len(raw_files), [filter_threshold]*len(raw_files)) )
    logging.info('Conversion complete')
