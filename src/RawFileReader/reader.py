from __future__ import annotations

from pythonnet import load

load("coreclr")

import clr
import sys
import numpy as np
import pandas as pd
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from pathlib import Path

# get absolute path of the current file
import os

logger = logging.getLogger(__name__)

current_file_path = Path(os.path.abspath(__file__))
# lib_path is a folder worked both with windows and linux
lib_path = current_file_path.parent / "lib"

sys.path.append(str(lib_path))

clr.AddReference('ThermoFisher.CommonCore.Data')
clr.AddReference('ThermoFisher.CommonCore.RawFileReader')
clr.AddReference('ThermoFisher.CommonCore.BackgroundSubtraction')
clr.AddReference('ThermoFisher.CommonCore.MassPrecisionEstimator')
clr.AddReference('ParallelRawFileReader')

from System import *
from System.Collections.Generic import *

from ThermoFisher.CommonCore.Data import ToleranceUnits
from ThermoFisher.CommonCore.Data import Extensions
from ThermoFisher.CommonCore.Data.Business import ChromatogramSignal, ChromatogramTraceSettings, DataUnits, Device, GenericDataTypes, SampleType, Scan, TraceType, MassOptions, Range
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IChromatogramSettings, IScanEventBase, IScanFilter, RawFileClassification
from ThermoFisher.CommonCore.MassPrecisionEstimator import PrecisionEstimate
from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
from ThermoFisher.CommonCore.Data.Business import RawFileReaderFactory
from ParallelRawFileReader import ParallelReader as CSharpParallelReader
from ParallelRawFileReader import MultiFileReader as CSharpMultiFileReader

# Import for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Optional Polars import for faster DataFrame construction
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

logger = logging.getLogger(__name__)

logger.info("Successfully loaded ThermoFisher.CommonCore.RawFileReader")


def DotNetArrayToNPArray(arr, dtype):
    """Convert .NET array to NumPy array efficiently."""
    if arr is None:
        return np.array([], dtype=dtype)
    # Use np.fromiter for faster conversion - avoids intermediate Python list
    # For large arrays this is significantly faster than np.array(list(arr))
    length = arr.Length if hasattr(arr, 'Length') else len(arr)
    if length == 0:
        return np.array([], dtype=dtype)
    return np.fromiter(arr, dtype=dtype, count=length)


def _get_spectrum_from_accessor(raw_file, scan_number: int, include_ms2: bool = False) -> tuple | None:
    """Extract spectrum data from a raw file accessor (thread-safe).

    This function is designed to work with thread accessors created from
    RawFileReaderFactory.CreateThreadManager() for parallel scan reading.
    """
    scan_statistics = raw_file.GetScanStatsForScanNumber(scan_number)
    scanFilter = IScanFilter(raw_file.GetFilterForScanNumber(scan_number))
    ms_order = scanFilter.MSOrder
    ms_order = 1 if ms_order == MSOrderType.Ms else 2
    polarity = "positive scan" if str(scanFilter.Polarity) == "Positive" else "negative scan"
    retention_time = raw_file.RetentionTimeFromScanNumber(scan_number)

    if not include_ms2 and ms_order == 2:
        return None

    if scan_statistics.IsCentroidScan:
        centroid_scan = raw_file.GetCentroidStream(scan_number, False)
        masses = DotNetArrayToNPArray(centroid_scan.Masses, float)
        intensities = DotNetArrayToNPArray(centroid_scan.Intensities, float)
        is_centroid = True
    else:
        segmented_scan = raw_file.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
        masses = DotNetArrayToNPArray(segmented_scan.Positions, float)
        intensities = DotNetArrayToNPArray(segmented_scan.Intensities, float)
        is_centroid = False

    precursor = None
    if ms_order > 1:
        scan_event = IScanEventBase(raw_file.GetScanEventForScanNumber(scan_number))
        if scan_event is not None:
            reaction = scan_event.GetReaction(0)
            isolation_width = float(reaction.IsolationWidth) if reaction.IsolationWidth is not None else None
            collision_energy = reaction.CollisionEnergy
            collision_energy = float(collision_energy) if collision_energy is not None else None
            if collision_energy is not None and np.isnan(collision_energy):
                collision_energy = None
            precursor = {
                "mz": float(reaction.PrecursorMass),
                "isolation_width": isolation_width,
                "collision_energy": collision_energy,
            }

    return retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor


class RawFileNotOpenError(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

    def __str__(self):
        return f"Cannot read RAW file: {self.errors}"


class RawFileReader:
    def __init__(self, file_path: str | Path):
        self.file_path: str | Path = file_path
        self.file_name = Path(file_path).stem
        self.rawFile = self.__open_raw_file()
        self.scan_range: list = self.__get_scan_number()
        self.instrument_info: dict = self.__get_instrument_info()
        self.max_retention_time: float = self.rawFile.RetentionTimeFromScanNumber(self.scan_range[1])

    def __open_raw_file(self):
        raw_file = RawFileReaderAdapter.FileFactory(str(self.file_path))
        if raw_file.IsOpen:
            # logger.info(f"Successfully opened {self.file_path}")
            # print("Successfully open the file")
            try:
                raw_file.SelectInstrument(Device.MS, 1)
                return raw_file
            except Exception as e:
                raise RawFileNotOpenError(f"Failed open RAW file: {self.file_name}")

        else:
            logger.error(f"Failed to open {self.file_path}")
            raise RawFileNotOpenError(f"Failed to open {self.file_path}")


    def __get_scan_number(self):
        first_scan = self.rawFile.RunHeaderEx.FirstSpectrum
        last_scan = self.rawFile.RunHeaderEx.LastSpectrum
        # logger.info(f"First scan: {first_scan}, Last scan: {last_scan}")
        # print(f"First scan: {first_scan}, Last scan: {last_scan}")
        # Get retention time of the first and last scan
        first_rt = self.rawFile.RetentionTimeFromScanNumber(first_scan)
        last_rt = self.rawFile.RetentionTimeFromScanNumber(last_scan)
        # logger.info(f"First RT: {first_rt}, Last RT: {last_rt}")
        # print(f"First RT: {first_rt}, Last RT: {last_rt}")
        return [first_scan, last_scan]

    def __get_instrument_info(self):
        raw_file_version = self.rawFile.FileHeader.Revision
        number_of_instruments = self.rawFile.InstrumentCount
        instrument_data = self.rawFile.GetInstrumentData()
        instrument_name = instrument_data.Name
        instrument_model = instrument_data.Model
        instrument_serial_number = instrument_data.SerialNumber
        mass_resolution = self.rawFile.RunHeaderEx.MassResolution
        return {
            "raw_file_version": raw_file_version,
            "number_of_instruments": number_of_instruments,
            "instrument_name": instrument_name,
            "instrument_model": instrument_model,
            "instrument_serial_number": instrument_serial_number,
            "mass_resolution": mass_resolution
        }

    def get_spectrum(self, scan_number: int, include_ms2: bool = False) -> tuple | None:
        scan_statistics = self.rawFile.GetScanStatsForScanNumber(scan_number)
        scanFilter = IScanFilter(self.rawFile.GetFilterForScanNumber(scan_number))
        ms_order = scanFilter.MSOrder
        ms_order = 1 if ms_order == MSOrderType.Ms else 2
        polarity = "positive scan" if str(scanFilter.Polarity) == "Positive" else "negative scan"
        retention_time = self.rawFile.RetentionTimeFromScanNumber(scan_number)
        if not include_ms2:
            if ms_order == 2:
                return None
        if scan_statistics.IsCentroidScan:
            centroid_scan = self.rawFile.GetCentroidStream(scan_number, False)
            masses = DotNetArrayToNPArray(centroid_scan.Masses, float)
            intensities = DotNetArrayToNPArray(centroid_scan.Intensities, float)
            is_centroid = True
        else:
            segmented_scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
            masses = DotNetArrayToNPArray(segmented_scan.Positions, float)
            intensities = DotNetArrayToNPArray(segmented_scan.Intensities, float)
            is_centroid = False
        precursor = None
        if ms_order > 1:
            scan_event = IScanEventBase(self.rawFile.GetScanEventForScanNumber(scan_number))
            if scan_event is not None:
                reaction = scan_event.GetReaction(0)
                isolation_width = float(reaction.IsolationWidth) if reaction.IsolationWidth is not None else None
                collision_energy = reaction.CollisionEnergy
                collision_energy = float(collision_energy) if collision_energy is not None else None
                if collision_energy is not None and np.isnan(collision_energy):
                    collision_energy = None
                precursor = {
                    "mz": float(reaction.PrecursorMass),
                    "isolation_width": isolation_width,
                    "collision_energy": collision_energy,
                }
        return retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor

    @staticmethod
    def __intensity_filter(threshold: int, mz_array: np.array, intensity: np.array):
        # filter the intensity and also remove the mz values
        indices_to_keep = np.where(intensity > threshold)
        return mz_array[indices_to_keep], intensity[indices_to_keep]

    def __single_scan_to_np_array(self, scan_number: int, include_ms2: bool = False, filter_threshold: int | None = None) -> np.ndarray | None:
        __spec_data = self.get_spectrum(scan_number, include_ms2)
        if __spec_data is None:
            return None
        else:
            retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor = __spec_data
            polarity = -1 if polarity == "negative scan" else 1
            if filter_threshold:
                masses, intensities = self.__intensity_filter(filter_threshold, masses, intensities)
                masses = np.round(masses, 6)
                intensities = np.round(intensities, 2)
        return np.array([
            masses,
            intensities,
            precursor,
        ], dtype=object)

    def to_numpy(self, include_ms2: bool = False, filter_threshold: int | None = None) -> np.ndarray:
        with logging_redirect_tqdm():
            whole_spectrum = [
                spectrum for spectrum in (self.__single_scan_to_np_array(scan, include_ms2, filter_threshold) for scan in trange(self.scan_range[0], self.scan_range[1]))
                if spectrum is not None
            ]
        return np.array(whole_spectrum, dtype=object)

    def to_series(self, scan_number: int, include_ms2: bool = False, filter_threshold: int | None = None) -> pd.DataFrame | None:
        __spec_data = self.get_spectrum(scan_number, include_ms2)
        if __spec_data is None:
            return None
        else:
            retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor = __spec_data
            polarity = -1 if polarity == "negative scan" else 1
        if filter_threshold:
            masses, intensities = self.__intensity_filter(filter_threshold, masses, intensities)
            masses = np.round(masses, 6)
            intensities = np.round(intensities, 2)
        precursor_mz = np.nan
        precursor_charge = np.nan
        isolation_width = np.nan
        collision_energy = np.nan
        if precursor:
            precursor_mz = precursor["mz"] if precursor["mz"] is not None else np.nan
            # precursor_charge = precursor["charge"] if precursor["charge"] is not None else np.nan
            isolation_width = precursor["isolation_width"] if precursor["isolation_width"] is not None else np.nan
            collision_energy = precursor["collision_energy"] if precursor["collision_energy"] is not None else np.nan
        row_count = masses.shape[0]
        precursor_mz_col = np.full(row_count, precursor_mz)
        # precursor_charge_col = np.full(row_count, precursor_charge)
        isolation_width_col = np.full(row_count, isolation_width)
        collision_energy_col = np.full(row_count, collision_energy)

        return pd.DataFrame(
            {
                "Scan": scan_number,
                "RetentionTime": round(retention_time, 3),
                "MS Order": ms_order,
                "Mass": masses,
                "Intensity": intensities,
                "Polarity": polarity,
                "PrecursorMz": precursor_mz_col,
                # "PrecursorCharge": precursor_charge_col,
                "IsolationWidth": isolation_width_col,
                "CollisionEnergy": collision_energy_col,
            }
        )

    def to_mzml(self, output_path: str | Path, include_ms2: bool = False, filter_threshold: int | None = None) -> None:
        from psims.mzml import MzMLWriter

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with MzMLWriter(str(output_path)) as writer:
            writer.controlled_vocabularies()
            writer.file_description([
                "MS1 spectrum",
                "MSn spectrum",
            ])
            writer.software_list([
                {"id": "psims-writer", "version": "0.1.2", "params": ["python-psims"]}
            ])
            source = writer.Source(1, ["electrospray ionization", "electrospray inlet"])
            analyzer = writer.Analyzer(2, ["fourier transform ion cyclotron resonance mass spectrometer"])
            detector = writer.Detector(3, ["inductive detector"])
            config = writer.InstrumentConfiguration(
                id="IC1",
                component_list=[source, analyzer, detector],
                params=[self.instrument_info.get("instrument_model", "Orbitrap")]
            )
            writer.instrument_configuration_list([config])
            methods = [
                writer.ProcessingMethod(order=1, software_reference="psims-writer", params=["Conversion to mzML"])
            ]
            processing = writer.DataProcessing(methods, id="DP1")
            writer.data_processing_list([processing])

            with writer.run(id="run1", instrument_configuration="IC1"):
                scan_count = self.scan_range[1] - self.scan_range[0] + 1
                with writer.spectrum_list(count=scan_count), logging_redirect_tqdm():
                    for scan_number in trange(
                        self.scan_range[0],
                        self.scan_range[1],
                        desc=f"Converting {self.file_name}",
                        leave=False,
                    ):
                        results = self.get_spectrum(scan_number, include_ms2)
                        if results is None:
                            continue
                        retention_time, ms_order, mz_array, intensity_array, polarity, is_centroid, precursor = results
                        scan_id = f"scan={scan_number}"
                        if filter_threshold:
                            mz_array, intensity_array = self.__intensity_filter(filter_threshold, mz_array, intensity_array)
                            mz_array = np.round(mz_array, 5)
                            intensity_array = np.round(intensity_array, 2)
                        precursor_info = None
                        if precursor:
                            precursor_mz = precursor["mz"]
                            isolation_width = precursor.get("isolation_width") or 0.0
                            half_width = isolation_width / 2 if isolation_width else 0.0
                            isolation_window = [
                                precursor_mz - half_width,
                                precursor_mz,
                                precursor_mz + half_width,
                            ]
                            activation = []
                            if precursor.get("collision_energy") is not None:
                                activation.append({"collision energy": precursor["collision_energy"]})
                            precursor_info = {
                                "mz": precursor_mz,
                                # "charge": precursor.get("charge"),
                                "isolation_window": isolation_window,
                                "activation": activation,
                            }
                        writer.write_spectrum(
                            mz_array,
                            intensity_array,
                            id=scan_id,
                            scan_start_time=retention_time,
                            polarity=polarity,
                            centroided=is_centroid,
                            params=[
                                f"MS{ms_order} spectrum",
                                {"ms level": ms_order},
                                {"total ion current": np.sum(intensity_array)},
                            ],
                            precursor_information=precursor_info,
                        )

    def to_mzml_parallel(
        self,
        output_path: str | Path,
        include_ms2: bool = False,
        filter_threshold: int | None = None,
        max_workers: int | None = None
    ) -> None:
        """Convert RAW file to mzML using parallel scan reading.

        Uses the ThermoFisher ThreadManager API for lockless parallel access
        to scan data. Scans are read in parallel and written sequentially
        (mzML requires sequential writes).

        Args:
            output_path: Output mzML file path
            include_ms2: Include MS2 spectra in output
            filter_threshold: Filter peaks below this intensity
            max_workers: Number of parallel workers (default: CPU count)
        """
        from psims.mzml import MzMLWriter
        from tqdm import tqdm

        if max_workers is None:
            max_workers = min(8, multiprocessing.cpu_count())

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create thread manager for parallel access
        thread_manager = RawFileReaderFactory.CreateThreadManager(str(self.file_path))

        scan_numbers = list(range(self.scan_range[0], self.scan_range[1]))
        results_dict = {}

        def process_scan(scan_number: int):
            """Read a single scan using a thread accessor."""
            thread_accessor = thread_manager.CreateThreadAccessor()
            thread_accessor.SelectInstrument(Device.MS, 1)

            spec_data = _get_spectrum_from_accessor(thread_accessor, scan_number, include_ms2)
            if spec_data is None:
                return scan_number, None

            retention_time, ms_order, mz_array, intensity_array, polarity, is_centroid, precursor = spec_data

            if filter_threshold:
                indices_to_keep = np.where(intensity_array > filter_threshold)
                mz_array = mz_array[indices_to_keep]
                intensity_array = intensity_array[indices_to_keep]
                mz_array = np.round(mz_array, 5)
                intensity_array = np.round(intensity_array, 2)

            return scan_number, {
                "retention_time": retention_time,
                "ms_order": ms_order,
                "mz_array": mz_array,
                "intensity_array": intensity_array,
                "polarity": polarity,
                "is_centroid": is_centroid,
                "precursor": precursor,
            }

        # Phase 1: Read all scans in parallel
        logger.info(f"Reading scans in parallel with {max_workers} workers...")
        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_scan, scan): scan for scan in scan_numbers}

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Reading {self.file_name}", leave=False):
                    scan_number, result = future.result()
                    if result is not None:
                        results_dict[scan_number] = result

        # Dispose thread manager after reading
        thread_manager.Dispose()

        # Phase 2: Write to mzML sequentially (required by mzML format)
        logger.info("Writing mzML file...")
        with MzMLWriter(str(output_path)) as writer:
            writer.controlled_vocabularies()
            writer.file_description(["MS1 spectrum", "MSn spectrum"])
            writer.software_list([
                {"id": "psims-writer", "version": "0.1.2", "params": ["python-psims"]}
            ])
            source = writer.Source(1, ["electrospray ionization", "electrospray inlet"])
            analyzer = writer.Analyzer(2, ["fourier transform ion cyclotron resonance mass spectrometer"])
            detector = writer.Detector(3, ["inductive detector"])
            config = writer.InstrumentConfiguration(
                id="IC1",
                component_list=[source, analyzer, detector],
                params=[self.instrument_info.get("instrument_model", "Orbitrap")]
            )
            writer.instrument_configuration_list([config])
            methods = [
                writer.ProcessingMethod(order=1, software_reference="psims-writer", params=["Conversion to mzML"])
            ]
            processing = writer.DataProcessing(methods, id="DP1")
            writer.data_processing_list([processing])

            with writer.run(id="run1", instrument_configuration="IC1"):
                with writer.spectrum_list(count=len(results_dict)), logging_redirect_tqdm():
                    for scan_number in tqdm(sorted(results_dict.keys()),
                                            desc=f"Writing {self.file_name}", leave=False):
                        data = results_dict[scan_number]
                        scan_id = f"scan={scan_number}"

                        precursor_info = None
                        if data["precursor"]:
                            precursor = data["precursor"]
                            precursor_mz = precursor["mz"]
                            isolation_width = precursor.get("isolation_width") or 0.0
                            half_width = isolation_width / 2 if isolation_width else 0.0
                            isolation_window = [
                                precursor_mz - half_width,
                                precursor_mz,
                                precursor_mz + half_width,
                            ]
                            activation = []
                            if precursor.get("collision_energy") is not None:
                                activation.append({"collision energy": precursor["collision_energy"]})
                            precursor_info = {
                                "mz": precursor_mz,
                                "isolation_window": isolation_window,
                                "activation": activation,
                            }

                        writer.write_spectrum(
                            data["mz_array"],
                            data["intensity_array"],
                            id=scan_id,
                            scan_start_time=data["retention_time"],
                            polarity=data["polarity"],
                            centroided=data["is_centroid"],
                            params=[
                                f"MS{data['ms_order']} spectrum",
                                {"ms level": data["ms_order"]},
                                {"total ion current": np.sum(data["intensity_array"])},
                            ],
                            precursor_information=precursor_info,
                        )

    def to_dataframe(self, include_ms2: bool = False, filter_threshold: int | None = None) -> pd.DataFrame:
        """Convert all scans to a DataFrame efficiently.

        Uses list accumulation instead of repeated DataFrame concatenation
        for O(n) instead of O(n²) memory complexity.
        """
        # Pre-allocate lists for each column
        all_scans = []
        all_rts = []
        all_ms_orders = []
        all_masses = []
        all_intensities = []
        all_polarities = []
        all_precursor_mz = []
        all_isolation_width = []
        all_collision_energy = []

        with logging_redirect_tqdm():
            for scan_number in trange(self.scan_range[0], self.scan_range[1],
                                      desc=f"Reading {self.file_name}", leave=False):
                spec_data = self.get_spectrum(scan_number, include_ms2)
                if spec_data is None:
                    continue

                retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor = spec_data
                polarity_val = -1 if polarity == "negative scan" else 1

                if filter_threshold:
                    masses, intensities = self.__intensity_filter(filter_threshold, masses, intensities)
                    masses = np.round(masses, 6)
                    intensities = np.round(intensities, 2)

                # Extract precursor info
                precursor_mz = np.nan
                isolation_width = np.nan
                collision_energy = np.nan
                if precursor:
                    precursor_mz = precursor["mz"] if precursor["mz"] is not None else np.nan
                    isolation_width = precursor["isolation_width"] if precursor["isolation_width"] is not None else np.nan
                    collision_energy = precursor["collision_energy"] if precursor["collision_energy"] is not None else np.nan

                row_count = len(masses)
                if row_count == 0:
                    continue

                # Extend lists with data from this scan
                all_scans.extend([scan_number] * row_count)
                all_rts.extend([round(retention_time, 3)] * row_count)
                all_ms_orders.extend([ms_order] * row_count)
                all_masses.extend(masses)
                all_intensities.extend(intensities)
                all_polarities.extend([polarity_val] * row_count)
                all_precursor_mz.extend([precursor_mz] * row_count)
                all_isolation_width.extend([isolation_width] * row_count)
                all_collision_energy.extend([collision_energy] * row_count)

        # Create single DataFrame at the end - much faster than repeated concat
        return pd.DataFrame({
            "Scan": all_scans,
            "RetentionTime": all_rts,
            "MS Order": all_ms_orders,
            "Mass": all_masses,
            "Intensity": all_intensities,
            "Polarity": all_polarities,
            "PrecursorMz": all_precursor_mz,
            "IsolationWidth": all_isolation_width,
            "CollisionEnergy": all_collision_energy,
        })

    def to_dataframe_parallel(
        self,
        include_ms2: bool = False,
        filter_threshold: int | None = None,
        max_workers: int | None = None
    ) -> pd.DataFrame:
        """Convert all scans to a DataFrame using parallel processing.

        Uses the ThermoFisher ThreadManager API for lockless parallel access
        to scan data, providing significant speedup on multi-core systems.

        Args:
            include_ms2: Include MS2 spectra in output
            filter_threshold: Filter peaks below this intensity
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            DataFrame with all scan data
        """
        if max_workers is None:
            max_workers = min(8, multiprocessing.cpu_count())

        # Create thread manager for parallel access
        thread_manager = RawFileReaderFactory.CreateThreadManager(str(self.file_path))

        scan_numbers = list(range(self.scan_range[0], self.scan_range[1]))
        results_dict = {}

        def process_scan(scan_number: int):
            """Process a single scan using a thread accessor."""
            # Create thread accessor for this thread (lockless parallel access)
            thread_accessor = thread_manager.CreateThreadAccessor()
            thread_accessor.SelectInstrument(Device.MS, 1)

            spec_data = _get_spectrum_from_accessor(thread_accessor, scan_number, include_ms2)
            if spec_data is None:
                return scan_number, None

            retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor = spec_data
            polarity_val = -1 if polarity == "negative scan" else 1

            if filter_threshold:
                indices_to_keep = np.where(intensities > filter_threshold)
                masses = masses[indices_to_keep]
                intensities = intensities[indices_to_keep]
                masses = np.round(masses, 6)
                intensities = np.round(intensities, 2)

            if len(masses) == 0:
                return scan_number, None

            # Extract precursor info
            precursor_mz = np.nan
            isolation_width = np.nan
            collision_energy = np.nan
            if precursor:
                precursor_mz = precursor["mz"] if precursor["mz"] is not None else np.nan
                isolation_width = precursor["isolation_width"] if precursor["isolation_width"] is not None else np.nan
                collision_energy = precursor["collision_energy"] if precursor["collision_energy"] is not None else np.nan

            return scan_number, {
                "retention_time": round(retention_time, 3),
                "ms_order": ms_order,
                "masses": masses,
                "intensities": intensities,
                "polarity": polarity_val,
                "precursor_mz": precursor_mz,
                "isolation_width": isolation_width,
                "collision_energy": collision_energy,
            }

        # Process scans in parallel using ThreadPoolExecutor
        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_scan, scan): scan for scan in scan_numbers}

                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Reading {self.file_name} (parallel)", leave=False):
                    scan_number, result = future.result()
                    if result is not None:
                        results_dict[scan_number] = result

        # Dispose thread manager
        thread_manager.Dispose()

        # Build DataFrame from results (sorted by scan number for consistent order)
        all_scans = []
        all_rts = []
        all_ms_orders = []
        all_masses = []
        all_intensities = []
        all_polarities = []
        all_precursor_mz = []
        all_isolation_width = []
        all_collision_energy = []

        for scan_number in sorted(results_dict.keys()):
            data = results_dict[scan_number]
            row_count = len(data["masses"])

            all_scans.extend([scan_number] * row_count)
            all_rts.extend([data["retention_time"]] * row_count)
            all_ms_orders.extend([data["ms_order"]] * row_count)
            all_masses.extend(data["masses"])
            all_intensities.extend(data["intensities"])
            all_polarities.extend([data["polarity"]] * row_count)
            all_precursor_mz.extend([data["precursor_mz"]] * row_count)
            all_isolation_width.extend([data["isolation_width"]] * row_count)
            all_collision_energy.extend([data["collision_energy"]] * row_count)

        return pd.DataFrame({
            "Scan": all_scans,
            "RetentionTime": all_rts,
            "MS Order": all_ms_orders,
            "Mass": all_masses,
            "Intensity": all_intensities,
            "Polarity": all_polarities,
            "PrecursorMz": all_precursor_mz,
            "IsolationWidth": all_isolation_width,
            "CollisionEnergy": all_collision_energy,
        })

    def extract_eic(self, mz: float | list[float], _tolerance: float = 5, start_scan: int = -1, end_scan: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract Extracted Ion Chromatogram (EIC) for a given m/z value or list of m/z values.
        Args:
            mz: float | list of float, m/z value(s) to extract
            _tolerance: tolerance in ppm
            start_scan: start scan number, -1 for the first scan
            end_scan: end scan number, -1 for the last scan

        Returns:
            scans: np.ndarray, scan numbers
            rts: np.ndarray, retention times
            intensities: np.ndarray, intensities
        """
        # Read the MS data
        filterMs = "ms"

        # Create the chromatogram trace settings for TIC (Total Ion Chromatogram)
        traceSettings = ChromatogramTraceSettings(TraceType.MassRange)
        traceSettings.Filter = filterMs
        allSettings = []
        if isinstance(mz, float):
            mz = [mz]
        for m in mz:
            traceSettings = ChromatogramTraceSettings(TraceType.MassRange)
            traceSettings.Filter = "ms"
            traceSettings.MassRanges = [Range(m, m)]
            allSettings.append(traceSettings)


        # Open MS data
        self.rawFile.SelectInstrument(Device.MS, 1)


        tolerance = MassOptions()
        tolerance.Tolerance = _tolerance
        tolerance.ToleranceUnits = ToleranceUnits.ppm

        data = self.rawFile.GetChromatogramData(allSettings, start_scan, end_scan, tolerance)

        scans = DotNetArrayToNPArray(data.ScanNumbersArray[0], int)
        rts = DotNetArrayToNPArray(data.PositionsArray[0], float)

        # IntensitiesArray is a 2D .NET array - convert each row separately
        # Handle case where different m/z channels may have different lengths
        n_mz = data.IntensitiesArray.Length
        n_points = len(rts)

        # Pre-allocate array with correct shape
        intensities = np.zeros((n_points, n_mz), dtype=float)

        for i in range(n_mz):
            arr = DotNetArrayToNPArray(data.IntensitiesArray[i], float)
            # Handle length mismatch by taking min length
            length = min(len(arr), n_points)
            if length > 0:
                intensities[:length, i] = arr[:length]

        return scans, rts, intensities

    def extract_tic(self, start_scan: int = -1, end_scan: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract Total Ion Chromatogram (TIC) from the raw file.

        Args:
            start_scan: start scan number, -1 for the first scan
            end_scan: end scan number, -1 for the last scan

        Returns:
            scans: np.ndarray, scan numbers
            rts: np.ndarray, retention times
            intensities: np.ndarray, total ion current intensities
        """
        # Create the chromatogram trace settings for TIC
        traceSettings = ChromatogramTraceSettings(TraceType.TIC)
        traceSettings.Filter = "ms"

        # Open MS data
        self.rawFile.SelectInstrument(Device.MS, 1)

        # Get the chromatogram data
        data = self.rawFile.GetChromatogramData([traceSettings], start_scan, end_scan)

        scans = DotNetArrayToNPArray(data.ScanNumbersArray[0], int)
        rts = DotNetArrayToNPArray(data.PositionsArray[0], float)
        intensities = DotNetArrayToNPArray(data.IntensitiesArray[0], float)

        return scans, rts, intensities

    def to_dataframe_fast(
        self,
        include_ms2: bool = False,
        filter_threshold: float = 0,
        max_workers: int = 0,
        use_polars: bool = True
    ) -> pd.DataFrame:
        """Convert all scans to DataFrame using C# native parallel processing.

        This is the fastest method for reading scan data, using:
        - Native C# parallel processing (bypasses Python GIL)
        - ThermoFisher ThreadManager for lockless parallel access
        - Polars for fast DataFrame construction (if available)

        Args:
            include_ms2: Include MS2 spectra in output
            filter_threshold: Filter peaks below this intensity (0 = no filter)
            max_workers: Number of parallel workers (0 = auto, max 8)
            use_polars: Use Polars for DataFrame construction if available

        Returns:
            pandas DataFrame with all scan data
        """
        # Use C# parallel reader
        csharp_reader = CSharpParallelReader(str(self.file_path))
        result = csharp_reader.ReadAllScansParallel(
            includeMs2=include_ms2,
            filterThreshold=filter_threshold,
            maxWorkers=max_workers if max_workers > 0 else 0
        )
        csharp_reader.Dispose()

        # Convert to DataFrame using Polars or Pandas
        if use_polars and HAS_POLARS:
            return self._bulk_result_to_dataframe_polars(result)
        else:
            return self._bulk_result_to_dataframe_pandas(result)

    def _bulk_result_to_dataframe_polars(self, result) -> pd.DataFrame:
        """Convert C# BulkScanResult to pandas DataFrame using Polars for speed."""
        # Convert .NET arrays directly to numpy (faster than list())
        total_points = int(result.TotalDataPoints)
        num_scans = int(result.TotalScans)

        # Get flattened data arrays
        all_masses = np.fromiter(result.AllMasses, dtype=np.float64, count=total_points)
        all_intensities = np.fromiter(result.AllIntensities, dtype=np.float64, count=total_points)

        # Get per-scan metadata
        scan_numbers = np.fromiter(result.ScanNumbers, dtype=np.int32, count=num_scans)
        retention_times = np.fromiter(result.RetentionTimes, dtype=np.float64, count=num_scans)
        ms_orders = np.fromiter(result.MsOrders, dtype=np.int32, count=num_scans)
        polarities = np.fromiter(result.Polarities, dtype=np.int32, count=num_scans)
        precursor_mzs = np.fromiter(result.PrecursorMzs, dtype=np.float64, count=num_scans)
        isolation_widths = np.fromiter(result.IsolationWidths, dtype=np.float64, count=num_scans)
        collision_energies = np.fromiter(result.CollisionEnergies, dtype=np.float64, count=num_scans)
        scan_start_indices = np.fromiter(result.ScanStartIndices, dtype=np.int32, count=num_scans)
        scan_lengths = np.fromiter(result.ScanLengths, dtype=np.int32, count=num_scans)

        # Use numpy repeat for fast expansion of per-scan values
        df_scan_numbers = np.repeat(scan_numbers, scan_lengths)
        df_retention_times = np.repeat(retention_times, scan_lengths)
        df_ms_orders = np.repeat(ms_orders, scan_lengths)
        df_polarities = np.repeat(polarities, scan_lengths)
        df_precursor_mzs = np.repeat(precursor_mzs, scan_lengths)
        df_isolation_widths = np.repeat(isolation_widths, scan_lengths)
        df_collision_energies = np.repeat(collision_energies, scan_lengths)

        # Build Polars DataFrame (much faster than pandas for large data)
        df_polars = pl.DataFrame({
            "Scan": df_scan_numbers,
            "RetentionTime": df_retention_times,
            "MS Order": df_ms_orders,
            "Mass": all_masses,
            "Intensity": all_intensities,
            "Polarity": df_polarities,
            "PrecursorMz": df_precursor_mzs,
            "IsolationWidth": df_isolation_widths,
            "CollisionEnergy": df_collision_energies,
        })

        # Convert to pandas (zero-copy where possible)
        return df_polars.to_pandas()

    def _bulk_result_to_dataframe_pandas(self, result) -> pd.DataFrame:
        """Convert C# BulkScanResult to pandas DataFrame directly."""
        # Convert .NET arrays to numpy
        total_points = int(result.TotalDataPoints)
        num_scans = int(result.TotalScans)

        all_masses = np.fromiter(result.AllMasses, dtype=np.float64, count=total_points)
        all_intensities = np.fromiter(result.AllIntensities, dtype=np.float64, count=total_points)

        scan_numbers = np.fromiter(result.ScanNumbers, dtype=np.int32, count=num_scans)
        retention_times = np.fromiter(result.RetentionTimes, dtype=np.float64, count=num_scans)
        ms_orders = np.fromiter(result.MsOrders, dtype=np.int32, count=num_scans)
        polarities = np.fromiter(result.Polarities, dtype=np.int32, count=num_scans)
        precursor_mzs = np.fromiter(result.PrecursorMzs, dtype=np.float64, count=num_scans)
        isolation_widths = np.fromiter(result.IsolationWidths, dtype=np.float64, count=num_scans)
        collision_energies = np.fromiter(result.CollisionEnergies, dtype=np.float64, count=num_scans)
        scan_lengths = np.fromiter(result.ScanLengths, dtype=np.int32, count=num_scans)

        # Use numpy repeat for expansion
        df_scan_numbers = np.repeat(scan_numbers, scan_lengths)
        df_retention_times = np.repeat(retention_times, scan_lengths)
        df_ms_orders = np.repeat(ms_orders, scan_lengths)
        df_polarities = np.repeat(polarities, scan_lengths)
        df_precursor_mzs = np.repeat(precursor_mzs, scan_lengths)
        df_isolation_widths = np.repeat(isolation_widths, scan_lengths)
        df_collision_energies = np.repeat(collision_energies, scan_lengths)

        return pd.DataFrame({
            "Scan": df_scan_numbers,
            "RetentionTime": df_retention_times,
            "MS Order": df_ms_orders,
            "Mass": all_masses,
            "Intensity": all_intensities,
            "Polarity": df_polarities,
            "PrecursorMz": df_precursor_mzs,
            "IsolationWidth": df_isolation_widths,
            "CollisionEnergy": df_collision_energies,
        })

    def to_polars(
        self,
        include_ms2: bool = False,
        filter_threshold: float = 0,
        max_workers: int = 0
    ):
        """Convert all scans to Polars DataFrame using C# native parallel processing.

        This returns a native Polars DataFrame (not converted to pandas).

        Args:
            include_ms2: Include MS2 spectra in output
            filter_threshold: Filter peaks below this intensity (0 = no filter)
            max_workers: Number of parallel workers (0 = auto)

        Returns:
            Polars DataFrame with all scan data

        Raises:
            ImportError: If Polars is not installed
        """
        if not HAS_POLARS:
            raise ImportError("Polars is not installed. Install with: pip install polars")

        # Use C# parallel reader
        csharp_reader = CSharpParallelReader(str(self.file_path))
        result = csharp_reader.ReadAllScansParallel(
            includeMs2=include_ms2,
            filterThreshold=filter_threshold,
            maxWorkers=max_workers if max_workers > 0 else 0
        )
        csharp_reader.Dispose()

        # Convert to Polars DataFrame
        total_points = int(result.TotalDataPoints)
        num_scans = int(result.TotalScans)

        all_masses = np.fromiter(result.AllMasses, dtype=np.float64, count=total_points)
        all_intensities = np.fromiter(result.AllIntensities, dtype=np.float64, count=total_points)

        scan_numbers = np.fromiter(result.ScanNumbers, dtype=np.int32, count=num_scans)
        retention_times = np.fromiter(result.RetentionTimes, dtype=np.float64, count=num_scans)
        ms_orders = np.fromiter(result.MsOrders, dtype=np.int32, count=num_scans)
        polarities = np.fromiter(result.Polarities, dtype=np.int32, count=num_scans)
        precursor_mzs = np.fromiter(result.PrecursorMzs, dtype=np.float64, count=num_scans)
        isolation_widths = np.fromiter(result.IsolationWidths, dtype=np.float64, count=num_scans)
        collision_energies = np.fromiter(result.CollisionEnergies, dtype=np.float64, count=num_scans)
        scan_lengths = np.fromiter(result.ScanLengths, dtype=np.int32, count=num_scans)

        # Use numpy repeat for expansion
        df_scan_numbers = np.repeat(scan_numbers, scan_lengths)
        df_retention_times = np.repeat(retention_times, scan_lengths)
        df_ms_orders = np.repeat(ms_orders, scan_lengths)
        df_polarities = np.repeat(polarities, scan_lengths)
        df_precursor_mzs = np.repeat(precursor_mzs, scan_lengths)
        df_isolation_widths = np.repeat(isolation_widths, scan_lengths)
        df_collision_energies = np.repeat(collision_energies, scan_lengths)

        return pl.DataFrame({
            "Scan": df_scan_numbers,
            "RetentionTime": df_retention_times,
            "MS Order": df_ms_orders,
            "Mass": all_masses,
            "Intensity": all_intensities,
            "Polarity": df_polarities,
            "PrecursorMz": df_precursor_mzs,
            "IsolationWidth": df_isolation_widths,
            "CollisionEnergy": df_collision_energies,
        })


def read_multiple_files(
    file_paths: list[str | Path],
    include_ms2: bool = False,
    filter_threshold: float = 0,
    max_files_parallel: int = 0,
    max_scans_parallel: int = 0,
    use_polars: bool = True,
    return_native_polars: bool = False
) -> dict[str, pd.DataFrame | "pl.DataFrame"]:
    """Read multiple RAW files in parallel using C# native multi-threading.

    This function processes multiple RAW files concurrently, bypassing Python's
    GIL limitation by doing all parallel work in native .NET code.

    Args:
        file_paths: List of paths to RAW files
        include_ms2: Include MS2 spectra in output
        filter_threshold: Filter peaks below this intensity (0 = no filter)
        max_files_parallel: Max files to process concurrently (0 = auto, typically 2-4)
        max_scans_parallel: Max parallel workers per file for scan reading (0 = auto)
        use_polars: Use Polars for DataFrame construction (faster)
        return_native_polars: Return native Polars DataFrames instead of pandas

    Returns:
        Dictionary mapping file paths to DataFrames. Failed files will have None.

    Example:
        >>> files = ["/path/to/file1.raw", "/path/to/file2.raw"]
        >>> results = read_multiple_files(files)
        >>> for path, df in results.items():
        ...     if df is not None:
        ...         print(f"{path}: {df.shape}")
    """
    from System import Array, String

    # Convert paths to strings
    str_paths = [str(p) for p in file_paths]
    file_array = Array[String](str_paths)

    # Call C# multi-file reader
    result = CSharpMultiFileReader.ReadMultipleFiles(
        file_array,
        includeMs2=include_ms2,
        filterThreshold=filter_threshold,
        maxFilesParallel=max_files_parallel if max_files_parallel > 0 else 0,
        maxScansParallel=max_scans_parallel if max_scans_parallel > 0 else 0
    )

    # Convert results to DataFrames
    output = {}
    for file_result in result.FileResults:
        file_path = str(file_result.FilePath)

        if not file_result.Success:
            logger.warning(f"Failed to read {file_result.FileName}: {file_result.ErrorMessage}")
            output[file_path] = None
            continue

        bulk_data = file_result.Data
        df = _bulk_result_to_dataframe(bulk_data, use_polars, return_native_polars)
        output[file_path] = df

    return output


def _bulk_result_to_dataframe(
    result,
    use_polars: bool = True,
    return_native_polars: bool = False
) -> pd.DataFrame | "pl.DataFrame":
    """Convert BulkScanResult to DataFrame.

    Internal helper function for converting C# BulkScanResult to DataFrame.
    """
    total_points = int(result.TotalDataPoints)
    num_scans = int(result.TotalScans)

    if total_points == 0:
        if return_native_polars and HAS_POLARS:
            return pl.DataFrame()
        return pd.DataFrame()

    # Extract arrays from C# result
    all_masses = np.fromiter(result.AllMasses, dtype=np.float64, count=total_points)
    all_intensities = np.fromiter(result.AllIntensities, dtype=np.float64, count=total_points)

    scan_numbers = np.fromiter(result.ScanNumbers, dtype=np.int32, count=num_scans)
    retention_times = np.fromiter(result.RetentionTimes, dtype=np.float64, count=num_scans)
    ms_orders = np.fromiter(result.MsOrders, dtype=np.int32, count=num_scans)
    polarities = np.fromiter(result.Polarities, dtype=np.int32, count=num_scans)
    precursor_mzs = np.fromiter(result.PrecursorMzs, dtype=np.float64, count=num_scans)
    isolation_widths = np.fromiter(result.IsolationWidths, dtype=np.float64, count=num_scans)
    collision_energies = np.fromiter(result.CollisionEnergies, dtype=np.float64, count=num_scans)
    scan_lengths = np.fromiter(result.ScanLengths, dtype=np.int32, count=num_scans)

    # Expand per-scan values to per-datapoint using numpy repeat
    df_scan_numbers = np.repeat(scan_numbers, scan_lengths)
    df_retention_times = np.repeat(retention_times, scan_lengths)
    df_ms_orders = np.repeat(ms_orders, scan_lengths)
    df_polarities = np.repeat(polarities, scan_lengths)
    df_precursor_mzs = np.repeat(precursor_mzs, scan_lengths)
    df_isolation_widths = np.repeat(isolation_widths, scan_lengths)
    df_collision_energies = np.repeat(collision_energies, scan_lengths)

    data = {
        "Scan": df_scan_numbers,
        "RetentionTime": df_retention_times,
        "MS Order": df_ms_orders,
        "Mass": all_masses,
        "Intensity": all_intensities,
        "Polarity": df_polarities,
        "PrecursorMz": df_precursor_mzs,
        "IsolationWidth": df_isolation_widths,
        "CollisionEnergy": df_collision_energies,
    }

    if return_native_polars:
        if not HAS_POLARS:
            raise ImportError("Polars is not installed. Install with: pip install polars")
        return pl.DataFrame(data)

    if use_polars and HAS_POLARS:
        return pl.DataFrame(data).to_pandas()

    return pd.DataFrame(data)

