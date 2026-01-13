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

from System import *
from System.Collections.Generic import *

from ThermoFisher.CommonCore.Data import ToleranceUnits
from ThermoFisher.CommonCore.Data import Extensions
from ThermoFisher.CommonCore.Data.Business import ChromatogramSignal, ChromatogramTraceSettings, DataUnits, Device, GenericDataTypes, SampleType, Scan, TraceType, MassOptions, Range
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IChromatogramSettings, IScanEventBase, IScanFilter, RawFileClassification
from ThermoFisher.CommonCore.MassPrecisionEstimator import PrecisionEstimate
from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

logger = logging.getLogger(__name__)

logger.info("Successfully loaded ThermoFisher.CommonCore.RawFileReader")


def DotNetArrayToNPArray(arr, dtype):
    if arr is None:
        return np.array([], dtype=dtype)
    return np.array(list(arr), dtype=dtype)


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


    def to_dataframe(self, include_ms2: bool = False) -> pd.DataFrame:
        with logging_redirect_tqdm():
            scan_list = [
                spectrum for spectrum in (self.to_series(scan, include_ms2) for scan in trange(self.scan_range[0], self.scan_range[1],desc=f"Reading {self.file_name}", leave=False))
                if spectrum is not None
            ]
        whole_spectrum = pd.concat(scan_list)
        return whole_spectrum

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
        intensities = DotNetArrayToNPArray(data.IntensitiesArray, float).T

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

