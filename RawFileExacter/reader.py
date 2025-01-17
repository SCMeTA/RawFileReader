from pythonnet import load
import logging
load("coreclr")

import clr
import sys
import numpy as np
# import pandas as pd
import polars as pl
import tqdm
from pathlib import Path

from psims.mzml import MzMLWriter

# get absolute path of the current file
import os

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
from ThermoFisher.CommonCore.Data.Business import ChromatogramSignal, ChromatogramTraceSettings, DataUnits, Device, GenericDataTypes, SampleType, Scan, TraceType
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import IChromatogramSettings, IScanEventBase, IScanFilter, RawFileClassification
from ThermoFisher.CommonCore.MassPrecisionEstimator import PrecisionEstimate
from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

logger = logging.getLogger(__name__)

logger.info("Successfully loaded ThermoFisher.CommonCore.RawFileReader")


def DotNetArrayToNPArray(arr, dtype):
    return np.array(list(arr), dtype=dtype)


class RawFileReader:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        print(f"Opening {self.file_path}")
        self.rawFile = self.__open_raw_file()
        self.scan_range: list = self.__get_scan_number()
        self.instrument_info: dict = self.__get_instrument_info()

    def __open_raw_file(self):
        raw_file = RawFileReaderAdapter.FileFactory(self.file_path)
        if raw_file.IsOpen:
            logger.info(f"Successfully opened {self.file_path}")
            print("Successfully open the file")
            raw_file.SelectInstrument(Device.MS, 1)
            return raw_file
        else:
            logger.error(f"Failed to open {self.file_path}")
            print("Failed to open the file")
            return None

    def __get_scan_number(self):
        first_scan = self.rawFile.RunHeaderEx.FirstSpectrum
        last_scan = self.rawFile.RunHeaderEx.LastSpectrum
        print(f"First scan: {first_scan}, Last scan: {last_scan}")
        # Get retention time of the first and last scan
        first_rt = self.rawFile.RetentionTimeFromScanNumber(first_scan)
        last_rt = self.rawFile.RetentionTimeFromScanNumber(last_scan)
        print(f"First RT: {first_rt}, Last RT: {last_rt}")
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
            # scan = pl.DataFrame(
            #     {
            #         "Scan": scan_number,
            #         "MS Order": ms_order,
            #         'Mass': DotNetArrayToNPArray(centroid_scan.Positions, float),
            #         'Intensity': DotNetArrayToNPArray(centroid_scan.Intensities, float)
            #     }
            # )
            masses = DotNetArrayToNPArray(centroid_scan.Masses, float)
            intensities = DotNetArrayToNPArray(centroid_scan.Intensities, int)
            is_centroid = True
        else:
            segmented_scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
            # scan = pl.DataFrame(
            #     {
            #         "Scan": scan_number,
            #         "MS Order": ms_order,
            #         'Mass': DotNetArrayToNPArray(segmented_scan.Positions, float),
            #         'Intensity': DotNetArrayToNPArray(segmented_scan.Intensities, float)
            #     }
            # )
            masses = DotNetArrayToNPArray(segmented_scan.Masses, float)
            intensities = DotNetArrayToNPArray(segmented_scan.Intensities, int)
            is_centroid = False
        # scan.reindex(columns=['Scan', 'RetentionTime', 'MS Order', 'Mass', 'Intensity'])
        return retention_time, ms_order, masses, intensities, polarity, is_centroid

    def intensity_filter(self, threshold: int, mz_array: np.array, intensity: np.array):
        # filter the intensity and also remove the mz values
        indices_to_keep = np.where(intensity > threshold)
        return mz_array[indices_to_keep], intensity[indices_to_keep]

    def to_series(self, scan_number: int, include_ms2: bool = False) -> pl.Series | None:
        retention_time, ms_order, masses, intensities, polarity, is_centroid = self.get_spectrum(scan_number, include_ms2)
        if masses is None:
            return None
        return pl.Series(
            {
                "Scan": scan_number,
                "MS Order": ms_order,
                "Mass": masses,
                "Intensity": intensities,
                "Polarity": polarity
            }
        )

    def to_mzml(self, output_path: str, include_ms2: bool = False, filter_threshold: int | None = None):
        with MzMLWriter(output_path) as writer:
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
            config = writer.InstrumentConfiguration(id="IC1", component_list=[source, analyzer, detector], params=["Orbitrap-Astral"])
            writer.instrument_configuration_list([config])
            methods = [
                writer.ProcessingMethod(order=1, software_reference="psims-writer", params=[
                    "Conversion to mzML"
                ])
            ]
            processing = writer.DataProcessing(methods, id='DP1')
            writer.data_processing_list([processing])
            with writer.run(id="run1", instrument_configuration='IC1'):
                scan_count = self.scan_range[1] - self.scan_range[0] + 1
                with writer.spectrum_list(count=scan_count):
                    for scan_number in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1])):
                        # scan_statistics = self.rawFile.GetScanStatsForScanNumber(scan_number)
                        # scanFilter = IScanFilter(self.rawFile.GetFilterForScanNumber(scan_number))
                        # ionization_mode = scanFilter.IonizationMode
                        # ms_order = scanFilter.MSOrder
                        # ms_order = 1 if ms_order == MSOrderType.Ms else 2
                        # if not include_ms2:
                        #     if ms_order == 2:
                        #         continue
                        # if scan_statistics.IsCentroidScan:
                        #     scan = self.rawFile.GetCentroidStream(scan_number, False)
                        # else:
                        #     scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
                        # scan_id = f"scan={scan_number}"
                        # retention_time = self.rawFile.RetentionTimeFromScanNumber(scan_number)
                        # mz_array = DotNetArrayToNPArray(scan.Positions, float)
                        # intensity_array = DotNetArrayToNPArray(scan.Intensities, float)
                        retention_time, ms_order, mz_array, intensity_array, polarity, is_centroid = self.get_spectrum(scan_number, include_ms2)
                        scan_id = f"scan={scan_number}"
                        if filter_threshold:
                            mz_array, intensity_array = self.intensity_filter(filter_threshold, mz_array, intensity_array)
                            mz_array = np.round(mz_array, 5)
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
                            ]
                        )


    def to_dataframe(self, include_ms2: bool = False) -> pl.DataFrame:
        scan_list = [
            spectrum for spectrum in (self.to_series(scan, include_ms2) for scan in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1])))
            if spectrum is not None
        ]
        whole_spectrum = pl.concat(scan_list)
        return whole_spectrum



if __name__ == "__main__":
    print("Hello")
    raw_file = RawFileReader("../Data/20241213_folate_standard_re_neg_f_01.raw")
    for key, item in raw_file.instrument_info.items():
        print(f"{key}: {item}")

    print(raw_file.scan_range)
    raw_file.to_dataframe()
