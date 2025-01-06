from pythonnet import load
import logging
load("coreclr")

import clr
import sys
import numpy as np
# import pandas as pd
import polars as pl
import tqdm

from psims.mzml import MzMLWriter

# get absolute path of the current file
import os
current_file_path = os.path.abspath(__file__)
print(current_file_path)
lib_path = f"{'/'.join(current_file_path.split('/')[:-1])}/lib"
print(lib_path)
sys.path.append(lib_path)

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
print("Successfully loaded ThermoFisher.CommonCore.RawFileReader")

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

    def get_spectrum(self, scan_number: int, include_ms2: bool = False) -> pl.DataFrame:
        scan_statistics = self.rawFile.GetScanStatsForScanNumber(scan_number)
        scanFilter = IScanFilter(self.rawFile.GetFilterForScanNumber(scan_number))
        ms_order = scanFilter.MSOrder
        ms_order = 1 if ms_order == MSOrderType.Ms else 2
        if not include_ms2:
            if ms_order == 2:
                return None
        if scan_statistics.IsCentroidScan:
            centroid_scan = self.rawFile.GetCentroidStream(scan_number, False)
            scan = pl.DataFrame(
                {
                    "Scan": scan_number,
                    "MS Order": ms_order,
                    'Mass': DotNetArrayToNPArray(centroid_scan.Positions, float),
                    'Intensity': DotNetArrayToNPArray(centroid_scan.Intensities, float)
                }
            )
        else:
            segmented_scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
            scan = pl.DataFrame(
                {
                    "Scan": scan_number,
                    "MS Order": ms_order,
                    'Mass': DotNetArrayToNPArray(segmented_scan.Positions, float),
                    'Intensity': DotNetArrayToNPArray(segmented_scan.Intensities, float)
                }
            )
        # scan.reindex(columns=['Scan', 'RetentionTime', 'MS Order', 'Mass', 'Intensity'])
        return scan

    def write_mzml(self, output_path: str, include_ms2: bool = False):
        with MzMLWriter(output_path) as writer:
            writer.controlled_vocabularies()
            writer.file_description([
                "MS1 spectrum",
                "MSn spectrum",
                "centroid spectrum"
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
                        scan_statistics = self.rawFile.GetScanStatsForScanNumber(scan_number)
                        scanFilter = IScanFilter(self.rawFile.GetFilterForScanNumber(scan_number))
                        ms_order = scanFilter.MSOrder
                        ms_order = 1 if ms_order == MSOrderType.Ms else 2
                        if not include_ms2:
                            if ms_order == 2:
                                return None
                        if scan_statistics.IsCentroidScan:
                            scan = self.rawFile.GetCentroidStream(scan_number, False)
                        else:
                            scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
                        scan_id = f"scan={scan_number}"
                        retention_time = self.rawFile.RetentionTimeFromScanNumber(scan_number) * 60
                        mz_array = DotNetArrayToNPArray(scan.Positions, float)
                        intensity_array = DotNetArrayToNPArray(scan.Intensities, float)
                        writer.write_spectrum(
                            mz_array,
                            intensity_array,
                            id=scan_id,
                            params=[
                                f"MS{ms_order} spectrum",
                                {"ms level": ms_order},
                                {"total ion current": np.sum(intensity_array)},
                                {"scan start time": retention_time}
                            ]
                        )


    def get_full_spectrum(self, ms2_include: bool = False) -> pl.DataFrame:
        scan_list = [
            spectrum for spectrum in (self.get_spectrum(scan, False) for scan in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1])))
            if spectrum is not None
        ]
        # for scan in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1])):
        #     scan_list.append(self.get_spectrum(scan, False))
        #     whole_spectrum = pd.concat(scan_list)
        whole_spectrum = pl.concat(scan_list)
        return whole_spectrum



if __name__ == "__main__":
    print("Hello")
    raw_file = RawFileReader("../Data/20241213_folate_standard_re_neg_f_01.raw")
    for key, item in raw_file.instrument_info.items():
        print(f"{key}: {item}")

    print(raw_file.scan_range)
    raw_file.get_full_spectrum()
