from pythonnet import load
import logging
load("coreclr")

import clr
import sys
import numpy as np
import pandas as pd
import tqdm

sys.path.append("./lib/")

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

    def get_spectrum(self, scan_number: int, include_ms2: bool = False) -> pd.DataFrame:
        scan_statistics = self.rawFile.GetScanStatsForScanNumber(scan_number)
        scanFilter = IScanFilter(self.rawFile.GetFilterForScanNumber(scan_number))
        ms_order = scanFilter.MSOrder
        ms_order = 1 if ms_order == MSOrderType.Ms else 2
        if not include_ms2:
            if ms_order == 2:
                return None
        if scan_statistics.IsCentroidScan:
            centroid_scan = self.rawFile.GetCentroidStream(scan_number, False)
            scan = pd.DataFrame(np.array([
                DotNetArrayToNPArray(centroid_scan.Masses, float),
                DotNetArrayToNPArray(centroid_scan.Intensities, float)]).transpose(),
                columns=['Mass', 'Intensity']
            )
        else:
            segmented_scan = self.rawFile.GetSegmentedScanFromScanNumber(scan_number, scan_statistics)
            scan = pd.DataFrame(np.array([
                DotNetArrayToNPArray(segmented_scan.Positions, float),
                DotNetArrayToNPArray(segmented_scan.Intensities, float)]).transpose(),
                columns=['Mass', 'Intensity']
            )
        scan["Scan"] = scan_number
        scan["RetentionTime"] = self.rawFile.RetentionTimeFromScanNumber(scan_number)
        scan["MS Order"] = ms_order
        scan.reindex(columns=['Scan', 'RetentionTime', 'MS Order', 'Mass', 'Intensity'])
        return scan

    def get_full_ms1_spectrum(self) -> pd.DataFrame:
        scan_list = [self.get_spectrum(scan, False) for scan in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1]))]
        # for scan in tqdm.tqdm(range(self.scan_range[0], self.scan_range[1])):
        #     scan_list.append(self.get_spectrum(scan, False))
        #     whole_spectrum = pd.concat(scan_list)
        whole_spectrum = pd.concat(scan_list)
        print(whole_spectrum.info())
        print(whole_spectrum.head())
        print(whole_spectrum.tail())
        tic = whole_spectrum.groupby('Scan').sum()
        print(tic.info())
        return whole_spectrum





if __name__ == "__main__":
    print("Hello")
    raw_file = RawFileReader("../20241213_folate_standard_re_neg_f_01.raw")
    for key, item in raw_file.instrument_info.items():
        print(f"{key}: {item}")

    print(raw_file.scan_range)
    raw_file.get_full_ms1_spectrum()
