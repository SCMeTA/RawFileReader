#!/usr/bin/env python3
"""
RAW File Server - runs as x64 process, serves data to ARM64 clients via IPC.
This script must be run with x64 Python (.venv/bin/python).
"""

import sys
import json
import struct
from pathlib import Path

# Add lib path
lib_path = Path(__file__).parent.parent / "src" / "RawFileReader" / "lib"
sys.path.insert(0, str(lib_path))

from pythonnet import load
load("coreclr")
import clr

clr.AddReference('ThermoFisher.CommonCore.Data')
clr.AddReference('ThermoFisher.CommonCore.RawFileReader')

from ThermoFisher.CommonCore.Data.Business import Device
from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter


class RawFileServer:
    def __init__(self):
        self.files = {}  # file_id -> rawFile object

    def open_file(self, file_path):
        """Open a RAW file and return file info."""
        raw_file = RawFileReaderAdapter.FileFactory(file_path)
        if not raw_file.IsOpen or raw_file.IsError:
            return {"error": "Failed to open file"}

        raw_file.SelectInstrument(Device.MS, 1)

        file_id = str(id(raw_file))
        self.files[file_id] = raw_file

        return {
            "file_id": file_id,
            "file_name": str(raw_file.FileName),
            "first_scan": raw_file.RunHeaderEx.FirstSpectrum,
            "last_scan": raw_file.RunHeaderEx.LastSpectrum,
            "num_scans": raw_file.RunHeaderEx.SpectraCount,
            "start_time": raw_file.RunHeaderEx.StartTime,
            "end_time": raw_file.RunHeaderEx.EndTime,
            "low_mass": raw_file.RunHeaderEx.LowMass,
            "high_mass": raw_file.RunHeaderEx.HighMass,
            "instrument_model": str(raw_file.GetInstrumentData().Model),
        }

    def get_scan(self, file_id, scan_number):
        """Get spectrum data for a scan."""
        raw_file = self.files.get(file_id)
        if not raw_file:
            return {"error": "File not found"}

        scan_stats = raw_file.GetScanStatsForScanNumber(scan_number)
        rt = raw_file.RetentionTimeFromScanNumber(scan_number)

        if scan_stats.IsCentroidScan:
            stream = raw_file.GetCentroidStream(scan_number, False)
            mz = [stream.Masses[i] for i in range(stream.Length)]
            intensity = [stream.Intensities[i] for i in range(stream.Length)]
        else:
            seg = raw_file.GetSegmentedScanFromScanNumber(scan_number, scan_stats)
            mz = [seg.Positions[i] for i in range(seg.Positions.Length)]
            intensity = [seg.Intensities[i] for i in range(seg.Intensities.Length)]

        return {
            "scan_number": scan_number,
            "rt": rt,
            "mz": mz,
            "intensity": intensity,
            "is_centroid": scan_stats.IsCentroidScan,
        }

    def close_file(self, file_id):
        """Close a RAW file."""
        raw_file = self.files.pop(file_id, None)
        if raw_file:
            raw_file.Dispose()
            return {"status": "closed"}
        return {"error": "File not found"}

    def handle_request(self, request):
        """Handle a JSON request."""
        cmd = request.get("cmd")

        if cmd == "open":
            return self.open_file(request["path"])
        elif cmd == "scan":
            return self.get_scan(request["file_id"], request["scan_number"])
        elif cmd == "close":
            return self.close_file(request["file_id"])
        elif cmd == "quit":
            return {"status": "quit"}
        else:
            return {"error": f"Unknown command: {cmd}"}


def main():
    """Main server loop - reads JSON from stdin, writes JSON to stdout."""
    server = RawFileServer()

    # Signal ready
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = server.handle_request(request)

            if response.get("status") == "quit":
                break

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
