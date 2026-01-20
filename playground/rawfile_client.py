#!/usr/bin/env python3
"""
RAW File Client - runs on ARM64 Python, communicates with x64 server process.
This script can be run with ARM64 Python.
"""

import subprocess
import json
import sys
import os
from pathlib import Path


class RawFileClient:
    """Client that communicates with x64 RawFileServer subprocess."""

    def __init__(self, x64_python=None):
        """
        Initialize client with path to x64 Python interpreter.

        Args:
            x64_python: Path to x64 Python. If None, tries to find .venv/bin/python
        """
        if x64_python is None:
            # Try to find x64 Python in .venv
            venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
            if venv_python.exists():
                x64_python = str(venv_python)
            else:
                raise RuntimeError("x64 Python not found. Please specify path.")

        self.x64_python = x64_python
        self.server_script = Path(__file__).parent / "rawfile_server.py"
        self.process = None
        self._start_server()

    def _start_server(self):
        """Start the x64 server subprocess."""
        self.process = subprocess.Popen(
            [self.x64_python, str(self.server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        response = json.loads(self.process.stdout.readline())
        if response.get("status") != "ready":
            raise RuntimeError(f"Server failed to start: {response}")

    def _send_request(self, request):
        """Send request to server and get response."""
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        return json.loads(response)

    def open(self, file_path):
        """Open a RAW file."""
        return self._send_request({"cmd": "open", "path": str(file_path)})

    def get_scan(self, file_id, scan_number):
        """Get spectrum for a scan."""
        return self._send_request({
            "cmd": "scan",
            "file_id": file_id,
            "scan_number": scan_number
        })

    def close(self, file_id):
        """Close a RAW file."""
        return self._send_request({"cmd": "close", "file_id": file_id})

    def shutdown(self):
        """Shutdown the server."""
        if self.process:
            try:
                self._send_request({"cmd": "quit"})
            except:
                pass
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass


class RawFileReaderARM64:
    """
    ARM64-compatible RAW file reader using IPC to x64 subprocess.
    Drop-in replacement API for reading Thermo RAW files on ARM64.
    """

    def __init__(self, file_path, x64_python=None):
        self.client = RawFileClient(x64_python)
        self.file_path = file_path

        # Open the file
        result = self.client.open(file_path)
        if "error" in result:
            raise RuntimeError(result["error"])

        self.file_id = result["file_id"]
        self.file_name = result["file_name"]
        self.first_scan = result["first_scan"]
        self.last_scan = result["last_scan"]
        self.num_scans = result["num_scans"]
        self.start_time = result["start_time"]
        self.end_time = result["end_time"]
        self.low_mass = result["low_mass"]
        self.high_mass = result["high_mass"]
        self.instrument_model = result["instrument_model"]

    def get_scan(self, scan_number):
        """Get spectrum data for a scan number."""
        result = self.client.get_scan(self.file_id, scan_number)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        """Close the RAW file."""
        self.client.close(self.file_id)
        self.client.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test the client
if __name__ == "__main__":
    import platform

    print("="*60)
    print("ARM64 RAW FILE CLIENT TEST")
    print("="*60)
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"Executable: {sys.executable}")
    print()

    # Find a test file
    data_dir = Path(__file__).parent.parent / "Data" / "test"
    raw_files = list(data_dir.glob("*.raw")) + list(data_dir.glob("*.RAW"))

    if not raw_files:
        print("No RAW files found in Data/test")
        sys.exit(1)

    test_file = raw_files[0]
    print(f"Testing with: {test_file.name}")
    print()

    try:
        print("Starting x64 server subprocess...")
        reader = RawFileReaderARM64(str(test_file))
        print("✓ Server started and file opened")
        print()

        print("File Information:")
        print(f"  File: {reader.file_name}")
        print(f"  Instrument: {reader.instrument_model}")
        print(f"  Scans: {reader.first_scan} - {reader.last_scan} ({reader.num_scans} total)")
        print(f"  Time: {reader.start_time:.2f} - {reader.end_time:.2f} min")
        print(f"  Mass: {reader.low_mass:.2f} - {reader.high_mass:.2f}")
        print()

        print(f"Reading scan {reader.first_scan}...")
        scan = reader.get_scan(reader.first_scan)
        print(f"✓ Got {len(scan['mz'])} data points")
        print(f"  RT: {scan['rt']:.4f} min")
        print(f"  Centroid: {scan['is_centroid']}")
        print(f"  m/z range: {min(scan['mz']):.4f} - {max(scan['mz']):.4f}")
        print()

        reader.close()
        print("✓ File closed and server shutdown")

        print()
        print("="*60)
        print("SUCCESS! ARM64 Python can read Thermo RAW files via IPC")
        print("="*60)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
