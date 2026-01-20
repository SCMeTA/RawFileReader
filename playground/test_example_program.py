#!/usr/bin/env python3
"""
Test script based on ExampleProgram/Framework/RawFileReaderExample.py
Tests the Thermo DLL loading directly using pythonnet.
"""

from __future__ import print_function
import platform
import sys
import os

print("="*60)
print("THERMO DLL TEST (ExampleProgram style)")
print("="*60)
print(f"Python: {platform.python_version()} ({platform.machine()})")
print(f"Executable: {sys.executable}")
print()

# Load pythonnet with coreclr
print("Loading pythonnet with coreclr...")
try:
    from pythonnet import load
    load("coreclr")
    import clr
    print("✓ pythonnet loaded successfully")
except Exception as e:
    print(f"✗ Failed to load pythonnet: {e}")
    sys.exit(1)

# Add the DLL path (using Net8 Assemblies like the example)
lib_path = os.path.join(os.path.dirname(__file__), "..", "Libs", "NetCore", "Net8", "Assemblies")
lib_path = os.path.abspath(lib_path)
print(f"\nAdding DLL path: {lib_path}")

if os.path.exists(lib_path):
    sys.path.append(lib_path)
    print(f"✓ Path exists")
else:
    print(f"✗ Path does not exist!")
    # Try alternative path
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src", "RawFileReader", "lib")
    lib_path = os.path.abspath(lib_path)
    print(f"Trying alternative: {lib_path}")
    sys.path.append(lib_path)

# Load DLLs
print("\nLoading Thermo DLLs...")
dlls = [
    'ThermoFisher.CommonCore.Data',
    'ThermoFisher.CommonCore.RawFileReader',
    'ThermoFisher.CommonCore.BackgroundSubtraction',
    'ThermoFisher.CommonCore.MassPrecisionEstimator',
]

for dll in dlls:
    try:
        clr.AddReference(dll)
        print(f"✓ {dll}")
    except Exception as e:
        print(f"✗ {dll}: {e}")
        sys.exit(1)

# Import .NET types
print("\nImporting .NET types...")
try:
    from System import Environment, DateTime, Enum
    from ThermoFisher.CommonCore.Data.Business import Device, DataUnits, SampleType
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
    print("✓ All .NET types imported successfully")
except Exception as e:
    print(f"✗ Failed to import .NET types: {e}")
    sys.exit(1)

# Print system info (like the example program)
print("\n" + "="*60)
print("SYSTEM INFORMATION (from .NET)")
print("="*60)
print(f"OS Version: {Environment.OSVersion}")
print(f"64-bit OS: {Environment.Is64BitOperatingSystem}")
print(f"64-bit Process: {Environment.Is64BitProcess}")
print(f"Computer: {Environment.MachineName}")
print(f"Processors: {Environment.ProcessorCount}")
print(f"Date: {DateTime.Now}")

# Find a test RAW file
print("\n" + "="*60)
print("TESTING RAW FILE ACCESS")
print("="*60)

data_dir = os.path.join(os.path.dirname(__file__), "..", "Data", "test")
raw_files = []
if os.path.exists(data_dir):
    raw_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.raw')]

if raw_files:
    test_file = os.path.join(data_dir, raw_files[0])
    print(f"Testing with: {os.path.basename(test_file)}")

    try:
        # Open the RAW file
        rawFile = RawFileReaderAdapter.FileFactory(test_file)

        if not rawFile.IsOpen or rawFile.IsError:
            print("✗ Unable to open the RAW file!")
            sys.exit(1)

        print("✓ RAW file opened successfully")

        # Select MS instrument
        rawFile.SelectInstrument(Device.MS, 1)

        # Get scan info
        firstScan = rawFile.RunHeaderEx.FirstSpectrum
        lastScan = rawFile.RunHeaderEx.LastSpectrum
        startTime = rawFile.RunHeaderEx.StartTime
        endTime = rawFile.RunHeaderEx.EndTime

        print(f"\nGeneral File Information:")
        print(f"  RAW file: {rawFile.FileName}")
        print(f"  RAW file version: {rawFile.FileHeader.Revision}")
        print(f"  Creation date: {rawFile.FileHeader.CreationDate}")
        print(f"  Instrument model: {rawFile.GetInstrumentData().Model}")
        print(f"  Number of scans: {rawFile.RunHeaderEx.SpectraCount}")
        print(f"  Scan range: {firstScan} - {lastScan}")
        print(f"  Time range: {startTime:.2f} - {endTime:.2f}")
        print(f"  Mass range: {rawFile.RunHeaderEx.LowMass:.4f} - {rawFile.RunHeaderEx.HighMass:.4f}")

        # Read first scan
        print(f"\nReading first scan ({firstScan})...")
        scanStats = rawFile.GetScanStatsForScanNumber(firstScan)

        if scanStats.IsCentroidScan:
            centroidStream = rawFile.GetCentroidStream(firstScan, False)
            print(f"✓ Centroid scan with {centroidStream.Length} data points")
        else:
            segmentedScan = rawFile.GetSegmentedScanFromScanNumber(firstScan, scanStats)
            print(f"✓ Profile scan with {segmentedScan.Positions.Length} data points")

        # Close file
        rawFile.Dispose()
        print("\n✓ RAW file closed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("No RAW files found in Data/test directory")
    print("Testing FileFactory with non-existent file...")
    try:
        rawFile = RawFileReaderAdapter.FileFactory("nonexistent.raw")
        print(f"FileFactory returned: IsOpen={rawFile.IsOpen}, IsError={rawFile.IsError}")
        print("✓ FileFactory works (file not found is expected)")
    except Exception as e:
        print(f"✗ FileFactory failed: {e}")

print("\n" + "="*60)
print("TEST COMPLETE - SUCCESS!")
print("="*60)
