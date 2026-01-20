#!/usr/bin/env python3
"""
Test actual RAW file reading functionality.
"""

import sys
from pathlib import Path

# Add the src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_import():
    """Test importing the RawFileReader module."""
    print("Testing import of RawFileReader...")
    try:
        from RawFileReader import RawFileReader
        print("✓ RawFileReader imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_factory():
    """Test that the RawFileReaderAdapter.FileFactory works."""
    print("\nTesting RawFileReaderAdapter.FileFactory...")
    try:
        from pythonnet import load
        load("coreclr")
        import clr

        lib_path = Path(__file__).parent.parent / "src" / "RawFileReader" / "lib"
        sys.path.append(str(lib_path))

        clr.AddReference('ThermoFisher.CommonCore.Data')
        clr.AddReference('ThermoFisher.CommonCore.RawFileReader')

        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

        # Try creating a file reader (will fail on non-existent file but should not crash)
        print("Testing FileFactory creation (expect file not found error)...")
        try:
            reader = RawFileReaderAdapter.FileFactory("nonexistent.raw")
            print(f"  FileFactory returned: {reader}")
            if reader:
                is_open = reader.IsOpen
                print(f"  IsOpen: {is_open}")
        except Exception as e:
            # This is expected for non-existent file
            print(f"  Expected behavior - file not found handled: {type(e).__name__}")

        print("✓ FileFactory test passed")
        return True
    except Exception as e:
        print(f"✗ FileFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_raw_files():
    """Find any .raw files in the project for testing."""
    project_root = Path(__file__).parent.parent
    raw_files = list(project_root.rglob("*.raw")) + list(project_root.rglob("*.RAW"))
    return raw_files


def test_with_real_file():
    """Test with a real RAW file if available."""
    raw_files = find_raw_files()

    if not raw_files:
        print("\nNo RAW files found for testing - skipping real file test")
        return None

    print(f"\nFound {len(raw_files)} RAW file(s), testing with first one...")
    test_file = raw_files[0]
    print(f"  File: {test_file}")

    try:
        from RawFileReader import RawFileReader

        reader = RawFileReader(str(test_file))
        print(f"✓ File opened successfully")
        print(f"  File name: {reader.file_name}")
        print(f"  Scan range: {reader.scan_range}")

        # Try reading first scan
        first_scan = reader.scan_range[0]
        result = reader.get_spectrum(first_scan)
        # result is (rt, scan_num, mz_array, intensity_array, filter, ...)
        mz = result[2]
        print(f"  First scan ({first_scan}) has {len(mz)} data points")

        print("✓ Real file test passed")
        return True
    except Exception as e:
        print(f"✗ Real file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("FULL FUNCTIONALITY TEST")
    print("="*60)

    import platform
    print(f"\nPython: {platform.python_version()} ({platform.machine()})")
    print(f"Executable: {sys.executable}")

    results = []

    # Test 1: Import
    results.append(("Import", test_import()))

    # Test 2: FileFactory
    results.append(("FileFactory", test_file_factory()))

    # Test 3: Real file (if available)
    real_file_result = test_with_real_file()
    if real_file_result is not None:
        results.append(("Real File", real_file_result))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed! The x64 Python + Rosetta 2 setup works.")
    else:
        print("\n✗ Some tests failed. See details above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
