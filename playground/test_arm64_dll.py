#!/usr/bin/env python3
"""
Test script to diagnose Thermo DLL loading issues on ARM64 Mac.

This script checks:
1. Current architecture
2. .NET runtime version and architecture
3. Python interpreter architecture
4. DLL loading capability
"""

import platform
import subprocess
import sys
import struct
from pathlib import Path


def get_python_arch():
    """Get the architecture of the current Python interpreter."""
    bits = struct.calcsize("P") * 8
    return f"{bits}-bit ({platform.machine()})"


def get_dotnet_info():
    """Get .NET runtime information."""
    try:
        # Get .NET version
        result = subprocess.run(
            ['dotnet', '--version'],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip()

        # Get .NET runtime info
        result_info = subprocess.run(
            ['dotnet', '--info'],
            capture_output=True,
            text=True
        )

        # Parse architecture from --info output
        arch = "unknown"
        for line in result_info.stdout.split('\n'):
            if 'Architecture' in line:
                arch = line.split(':')[-1].strip()
                break

        return {
            'version': version,
            'architecture': arch,
            'full_info': result_info.stdout
        }
    except FileNotFoundError:
        return None


def check_rosetta():
    """Check if running under Rosetta 2."""
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'sysctl.proc_translated'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == '1'
    except:
        return False


def check_dll_architecture(dll_path):
    """Check the architecture of a DLL file."""
    try:
        with open(dll_path, 'rb') as f:
            # Read DOS header
            dos_header = f.read(64)
            if dos_header[:2] != b'MZ':
                return "Not a valid PE file"

            # Get PE header offset
            pe_offset = int.from_bytes(dos_header[60:64], 'little')
            f.seek(pe_offset)

            # Read PE signature
            pe_sig = f.read(4)
            if pe_sig != b'PE\x00\x00':
                return "Invalid PE signature"

            # Read machine type
            machine = int.from_bytes(f.read(2), 'little')

            machine_types = {
                0x14c: 'x86 (32-bit)',
                0x8664: 'x64 (AMD64)',
                0xaa64: 'ARM64',
                0x1c0: 'ARM',
            }
            return machine_types.get(machine, f'Unknown (0x{machine:x})')
    except Exception as e:
        return f"Error: {e}"


def try_load_dll():
    """Try to load the Thermo DLLs using pythonnet."""
    print("\n" + "="*60)
    print("ATTEMPTING TO LOAD DLLs")
    print("="*60)

    try:
        from pythonnet import load
        print("✓ pythonnet imported successfully")

        # Try loading coreclr
        print("\nAttempting to load coreclr...")
        load("coreclr")
        print("✓ coreclr loaded successfully")

        import clr
        print("✓ clr imported successfully")

        # Add lib path to sys.path
        lib_path = Path(__file__).parent.parent / "src" / "RawFileReader" / "lib"
        if lib_path.exists():
            sys.path.append(str(lib_path))
            print(f"✓ Added lib path: {lib_path}")
        else:
            print(f"✗ Lib path not found: {lib_path}")
            return False

        # Try loading each DLL
        dlls_to_load = [
            'ThermoFisher.CommonCore.Data',
            'ThermoFisher.CommonCore.RawFileReader',
            'ThermoFisher.CommonCore.BackgroundSubtraction',
            'ThermoFisher.CommonCore.MassPrecisionEstimator',
        ]

        for dll in dlls_to_load:
            try:
                print(f"\nLoading {dll}...")
                clr.AddReference(dll)
                print(f"✓ {dll} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {dll}: {e}")
                return False

        # Try to use the API
        print("\nTrying to import Thermo classes...")
        from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter
        print("✓ RawFileReaderAdapter imported successfully")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("THERMO DLL ARM64 COMPATIBILITY DIAGNOSTIC")
    print("="*60)

    # System info
    print("\n--- SYSTEM INFORMATION ---")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Python architecture: {get_python_arch()}")
    print(f"Python executable: {sys.executable}")

    # Check Rosetta
    is_rosetta = check_rosetta()
    print(f"Running under Rosetta 2: {is_rosetta}")

    # .NET info
    print("\n--- .NET INFORMATION ---")
    dotnet_info = get_dotnet_info()
    if dotnet_info:
        print(f".NET version: {dotnet_info['version']}")
        print(f".NET architecture: {dotnet_info['architecture']}")
    else:
        print("✗ .NET not found!")

    # Check DLL architectures
    print("\n--- DLL ARCHITECTURE CHECK ---")
    lib_path = Path(__file__).parent.parent / "src" / "RawFileReader" / "lib"
    if lib_path.exists():
        dll_files = list(lib_path.glob("*.dll"))
        for dll in dll_files[:5]:  # Check first 5 DLLs
            arch = check_dll_architecture(dll)
            print(f"{dll.name}: {arch}")
    else:
        print(f"Lib path not found: {lib_path}")

    # Architecture compatibility check
    print("\n--- COMPATIBILITY ANALYSIS ---")
    machine = platform.machine().lower()

    if machine in ['x86_64', 'amd64', 'x64']:
        print("✓ Running on x64 architecture - compatible")
    elif machine == 'arm64':
        print("✗ Running on ARM64 architecture")
        print("\nThe Thermo Fisher DLLs are compiled for x64 only.")
        print("To use this package on ARM64 Mac, you have two options:\n")
        print("OPTION 1: Use Rosetta 2 with x64 Python")
        print("  1. Install x64 Homebrew: arch -x86_64 /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("  2. Install x64 Python: arch -x86_64 /usr/local/bin/brew install python@3.12")
        print("  3. Create x64 venv: arch -x86_64 /usr/local/bin/python3.12 -m venv venv_x64")
        print("  4. Activate and install: source venv_x64/bin/activate && pip install -e .")
        print("")
        print("OPTION 2: Force current shell to x64 mode")
        print("  Run: arch -x86_64 python3 <your_script.py>")
    else:
        print(f"? Unknown architecture: {machine}")

    # Try loading DLLs
    try_load_dll()

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
