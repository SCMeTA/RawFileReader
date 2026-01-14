import platform
import sys
import subprocess


def check_dotnet():
    try:
        result = subprocess.run(
            ['dotnet', '--version'],
            capture_output=True,
            text=True
        )
        version = result.stdout.strip()
        major_version = int(version.split('.')[0])
        if major_version < 8:
            raise RuntimeError(f"Requires .NET 8.0 or higher. Found: {version}")
    except FileNotFoundError:
        raise RuntimeError(
            ".NET 8.0 runtime is required but not found. "
            "Please install from https://dotnet.microsoft.com/download"
        )

check_dotnet()
if platform.machine().lower() not in ['x86_64', 'amd64', 'x64']:
    raise RuntimeError(
        f"This package requires x64 architecture. "
        f"Current architecture: {platform.machine()}"
    )

from .reader import RawFileReader, read_multiple_files
from .cli import convert_raw_to_mzml, convert_folder_to_mzml, cli

__all__ = ['RawFileReader', 'read_multiple_files', 'convert_raw_to_mzml', 'convert_folder_to_mzml', 'cli']

