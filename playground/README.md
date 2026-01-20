# ARM64 Mac Compatibility - Findings and Solution

## Problem

The Thermo Fisher RawFileReader DLLs are compiled for **x64 architecture only**. On ARM64 Mac (Apple Silicon), using native ARM64 Python results in:

```
RuntimeError: This package requires x64 architecture. Current architecture: arm64
```

Even after bypassing the architecture check in `__init__.py`, the DLLs fail to load:
```
Could not load file or assembly 'ThermoFisher.CommonCore.RawFileReader...'
```

## Root Cause

| Component | Architecture |
|-----------|-------------|
| Thermo DLLs | x64 (AMD64) |
| ARM64 Python | arm64 |
| ARM64 .NET | arm64 |

The x64 DLLs cannot be loaded by an ARM64 Python process, even with Rosetta 2 available at the system level.

## Solution

**Use x64 Python running under Rosetta 2**

The `.venv` in this project already has x64 Python installed via UV:

```bash
# The .venv Python is x86_64
$ file .venv/bin/python
.venv/bin/python: Mach-O 64-bit executable x86_64

# Use it directly:
.venv/bin/python your_script.py

# Or activate the venv:
source .venv/bin/activate
python your_script.py
```

## Why This Works

When you run x64 Python on ARM64 Mac:
1. macOS automatically invokes Rosetta 2 to translate x64 instructions
2. Python reports `platform.machine()` as `x86_64`
3. pythonnet loads the x64 DLLs successfully
4. All Thermo API calls work correctly

## Test Results

Using `.venv/bin/python`:

```
✓ pythonnet imported successfully
✓ coreclr loaded successfully
✓ ThermoFisher.CommonCore.Data loaded successfully
✓ ThermoFisher.CommonCore.RawFileReader loaded successfully
✓ ThermoFisher.CommonCore.BackgroundSubtraction loaded successfully
✓ ThermoFisher.CommonCore.MassPrecisionEstimator loaded successfully
✓ RawFileReaderAdapter imported successfully
✓ RAW file reading works (46,952 scans, 25,663 data points per scan)
```

## Creating Your Own x64 Python Environment

If you need to create a new x64 Python environment:

### Option 1: Using UV (Recommended)
```bash
# UV automatically downloads the correct architecture
arch -x86_64 uv venv --python 3.12 venv_x64
source venv_x64/bin/activate
pip install -e .
```

### Option 2: Using Homebrew
```bash
# Install x64 Homebrew (if not present)
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install x64 Python
arch -x86_64 /usr/local/bin/brew install python@3.12

# Create venv
arch -x86_64 /usr/local/bin/python3.12 -m venv venv_x64
```

### Option 3: Using Conda/Miniforge
```bash
# Create x64 conda env
CONDA_SUBDIR=osx-64 conda create -n rawfilereader_x64 python=3.12
conda activate rawfilereader_x64
conda config --env --set subdir osx-64
pip install -e .
```

## Test Scripts

Two test scripts are provided in this folder:

1. **test_arm64_dll.py** - Diagnostic script that checks:
   - System architecture
   - Python architecture
   - .NET version and architecture
   - DLL architectures
   - Attempts to load DLLs and reports errors

2. **test_full_functionality.py** - Full functionality test:
   - Tests module import
   - Tests FileFactory creation
   - Tests reading an actual RAW file (if available)

Run them with:
```bash
# With ARM64 Python (will show the problem)
python3 playground/test_arm64_dll.py

# With x64 Python (will work)
.venv/bin/python playground/test_arm64_dll.py
.venv/bin/python playground/test_full_functionality.py
```

## Summary

| Python Architecture | Works? |
|---------------------|--------|
| ARM64 (native) | ❌ No |
| x64 (Rosetta 2) | ✅ Yes |

**The solution is simple: always use x64 Python on ARM64 Mac for this package.**
