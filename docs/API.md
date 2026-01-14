# RawFileReader API Documentation

A Python package for reading Thermo Fisher Scientific RAW files using the official RawFileReader .NET library with high-performance parallel processing capabilities.

## Requirements

- Python >= 3.12
- .NET 8.0 Runtime
- x64 architecture

## Installation

```bash
pip install RawFileReader
```

---

## Quick Start

```python
from RawFileReader import RawFileReader, read_multiple_files

# Single file
reader = RawFileReader("sample.raw")
df = reader.to_dataframe_fast()  # Fastest method

# Multiple files in parallel
files = ["file1.raw", "file2.raw", "file3.raw"]
results = read_multiple_files(files)
```

---

## Class: RawFileReader

Main class for reading Thermo RAW files.

### Constructor

```python
RawFileReader(file_path: str | Path)
```

**Parameters:**
- `file_path`: Path to the RAW file

**Raises:**
- `RawFileNotOpenError`: If the file cannot be opened

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str \| Path` | Path to the RAW file |
| `file_name` | `str` | File name without extension |
| `scan_range` | `list[int]` | `[first_scan, last_scan]` |
| `max_retention_time` | `float` | Maximum retention time (minutes) |
| `instrument_info` | `dict` | Instrument metadata |

**`instrument_info` dictionary:**
```python
{
    "raw_file_version": int,
    "number_of_instruments": int,
    "instrument_name": str,
    "instrument_model": str,
    "instrument_serial_number": str,
    "mass_resolution": float
}
```

---

## Methods

### get_spectrum

Get spectrum data for a single scan.

```python
def get_spectrum(
    scan_number: int,
    include_ms2: bool = False
) -> tuple | None
```

**Parameters:**
- `scan_number`: Scan number to retrieve
- `include_ms2`: Include MS2 scans (default: False)

**Returns:**
- `tuple`: `(retention_time, ms_order, masses, intensities, polarity, is_centroid, precursor)`
- `None`: If scan is MS2 and `include_ms2=False`

**Example:**
```python
reader = RawFileReader("sample.raw")
result = reader.get_spectrum(100, include_ms2=True)
if result:
    rt, ms_order, masses, intensities, polarity, is_centroid, precursor = result
```

---

### to_dataframe_fast

**Recommended** - Convert all scans to DataFrame using C# native parallel processing.

```python
def to_dataframe_fast(
    include_ms2: bool = False,
    filter_threshold: float = 0,
    max_workers: int = 0,
    use_polars: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity, 0 = no filter (default: 0)
- `max_workers`: Number of parallel workers, 0 = auto (default: 0)
- `use_polars`: Use Polars for faster DataFrame construction (default: True)

**Returns:**
- `pd.DataFrame`: DataFrame with columns:
  - `Scan`: Scan number (int)
  - `RetentionTime`: Retention time in minutes (float)
  - `MS Order`: MS level, 1 or 2 (int)
  - `Mass`: m/z value (float)
  - `Intensity`: Intensity value (float)
  - `Polarity`: 1 = positive, -1 = negative (int)
  - `PrecursorMz`: Precursor m/z for MS2 (float, NaN for MS1)
  - `IsolationWidth`: Isolation width for MS2 (float, NaN for MS1)
  - `CollisionEnergy`: Collision energy for MS2 (float, NaN for MS1)

**Performance:** ~6x faster than `to_dataframe()`

**Example:**
```python
reader = RawFileReader("sample.raw")
df = reader.to_dataframe_fast(include_ms2=True, max_workers=4)
print(df.shape)  # (millions_of_rows, 9)
```

---

### to_polars

Convert all scans to native Polars DataFrame.

```python
def to_polars(
    include_ms2: bool = False,
    filter_threshold: float = 0,
    max_workers: int = 0
) -> pl.DataFrame
```

**Parameters:**
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity (default: 0)
- `max_workers`: Number of parallel workers (default: 0 = auto)

**Returns:**
- `pl.DataFrame`: Polars DataFrame (same columns as `to_dataframe_fast`)

**Raises:**
- `ImportError`: If Polars is not installed

**Example:**
```python
reader = RawFileReader("sample.raw")
df = reader.to_polars()  # Native Polars, no pandas conversion
```

---

### to_dataframe

Convert all scans to DataFrame (sequential processing).

```python
def to_dataframe(
    include_ms2: bool = False,
    filter_threshold: int | None = None
) -> pd.DataFrame
```

**Parameters:**
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity (default: None)

**Returns:**
- `pd.DataFrame`: DataFrame with scan data

**Note:** Use `to_dataframe_fast()` for better performance.

---

### to_mzml

Convert RAW file to mzML format.

```python
def to_mzml(
    output_path: str | Path,
    include_ms2: bool = False,
    filter_threshold: int | None = None
) -> None
```

**Parameters:**
- `output_path`: Output mzML file path
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity (default: None)

**Example:**
```python
reader = RawFileReader("sample.raw")
reader.to_mzml("output/sample.mzML", include_ms2=True)
```

---

### to_mzml_parallel

Convert RAW file to mzML using parallel scan reading.

```python
def to_mzml_parallel(
    output_path: str | Path,
    include_ms2: bool = False,
    filter_threshold: int | None = None,
    max_workers: int | None = None
) -> None
```

**Parameters:**
- `output_path`: Output mzML file path
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity (default: None)
- `max_workers`: Number of parallel workers (default: auto)

---

### extract_tic

Extract Total Ion Chromatogram (TIC).

```python
def extract_tic(
    start_scan: int = -1,
    end_scan: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Parameters:**
- `start_scan`: Start scan number, -1 for first scan (default: -1)
- `end_scan`: End scan number, -1 for last scan (default: -1)

**Returns:**
- `tuple`: `(scan_numbers, retention_times, intensities)`

**Example:**
```python
reader = RawFileReader("sample.raw")
scans, rts, tic = reader.extract_tic()

import matplotlib.pyplot as plt
plt.plot(rts, tic)
plt.xlabel("Retention Time (min)")
plt.ylabel("Total Ion Current")
```

---

### extract_eic

Extract Extracted Ion Chromatogram (EIC) for specific m/z values.

```python
def extract_eic(
    mz: float | list[float],
    _tolerance: float = 5,
    start_scan: int = -1,
    end_scan: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Parameters:**
- `mz`: Target m/z value(s)
- `_tolerance`: Mass tolerance in ppm (default: 5)
- `start_scan`: Start scan number, -1 for first scan (default: -1)
- `end_scan`: End scan number, -1 for last scan (default: -1)

**Returns:**
- `tuple`: `(scan_numbers, retention_times, intensities)`
  - For multiple m/z values, `intensities` shape is `(n_scans, n_mz_values)`

**Example:**
```python
reader = RawFileReader("sample.raw")

# Single m/z
scans, rts, eic = reader.extract_eic(500.25, _tolerance=10)

# Multiple m/z values
scans, rts, eics = reader.extract_eic([500.25, 600.30, 700.35])
```

---

### to_numpy

Convert all scans to NumPy array.

```python
def to_numpy(
    include_ms2: bool = False,
    filter_threshold: int | None = None
) -> np.ndarray
```

**Returns:**
- `np.ndarray`: Array of shape `(n_scans, 3)` with `[masses, intensities, precursor]` per scan

---

### to_series

Get a single scan as DataFrame.

```python
def to_series(
    scan_number: int,
    include_ms2: bool = False,
    filter_threshold: int | None = None
) -> pd.DataFrame | None
```

**Returns:**
- `pd.DataFrame`: Single scan data
- `None`: If scan is MS2 and `include_ms2=False`

---

## Function: read_multiple_files

Read multiple RAW files in parallel using C# native multi-threading.

```python
def read_multiple_files(
    file_paths: list[str | Path],
    include_ms2: bool = False,
    filter_threshold: float = 0,
    max_files_parallel: int = 0,
    max_scans_parallel: int = 0,
    use_polars: bool = True,
    return_native_polars: bool = False
) -> dict[str, pd.DataFrame | pl.DataFrame]
```

**Parameters:**
- `file_paths`: List of RAW file paths
- `include_ms2`: Include MS2 spectra (default: False)
- `filter_threshold`: Filter peaks below this intensity (default: 0)
- `max_files_parallel`: Max files to process concurrently, 0 = auto (default: 0)
- `max_scans_parallel`: Max parallel workers per file, 0 = auto (default: 0)
- `use_polars`: Use Polars for DataFrame construction (default: True)
- `return_native_polars`: Return Polars DataFrames instead of pandas (default: False)

**Returns:**
- `dict`: Mapping of file paths to DataFrames. Failed files have `None` value.

**Example:**
```python
from RawFileReader import read_multiple_files
from pathlib import Path

# Find all RAW files
files = list(Path("data").glob("*.raw"))

# Process in parallel
results = read_multiple_files(files, max_files_parallel=2)

# Access results
for path, df in results.items():
    if df is not None:
        print(f"{Path(path).name}: {df.shape}")
```

---

## Command Line Interface

Convert RAW files to mzML format from command line.

```bash
python -m RawFileReader.cli INPUT_FOLDER OUTPUT_FOLDER [OPTIONS]
```

**Options:**
- `--include-ms2`: Include MS2 spectra in output
- `--filter-threshold INT`: Filter peaks below this intensity
- `--workers INT`: Number of parallel workers (default: auto)

**Example:**
```bash
# Convert all RAW files in a folder
python -m RawFileReader.cli ./raw_files ./mzml_output --include-ms2 --workers 4
```

---

## Performance Comparison

Benchmark results for a 414 MB RAW file (~600 scans, 16M data points):

| Method | Time | Speedup |
|--------|------|---------|
| `to_dataframe()` | 33.94s | 1.0x |
| `to_dataframe_fast(use_polars=False)` | 6.60s | 5.1x |
| `to_dataframe_fast(use_polars=True)` | 5.86s | 5.8x |
| `to_polars()` | 5.55s | **6.1x** |

### Multi-File Processing

Processing 9 files (3.7 GB total):

| Method | Time | Speedup |
|--------|------|---------|
| Sequential | 3.32s | 1.0x |
| Parallel (2 files) | 2.23s | **1.49x** |

---

## DataFrame Schema

All DataFrame methods return the same column structure:

| Column | Type | Description |
|--------|------|-------------|
| `Scan` | int32 | Scan number |
| `RetentionTime` | float64 | Retention time (minutes) |
| `MS Order` | int32 | MS level (1 or 2) |
| `Mass` | float64 | m/z value |
| `Intensity` | float64 | Intensity value |
| `Polarity` | int32 | 1 = positive, -1 = negative |
| `PrecursorMz` | float64 | Precursor m/z (NaN for MS1) |
| `IsolationWidth` | float64 | Isolation width (NaN for MS1) |
| `CollisionEnergy` | float64 | Collision energy (NaN for MS1) |

---

## Exceptions

### RawFileNotOpenError

Raised when a RAW file cannot be opened.

```python
from RawFileReader import RawFileReader

try:
    reader = RawFileReader("nonexistent.raw")
except Exception as e:
    print(f"Error: {e}")
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pythonnet | >=3.0.5 | .NET interop |
| numpy | >=2.3.4 | Array operations |
| pandas | >=2.3.3 | DataFrame support |
| polars | >=1.0.0 | Fast DataFrame construction |
| psims | >=1.3.5 | mzML writing |
| tqdm | >=4.67.1 | Progress bars |
| click | >=8.3.1 | CLI |
