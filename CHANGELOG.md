# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `extract_eic_multiple_files()` - Extract EIC from multiple RAW files in parallel
  - Uses ThreadPoolExecutor for concurrent file processing
  - Configurable number of workers (default: min(8, cpu_count))
  - Optional DataFrame output format
- `extract_eic_to_dataframe()` - Convenience function to extract EIC from multiple files and combine into a single DataFrame with 'File' column

### Changed

- **Performance**: `DotNetArrayToNPArray()` now uses `np.asarray` instead of `np.fromiter` (~11x faster)
  - `np.asarray` can directly access .NET buffer via pythonnet protocol
- **Performance**: `to_dataframe()` and `to_dataframe_parallel()` now convert lists to numpy arrays before DataFrame creation (~100x faster for DataFrame construction step)

### Fixed

- Removed dead code in `extract_eic()` function (unused variables)

### Performance

Benchmarks on 309 MB RAW file (46,951 scans, 16.7M data points):

| Method | Time | vs Sequential |
|--------|------|---------------|
| Python Sequential (`to_dataframe`) | 10.39s | 1.0x |
| C# Parallel + Pandas | 5.79s | 1.8x |
| C# Parallel + Polars | 4.97s | 2.1x |
| C# Parallel + Native Polars | 4.77s | 2.2x |

Multi-file EIC extraction (4 files, 5 m/z channels):

| Method | Time | Speedup |
|--------|------|---------|
| Sequential | 3.50s | 1.0x |
| Parallel (4 workers) | 0.63s | 5.6x |

## [0.1.0] - 2025-01-13

### Added

- Initial release
- `RawFileReader` class for reading Thermo RAW files
- `to_dataframe()`, `to_dataframe_parallel()`, `to_dataframe_fast()` methods
- `to_polars()` for native Polars DataFrame output
- `to_mzml()` and `to_mzml_parallel()` for mzML conversion
- `extract_eic()` and `extract_tic()` for chromatogram extraction
- `read_multiple_files()` for batch processing
- CLI tool for RAW to mzML conversion
- `EmptyRawFileError` exception for empty RAW files
