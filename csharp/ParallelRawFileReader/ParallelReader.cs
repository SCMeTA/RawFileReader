using System.Collections.Concurrent;
using ThermoFisher.CommonCore.Data;
using ThermoFisher.CommonCore.Data.Business;
using ThermoFisher.CommonCore.Data.FilterEnums;
using ThermoFisher.CommonCore.Data.Interfaces;
using ThermoFisher.CommonCore.RawFileReader;

namespace ParallelRawFileReader;

/// <summary>
/// Scan data structure for returning results to Python
/// </summary>
public class ScanData
{
    public int ScanNumber { get; set; }
    public double RetentionTime { get; set; }
    public int MsOrder { get; set; }
    public double[] Masses { get; set; } = Array.Empty<double>();
    public double[] Intensities { get; set; } = Array.Empty<double>();
    public string Polarity { get; set; } = "";
    public bool IsCentroid { get; set; }
    public double? PrecursorMz { get; set; }
    public double? IsolationWidth { get; set; }
    public double? CollisionEnergy { get; set; }
}

/// <summary>
/// Bulk scan result for efficient data transfer to Python
/// </summary>
public class BulkScanResult
{
    public int[] ScanNumbers { get; set; } = Array.Empty<int>();
    public double[] RetentionTimes { get; set; } = Array.Empty<double>();
    public int[] MsOrders { get; set; } = Array.Empty<int>();
    public int[] Polarities { get; set; } = Array.Empty<int>(); // 1 = positive, -1 = negative
    public double[] PrecursorMzs { get; set; } = Array.Empty<double>();
    public double[] IsolationWidths { get; set; } = Array.Empty<double>();
    public double[] CollisionEnergies { get; set; } = Array.Empty<double>();

    // Flattened arrays for all masses/intensities
    public double[] AllMasses { get; set; } = Array.Empty<double>();
    public double[] AllIntensities { get; set; } = Array.Empty<double>();

    // Index arrays to reconstruct per-scan data
    public int[] ScanStartIndices { get; set; } = Array.Empty<int>();
    public int[] ScanLengths { get; set; } = Array.Empty<int>();

    public int TotalScans { get; set; }
    public long TotalDataPoints { get; set; }
}

/// <summary>
/// Parallel RAW file reader using ThermoFisher ThreadManager for true multi-threaded access.
/// This bypasses Python's GIL limitation by doing all parallel work in native .NET code.
/// </summary>
public class ParallelReader : IDisposable
{
    private readonly string _filePath;
    private IRawDataPlus? _rawFile;
    private bool _disposed;

    public int FirstScan { get; private set; }
    public int LastScan { get; private set; }
    public string InstrumentModel { get; private set; } = "";

    public ParallelReader(string filePath)
    {
        _filePath = filePath;
        OpenFile();
    }

    private void OpenFile()
    {
        _rawFile = RawFileReaderAdapter.FileFactory(_filePath);

        if (!_rawFile.IsOpen)
            throw new InvalidOperationException($"Failed to open RAW file: {_filePath}");

        if (_rawFile.IsError)
            throw new InvalidOperationException($"Error opening RAW file: {_rawFile.FileError}");

        _rawFile.SelectInstrument(Device.MS, 1);

        FirstScan = _rawFile.RunHeaderEx.FirstSpectrum;
        LastScan = _rawFile.RunHeaderEx.LastSpectrum;
        InstrumentModel = _rawFile.GetInstrumentData().Model;
    }

    /// <summary>
    /// Read all scans in parallel using the ThreadManager API.
    /// Returns a BulkScanResult with flattened arrays for efficient Python interop.
    /// </summary>
    /// <param name="includeMs2">Include MS2 scans</param>
    /// <param name="filterThreshold">Filter intensities below this value (0 = no filter)</param>
    /// <param name="maxWorkers">Number of parallel workers (0 = auto)</param>
    public BulkScanResult ReadAllScansParallel(bool includeMs2 = false, double filterThreshold = 0, int maxWorkers = 0)
    {
        if (maxWorkers <= 0)
            maxWorkers = Math.Min(8, Environment.ProcessorCount);

        // Use ThreadManager for lockless parallel access
        using var threadManager = RawFileReaderFactory.CreateThreadManager(_filePath);

        var scanNumbers = Enumerable.Range(FirstScan, LastScan - FirstScan + 1).ToArray();
        var scanResults = new ConcurrentDictionary<int, ScanData>();

        // Process scans in parallel using native .NET parallelism
        var options = new ParallelOptions { MaxDegreeOfParallelism = maxWorkers };

        Parallel.ForEach(scanNumbers, options, scanNumber =>
        {
            // Each thread gets its own accessor (lockless parallel access)
            var accessor = threadManager.CreateThreadAccessor();
            accessor.SelectInstrument(Device.MS, 1);

            var scanData = ReadSingleScan(accessor, scanNumber, includeMs2, filterThreshold);
            if (scanData != null)
            {
                scanResults[scanNumber] = scanData;
            }
        });

        // Convert to BulkScanResult with flattened arrays
        return ConvertToBulkResult(scanResults, scanNumbers);
    }

    /// <summary>
    /// Read scans sequentially (for comparison/benchmarking)
    /// </summary>
    public BulkScanResult ReadAllScansSequential(bool includeMs2 = false, double filterThreshold = 0)
    {
        if (_rawFile == null)
            throw new InvalidOperationException("File not open");

        var scanNumbers = Enumerable.Range(FirstScan, LastScan - FirstScan + 1).ToArray();
        var scanResults = new ConcurrentDictionary<int, ScanData>();

        foreach (var scanNumber in scanNumbers)
        {
            var scanData = ReadSingleScan(_rawFile, scanNumber, includeMs2, filterThreshold);
            if (scanData != null)
            {
                scanResults[scanNumber] = scanData;
            }
        }

        return ConvertToBulkResult(scanResults, scanNumbers);
    }

    private static ScanData? ReadSingleScan(IRawDataPlus rawFile, int scanNumber, bool includeMs2, double filterThreshold)
    {
        try
        {
            var scanStats = rawFile.GetScanStatsForScanNumber(scanNumber);
            var scanFilter = rawFile.GetFilterForScanNumber(scanNumber);

            int msOrder = scanFilter.MSOrder == MSOrderType.Ms ? 1 : 2;

            // Skip MS2 if not requested
            if (!includeMs2 && msOrder == 2)
                return null;

            string polarity = scanFilter.Polarity == PolarityType.Positive ? "positive scan" : "negative scan";
            double retentionTime = rawFile.RetentionTimeFromScanNumber(scanNumber);

            double[] masses;
            double[] intensities;
            bool isCentroid;

            if (scanStats.IsCentroidScan)
            {
                var centroidStream = rawFile.GetCentroidStream(scanNumber, false);
                masses = centroidStream.Masses ?? Array.Empty<double>();
                intensities = centroidStream.Intensities ?? Array.Empty<double>();
                isCentroid = true;
            }
            else
            {
                var segmentedScan = rawFile.GetSegmentedScanFromScanNumber(scanNumber, scanStats);
                masses = segmentedScan.Positions ?? Array.Empty<double>();
                intensities = segmentedScan.Intensities ?? Array.Empty<double>();
                isCentroid = false;
            }

            // Apply intensity filter
            if (filterThreshold > 0 && masses.Length > 0)
            {
                var filtered = masses.Zip(intensities, (m, i) => (m, i))
                    .Where(x => x.i > filterThreshold)
                    .ToArray();
                masses = filtered.Select(x => Math.Round(x.m, 6)).ToArray();
                intensities = filtered.Select(x => Math.Round(x.i, 2)).ToArray();
            }

            // Get precursor info for MS2
            double? precursorMz = null;
            double? isolationWidth = null;
            double? collisionEnergy = null;

            if (msOrder > 1)
            {
                var scanEvent = rawFile.GetScanEventForScanNumber(scanNumber);
                if (scanEvent != null)
                {
                    var reaction = scanEvent.GetReaction(0);
                    precursorMz = reaction.PrecursorMass;
                    isolationWidth = reaction.IsolationWidth;
                    collisionEnergy = double.IsNaN(reaction.CollisionEnergy) ? null : reaction.CollisionEnergy;
                }
            }

            return new ScanData
            {
                ScanNumber = scanNumber,
                RetentionTime = retentionTime,
                MsOrder = msOrder,
                Masses = masses,
                Intensities = intensities,
                Polarity = polarity,
                IsCentroid = isCentroid,
                PrecursorMz = precursorMz,
                IsolationWidth = isolationWidth,
                CollisionEnergy = collisionEnergy
            };
        }
        catch
        {
            return null;
        }
    }

    private static BulkScanResult ConvertToBulkResult(ConcurrentDictionary<int, ScanData> scanResults, int[] allScanNumbers)
    {
        // Sort by scan number
        var sortedScans = allScanNumbers
            .Where(scanResults.ContainsKey)
            .OrderBy(x => x)
            .Select(x => scanResults[x])
            .ToList();

        int totalScans = sortedScans.Count;
        long totalDataPoints = sortedScans.Sum(s => s.Masses.Length);

        var result = new BulkScanResult
        {
            TotalScans = totalScans,
            TotalDataPoints = totalDataPoints,
            ScanNumbers = new int[totalScans],
            RetentionTimes = new double[totalScans],
            MsOrders = new int[totalScans],
            Polarities = new int[totalScans],
            PrecursorMzs = new double[totalScans],
            IsolationWidths = new double[totalScans],
            CollisionEnergies = new double[totalScans],
            ScanStartIndices = new int[totalScans],
            ScanLengths = new int[totalScans],
            AllMasses = new double[totalDataPoints],
            AllIntensities = new double[totalDataPoints]
        };

        int dataIndex = 0;
        for (int i = 0; i < totalScans; i++)
        {
            var scan = sortedScans[i];
            result.ScanNumbers[i] = scan.ScanNumber;
            result.RetentionTimes[i] = Math.Round(scan.RetentionTime, 3);
            result.MsOrders[i] = scan.MsOrder;
            result.Polarities[i] = scan.Polarity == "positive scan" ? 1 : -1;
            result.PrecursorMzs[i] = scan.PrecursorMz ?? double.NaN;
            result.IsolationWidths[i] = scan.IsolationWidth ?? double.NaN;
            result.CollisionEnergies[i] = scan.CollisionEnergy ?? double.NaN;
            result.ScanStartIndices[i] = dataIndex;
            result.ScanLengths[i] = scan.Masses.Length;

            // Copy mass/intensity data
            Array.Copy(scan.Masses, 0, result.AllMasses, dataIndex, scan.Masses.Length);
            Array.Copy(scan.Intensities, 0, result.AllIntensities, dataIndex, scan.Masses.Length);
            dataIndex += scan.Masses.Length;
        }

        return result;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _rawFile?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Result for a single file in multi-file processing
/// </summary>
public class FileResult
{
    public string FilePath { get; set; } = "";
    public string FileName { get; set; } = "";
    public BulkScanResult? Data { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public double ProcessingTimeSeconds { get; set; }
}

/// <summary>
/// Result for multi-file parallel processing
/// </summary>
public class MultiFileResult
{
    public FileResult[] FileResults { get; set; } = Array.Empty<FileResult>();
    public int TotalFiles { get; set; }
    public int SuccessfulFiles { get; set; }
    public int FailedFiles { get; set; }
    public double TotalProcessingTimeSeconds { get; set; }
}

/// <summary>
/// Multi-file parallel reader for batch processing multiple RAW files simultaneously.
/// Supports two levels of parallelism:
/// 1. File-level: Multiple files processed concurrently
/// 2. Scan-level: Within each file, scans can be read in parallel
/// </summary>
public static class MultiFileReader
{
    /// <summary>
    /// Read multiple RAW files in parallel.
    /// </summary>
    /// <param name="filePaths">Array of file paths to process</param>
    /// <param name="includeMs2">Include MS2 scans</param>
    /// <param name="filterThreshold">Filter intensities below this value</param>
    /// <param name="maxFilesParallel">Max files to process concurrently (0 = auto)</param>
    /// <param name="maxScansParallel">Max parallel workers per file for scan reading (0 = auto)</param>
    /// <returns>MultiFileResult containing results for all files</returns>
    public static MultiFileResult ReadMultipleFiles(
        string[] filePaths,
        bool includeMs2 = false,
        double filterThreshold = 0,
        int maxFilesParallel = 0,
        int maxScansParallel = 0)
    {
        if (maxFilesParallel <= 0)
            maxFilesParallel = Math.Min(4, Environment.ProcessorCount / 2);

        var startTime = System.Diagnostics.Stopwatch.StartNew();
        var fileResults = new ConcurrentBag<FileResult>();

        var options = new ParallelOptions { MaxDegreeOfParallelism = maxFilesParallel };

        Parallel.ForEach(filePaths, options, filePath =>
        {
            var fileStart = System.Diagnostics.Stopwatch.StartNew();
            var result = new FileResult
            {
                FilePath = filePath,
                FileName = Path.GetFileName(filePath)
            };

            try
            {
                using var reader = new ParallelReader(filePath);
                result.Data = reader.ReadAllScansParallel(includeMs2, filterThreshold, maxScansParallel);
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }

            fileStart.Stop();
            result.ProcessingTimeSeconds = fileStart.Elapsed.TotalSeconds;
            fileResults.Add(result);
        });

        startTime.Stop();

        var resultsArray = fileResults.ToArray();
        return new MultiFileResult
        {
            FileResults = resultsArray,
            TotalFiles = filePaths.Length,
            SuccessfulFiles = resultsArray.Count(r => r.Success),
            FailedFiles = resultsArray.Count(r => !r.Success),
            TotalProcessingTimeSeconds = startTime.Elapsed.TotalSeconds
        };
    }

    /// <summary>
    /// Read multiple RAW files sequentially (for benchmarking comparison).
    /// Each file still uses parallel scan reading internally.
    /// </summary>
    public static MultiFileResult ReadMultipleFilesSequential(
        string[] filePaths,
        bool includeMs2 = false,
        double filterThreshold = 0,
        int maxScansParallel = 0)
    {
        var startTime = System.Diagnostics.Stopwatch.StartNew();
        var fileResults = new List<FileResult>();

        foreach (var filePath in filePaths)
        {
            var fileStart = System.Diagnostics.Stopwatch.StartNew();
            var result = new FileResult
            {
                FilePath = filePath,
                FileName = Path.GetFileName(filePath)
            };

            try
            {
                using var reader = new ParallelReader(filePath);
                result.Data = reader.ReadAllScansParallel(includeMs2, filterThreshold, maxScansParallel);
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }

            fileStart.Stop();
            result.ProcessingTimeSeconds = fileStart.Elapsed.TotalSeconds;
            fileResults.Add(result);
        }

        startTime.Stop();

        return new MultiFileResult
        {
            FileResults = fileResults.ToArray(),
            TotalFiles = filePaths.Length,
            SuccessfulFiles = fileResults.Count(r => r.Success),
            FailedFiles = fileResults.Count(r => !r.Success),
            TotalProcessingTimeSeconds = startTime.Elapsed.TotalSeconds
        };
    }
}
