from RawFileExacter import RawFileReader
import polars as pl

if __name__ == "__main__":

    file_path = "Data/20241219_fiona_t13_neg_01.raw"

    raw_file_reader = RawFileReader(file_path)

    raw_file_reader.write_mzml("example_output.mzML", filter_threshold=100)
    # result = raw_file_reader.get_full_spectrum()
    # print(f"memory usage: {result.estimated_size('mb')} MB")
    # result = result.filter(pl.col("Intensity") > 0)
    # print(f"memory usage: {result.estimated_size('mb')} MB")
    # result = result.filter(pl.col("Intensity") > 10)
    # print(f"memory usage: {result.estimated_size('mb')} MB")
    # result = result.filter(pl.col("Intensity") > 100)
    # print(f"memory usage: {result.estimated_size('mb')} MB")
