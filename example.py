from RawFileExacter import RawFileReader
import polars as pl

if __name__ == "__main__":

    file_path = "example_path"

    raw_file_reader = RawFileReader(file_path)
    result = raw_file_reader.get_full_spectrum()
    print(f"memory usage: {result.estimated_size('mb')} MB")
    result = result.filter(pl.col("Intensity") > 0)
    print(f"memory usage: {result.estimated_size('mb')} MB")
    result = result.filter(pl.col("Intensity") > 10)
    print(f"memory usage: {result.estimated_size('mb')} MB")
    result = result.filter(pl.col("Intensity") > 100)
    print(f"memory usage: {result.estimated_size('mb')} MB")

    result.write_csv("example/path")
