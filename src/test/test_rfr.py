from RawFileReader import RawFileReader

def test_convert_raw_to_mzml():
    input_path = "tests/data/sample.raw"
    output_path = "tests/data/sample_converted.mzML"
    raw_file_reader = RawFileReader(input_path)
    raw_file_reader.to_mzml(output_path, include_ms2=False, filter_threshold=None)
    # Check if the output file is created
    import os
    assert os.path.exists(output_path)
    # Clean up