import polars as pl

from psims.mzml.writer import MzMLWriter


def write_mzml(df: pl.DataFrame, output_file: str):
    with MzMLWriter(open(output_file, 'wb'), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        # Open the run and spectrum list sections
        with out.run(id="my_analysis"):
            spectrum_count = len(df)
            with out.spectrum_list(count=spectrum_count):
                for row in df.rows():
                    # Write Precursor scan
                    out.write_spectrum(
                        row[1], row[2],
                        id=row[0], params=[
                            "MS1 Spectrum",
                            {"ms level": row[3]},
                            {"total ion current": sum(row[2])}
                        ])
                    # Write MSn scans
                    if row[3] == 2:
                        out.write_spectrum(
                            row[1], row[2],
                            id=row[0], params=[
                                "MSn Spectrum",
                                {"ms level": 2},
                                {"total ion current": sum(row[2])}
                            ],
                            # Include precursor information
                            precursor_information={
                                "mz": row[1],
                                "intensity": row[2],
                                "charge": 1,
                                "scan_id": row[0],
                                "activation": ["beam-type collisional dissociation", {"collision energy": 25}],
                                "isolation_window": [row[1] - 1, row[1], row[1] + 1]
                            })
