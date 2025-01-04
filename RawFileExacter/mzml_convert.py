import pandas as pd

from psims.mzml.writer import MzMLWriter


def write_mzml(df: pd.DataFrame, output_file: str):
    with MzMLWriter(open(output_file, 'wb'), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        # Open the run and spectrum list sections
        with out.run(id="my_analysis"):
            spectrum_count = len(df)
            with out.spectrum_list(count=spectrum_count):
                for index, row in df.iterrows():
                    # Write Precursor scan
                    out.write_spectrum(
                        row["Mass"], row["Intensity"],
                        id=row["Scan"], params=[
                            "MS1 Spectrum",
                            {"ms level": row["MS Order"]},
                            {"total ion current": sum(row["Intensity"])}
                         ])
                    # Write MSn scans
                    if row["MS Order"] == 2:
                        out.write_spectrum(
                            row["Mass"], row["Intensity"],
                            id=row["Scan"], params=[
                                "MSn Spectrum",
                                {"ms level": 2},
                                {"total ion current": sum(row["Intensity"])}
                             ],
                             # Include precursor information
                             precursor_information={
                                "mz": row["Mass"],
                                "intensity": row["Intensity"],
                                "charge": 1,
                                "scan_id": row["Scan"],
                                "activation": ["beam-type collisional dissociation", {"collision energy": 25}],
                                "isolation_window": [row["Mass"] - 1, row["Mass"], row["Mass"] + 1]
                             })