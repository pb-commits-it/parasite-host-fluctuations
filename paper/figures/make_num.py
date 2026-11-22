"""Extract the nine original pilot cells at phi=0.6 without changing values.

The figure is historical: later corrected-pilot values belong to the text.
Run with python3 -I ../paper/figures/make_num.py.
"""

import csv
from pathlib import Path


def main():
    figures = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
    source = (__import__('pathlib').Path(__file__).resolve().parent / '../../code/virtualworld/results/pilot_cells.csv').resolve()
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        (float(row["sigma"]), float(row["stay"])): row
        for row in rows if float(row["phi"]) == 0.6
    }
    sigmas = ((0.0, "0"), (0.3, "03"), (0.6, "06"))
    stays = (1.0, 5.0, 20.0)
    assert set(selected) == {(s, t) for s, _ in sigmas for t in stays}
    # Published counts are the extraction benchmark, checked before writing.
    expected = {(0.0, 1.0): 67, (0.0, 5.0): 70, (0.0, 20.0): 36,
                (0.3, 1.0): 138, (0.3, 5.0): 102, (0.3, 20.0): 53,
                (0.6, 1.0): 194, (0.6, 5.0): 157, (0.6, 20.0): 77}
    for key, row in selected.items():
        assert int(row["n"]) == 200
        assert int(row["extinct"]) == expected[key]
        assert abs(float(row["extinction"]) - int(row["extinct"]) / 200) < 1e-15
    fields = ["stay", "phi"]
    mapping = {"n": "n", "extinct": "extinct", "p": "extinction",
               "low": "extinction_low", "high": "extinction_high"}
    fields += [f"{name}{suffix}" for _, suffix in sigmas for name in mapping]
    destination = figures / "data/num-1.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stay in stays:
            row = {"stay": selected[(0.0, stay)]["stay"], "phi": "0.6"}
            for sigma, suffix in sigmas:
                for name, original in mapping.items():
                    row[f"{name}{suffix}"] = selected[(sigma, stay)][original]
            writer.writerow(row)
    print("Extracted 9/9 cells; original counts, denominators, and intervals retained.")


if __name__ == "__main__":
    main()
