"""Process historical viewing data and save matches and data-quality statistics.

Run from the project root:
    python -B exam/process_matching.py

Uses the in-memory matcher separately for training and validation. Original CSVs
are opened read-only. An existing output directory is never overwritten.
"""
import csv
import json
import time
from contextlib import ExitStack
from dataclasses import asdict, fields
from datetime import datetime
from itertools import chain
from pathlib import Path

if __package__:
    from .matching import BroadcastMatch, QuarantinedViewing, match_viewings
else:
    from matching import BroadcastMatch, QuarantinedViewing, match_viewings


def save_csv(path, records, record_type):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[field.name for field in fields(record_type)]
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            if "reasons" in row:
                row["reasons"] = "|".join(row["reasons"])
            writer.writerow(row)


def main():
    exam_directory = Path(__file__).resolve().parent
    source = exam_directory / "dataset-sweet-tv"
    output = exam_directory / "derived" / "matching"
    required_files = (
        "dataset11-30.csv",
        "export_arh_11-20-final.csv",
        "export_arh_21-30-final.csv",
    )
    for filename in required_files:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Missing input: {source / filename}")
    output.mkdir(parents=True, exist_ok=False)

    statistics = {
        "counter_note": (
            "Source-quality checks precede window clipping. Their counts repeat "
            "across partitions and must not be summed."
        ),
        "partitions": {},
    }
    windows = [
        ("train", datetime(2020, 3, 9), datetime(2020, 5, 18)),
        ("validation", datetime(2020, 5, 18), datetime(2020, 7, 27)),
    ]

    for name, start, end in windows:
        print(f"Processing {name}: {start} to {end}", flush=True)
        started = time.perf_counter()
        with ExitStack() as stack:
            def reader(filename):
                stream = stack.enter_context(
                    (source / filename).open(encoding="utf-8", newline="")
                )
                return csv.DictReader(stream)

            result = match_viewings(
                reader("dataset11-30.csv"),
                chain(
                    reader("export_arh_11-20-final.csv"),
                    reader("export_arh_21-30-final.csv"),
                ),
                window_start=start,
                window_end=end,
            )

        statistics["partitions"][name] = {
            "window_start_inclusive": start.isoformat(),
            "window_end_exclusive": end.isoformat(),
            "matching_seconds": round(time.perf_counter() - started, 3),
            "audit": result.audit,
        }
        (output / "data-quality.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
        save_csv(output / f"{name}-matches.csv", result.matches, BroadcastMatch)
        save_csv(
            output / f"{name}-quarantined.csv",
            result.quarantined,
            QuarantinedViewing,
        )
        print(json.dumps(statistics["partitions"][name], indent=2), flush=True)
        del result

    print(f"Saved outputs to {output}")


if __name__ == "__main__":
    main()
