from pathlib import Path
import csv
import random
from typing import Sequence


def sample_csv(
        input_path: Path,
        output_path: Path,
        n_samples: int = 100,
        extra_columns: Sequence[str] = ("gold",),
) -> None:
    """
    pulls a sample of n_samples rows from a CSV file, adding extra columns
    ex : gold for manuel labelling
    """
    with input_path.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter=";"))

    if not reader:
        raise ValueError(f"Empty CSV: {input_path}")

    header = reader[0]
    rows = reader[1:]

    if len(rows) <= n_samples:
        sampled = rows
    else:
        sampled = random.sample(rows, n_samples)

    new_header = list(header) + list(extra_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(new_header)
        for row in sampled:
            writer.writerow(list(row) + ["" for _ in extra_columns])
