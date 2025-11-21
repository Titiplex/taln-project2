import csv
from pathlib import Path
from typing import Iterable, Sequence


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """
    Writes rows (iterables of sentences) to a csv file with a semicolon separator.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
