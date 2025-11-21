from pathlib import Path
import csv
from typing import Tuple


def eval_precision_from_gold(
        path: Path,
        gold_col: str = "gold",
) -> Tuple[int, int, float]:
    """
    compute precision from gold labels

    returns (nb_labelled, nb_true, precision).
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if gold_col not in reader.fieldnames:
            raise ValueError(f"Missing column '{gold_col}' in {path}")

        total = 0
        true_pos = 0
        for row in reader:
            val = row[gold_col].strip()
            if val == "":
                # ligne pas annotée -> ignore
                continue
            total += 1
            if val in {"1", "true", "True", "YES", "yes"}:
                true_pos += 1

    if total == 0:
        precision = 0.0
    else:
        precision = true_pos / total

    return total, true_pos, precision


def acronym_matches_long_form(acronym: str, long_form: str) -> bool:
    """
    verifies if acronym matches long form initials

    Example : 'EU' vs 'European Union' -> True.
    checks coherence
    """
    ac = acronym.replace(".", "").upper()
    if not ac:
        return False

    tokens = [w for w in long_form.split() if any(c.isalpha() for c in w)]
    if not tokens:
        return False

    initials = "".join(w[0].upper() for w in tokens if w)
    return ac == initials


def evaluate_acronym_consistency(path: Path) -> float:
    """
    reads outputs/acronyms.csv and returns line %
   where acronym matches long form
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if "acronym" not in reader.fieldnames or "long_form" not in reader.fieldnames:
            raise ValueError("csv must contain 'acronym' and 'long_form'")

        total = 0
        ok = 0
        for row in reader:
            acr = row["acronym"]
            long_form = row["long_form"]
            total += 1
            if acronym_matches_long_form(acr, long_form):
                ok += 1

    return ok / total if total > 0 else 0.0
