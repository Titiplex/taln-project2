from pathlib import Path
from typing import Optional

from .config import N_SENTENCES, DATA_DIR
from .dataset import build_nlp, collect_sentences
from .io_utils import write_csv
from .extractors.entities import extract_named_entities
from .extractors.acronyms import extract_acronyms
from .extractors.taxonomy import extract_is_a
from .eval.sampling import sample_csv
from .eval.evaluation import eval_precision_from_gold, evaluate_acronym_consistency


def run_extraction(max_sentences: int = N_SENTENCES, out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = DATA_DIR

    print("Loading spaCy model...")
    nlp = build_nlp()

    print(f"Collecting up to {max_sentences} sentences from the dataset...")
    sentences = collect_sentences(max_sentences=max_sentences, nlp=nlp)
    print(f"Collected {len(sentences)} sentences.")

    print("Extracting named entities...")
    ent_rows = extract_named_entities(nlp, sentences)
    ent_path = out_dir / "entities.csv"
    write_csv(
        ent_path,
        ["sent_id", "entity", "label", "normalized", "sentence"],
        ent_rows,
    )
    print(f"Wrote {len(ent_rows)} entity rows to {ent_path}.")

    print("Extracting acronyms...")
    acr_rows = extract_acronyms(nlp, sentences)
    acr_path = out_dir / "acronyms.csv"
    write_csv(
        acr_path,
        ["sent_id", "acronym", "long_form", "sentence"],
        acr_rows,
    )
    print(f"Wrote {len(acr_rows)} acronym rows to {acr_path}.")

    print("Extracting IS_A relations...")
    tax_rows = extract_is_a(nlp, sentences)
    tax_path = out_dir / "is_a_relations.csv"
    write_csv(
        tax_path,
        ["sent_id", "hyponym", "relation", "hypernym", "sentence"],
        tax_rows,
    )
    print(f"Wrote {len(tax_rows)} IS_A relation rows to {tax_path}.")

    print(f"All done. CSV files are in: {out_dir}")


def create_samples_for_manual_annotation(
        out_dir: Optional[Path] = None,
        n_samples_entities: int = 150,
        n_samples_acronyms: int = 150,
        n_samples_is_a: int = 150,
) -> None:
    if out_dir is None:
        out_dir = DATA_DIR

    print(f"Creating samples for manual annotation in {out_dir}...")

    # Entities
    entities_in = out_dir / "entities.csv"
    entities_out = out_dir / "entities_sample_for_annot.csv"
    sample_csv(
        entities_in,
        entities_out,
        n_samples=n_samples_entities,
        extra_columns=("gold",),
    )
    print(f"Sampled entities -> {entities_out}")

    # Acronyms
    acr_in = out_dir / "acronyms.csv"
    acr_out = out_dir / "acronyms_sample_for_annot.csv"
    sample_csv(
        acr_in,
        acr_out,
        n_samples=n_samples_acronyms,
        extra_columns=("gold",),
    )
    print(f"Sampled acronyms -> {acr_out}")

    # IS_A relations
    isa_in = out_dir / "is_a_relations.csv"
    isa_out = out_dir / "is_a_relations_sample_for_annot.csv"
    sample_csv(
        isa_in,
        isa_out,
        n_samples=n_samples_is_a,
        extra_columns=("gold",),
    )
    print(f"Sampled IS_A relations -> {isa_out}")

    print("Now open those *_sample_for_annot.csv files and fill the 'gold' column (1/0).")


def run_evaluation_from_annot(out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = DATA_DIR

    print("Evaluating from annotated samples...")

    # entities
    ent_sample = out_dir / "entities_sample_for_annot.csv"
    ent_total, ent_true, ent_prec = eval_precision_from_gold(ent_sample)
    print(f"[ENTITIES] {ent_true}/{ent_total} corrects -> precision = {ent_prec:.3f}")

    # Acronyms
    acr_sample = out_dir / "acronyms_sample_for_annot.csv"
    acr_total, acr_true, acr_prec = eval_precision_from_gold(acr_sample)
    print(f"[ACRONYMS] {acr_true}/{acr_total} corrects -> precision = {acr_prec:.3f}")

    # Relations IS_A
    isa_sample = out_dir / "is_a_relations_sample_for_annot.csv"
    isa_total, isa_true, isa_prec = eval_precision_from_gold(isa_sample)
    print(f"[IS_A] {isa_true}/{isa_total} corrects -> precision = {isa_prec:.3f}")

    # internal consistency
    acr_full = out_dir / "acronyms.csv"
    consistency = evaluate_acronym_consistency(acr_full)
    print(f"[ACRONYMS] internal consistency (initials check) = {consistency:.3f}")
