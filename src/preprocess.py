"""
Preprocess MIMIC-III data for contrastive embedding training.

Builds temporal note pairs (anchor at time t, positive at time t+1)
grouped by patient (subject_id), with ICD diagnosis labels for
hierarchical contrastive learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MIMIC_DIR = Path(__file__).resolve().parent.parent / "mimic-iii-clinical-database-demo-1.4"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def load_noteevents(mimic_dir: Path) -> pd.DataFrame:
    """Load and clean NOTEEVENTS, sorted by patient and time."""
    df = pd.read_csv(mimic_dir / "NOTEEVENTS.csv", parse_dates=["chartdate", "charttime"])
    df = df[df["iserror"] != 1].copy()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 50]
    df = df.sort_values(["subject_id", "chartdate", "charttime"]).reset_index(drop=True)
    return df


def load_diagnoses(mimic_dir: Path) -> pd.DataFrame:
    """Load ICD-9 diagnoses with descriptions."""
    diag = pd.read_csv(mimic_dir / "DIAGNOSES_ICD.csv")
    d_icd = pd.read_csv(mimic_dir / "D_ICD_DIAGNOSES.csv")
    diag = diag.merge(d_icd[["icd9_code", "short_title", "long_title"]], on="icd9_code", how="left")
    return diag


def load_admissions(mimic_dir: Path) -> pd.DataFrame:
    """Load admissions data."""
    return pd.read_csv(mimic_dir / "ADMISSIONS.csv", parse_dates=["admittime", "dischtime"])


def load_patients(mimic_dir: Path) -> pd.DataFrame:
    """Load patient demographics."""
    return pd.read_csv(mimic_dir / "PATIENTS.csv", parse_dates=["dob", "dod"])


def build_temporal_pairs(notes: pd.DataFrame) -> list[dict]:
    """
    Build (anchor, positive) note pairs from sequential notes per patient.

    For each patient, consecutive notes form a pair:
      anchor = note at time t
      positive = note at time t+1
    """
    pairs = []
    for subject_id, group in notes.groupby("subject_id"):
        group = group.sort_values(["chartdate", "charttime"]).reset_index(drop=True)
        for i in range(len(group) - 1):
            pairs.append({
                "subject_id": int(subject_id),
                "anchor_text": group.loc[i, "text"],
                "positive_text": group.loc[i + 1, "text"],
                "anchor_hadm_id": int(group.loc[i, "hadm_id"]) if pd.notna(group.loc[i, "hadm_id"]) else None,
                "positive_hadm_id": int(group.loc[i + 1, "hadm_id"]) if pd.notna(group.loc[i + 1, "hadm_id"]) else None,
                "anchor_category": group.loc[i, "category"],
                "positive_category": group.loc[i + 1, "category"],
                "anchor_date": str(group.loc[i, "chartdate"]),
                "positive_date": str(group.loc[i + 1, "chartdate"]),
            })
    return pairs


def build_icd_hierarchy(diagnoses: pd.DataFrame) -> dict[str, list[str]]:
    """
    Map hadm_id -> list of ICD-9 codes for that admission.
    Also extracts the ICD chapter (first 3 chars) for hierarchical grouping.
    """
    icd_map: dict[str, list[str]] = {}
    for hadm_id, group in diagnoses.groupby("hadm_id"):
        icd_map[str(int(hadm_id))] = group["icd9_code"].dropna().tolist()
    return icd_map


def build_note_level_dataset(
    notes: pd.DataFrame,
    diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a per-note dataset with ICD codes attached via hadm_id,
    suitable for embedding + downstream classification.
    """
    icd_per_admission = (
        diagnoses.groupby("hadm_id")["icd9_code"]
        .apply(list)
        .reset_index()
        .rename(columns={"icd9_code": "icd_codes"})
    )
    merged = notes.merge(icd_per_admission, on="hadm_id", how="left")
    merged["icd_codes"] = merged["icd_codes"].apply(lambda x: x if isinstance(x, list) else [])
    return merged


def get_icd_chapter(code: str) -> str:
    """Map ICD-9 code to its chapter for hierarchical grouping."""
    if not code or not isinstance(code, str):
        return "unknown"
    prefix = code[0]
    if prefix == "E":
        return "E_external_causes"
    if prefix == "V":
        return "V_supplementary"
    try:
        num = int(code[:3])
    except ValueError:
        return "unknown"
    chapters = [
        (139, "001-139_infectious"),
        (239, "140-239_neoplasms"),
        (279, "240-279_endocrine"),
        (289, "280-289_blood"),
        (319, "290-319_mental"),
        (389, "320-389_nervous"),
        (459, "390-459_circulatory"),
        (519, "460-519_respiratory"),
        (579, "520-579_digestive"),
        (629, "580-629_genitourinary"),
        (679, "630-679_pregnancy"),
        (709, "680-709_skin"),
        (739, "710-739_musculoskeletal"),
        (759, "740-759_congenital"),
        (779, "760-779_perinatal"),
        (799, "780-799_symptoms"),
        (999, "800-999_injury"),
    ]
    for upper, name in chapters:
        if num <= upper:
            return name
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-III for embedding training")
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    diagnoses = load_diagnoses(args.mimic_dir)
    admissions = load_admissions(args.mimic_dir)
    patients = load_patients(args.mimic_dir)

    # Save ICD hierarchy map
    icd_map = build_icd_hierarchy(diagnoses)
    with open(args.output_dir / "icd_hierarchy.json", "w") as f:
        json.dump(icd_map, f, indent=2)
    print(f"ICD hierarchy: {len(icd_map)} admissions with diagnoses")

    # Save patient-admission summary
    adm_summary = admissions.merge(patients[["subject_id", "gender", "dob", "expire_flag"]], on="subject_id")
    adm_summary.to_csv(args.output_dir / "admissions_summary.csv", index=False)
    print(f"Admissions summary: {len(adm_summary)} rows")

    # Save diagnosis labels per admission (for downstream classification)
    diag_labels = diagnoses[["hadm_id", "icd9_code", "seq_num"]].copy()
    diag_labels["icd_chapter"] = diag_labels["icd9_code"].apply(get_icd_chapter)
    diag_labels.to_csv(args.output_dir / "diagnosis_labels.csv", index=False)
    print(f"Diagnosis labels: {len(diag_labels)} rows")

    # Try to build temporal note pairs
    notes = load_noteevents(args.mimic_dir)
    if len(notes) > 0:
        pairs = build_temporal_pairs(notes)
        with open(args.output_dir / "temporal_pairs.json", "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Temporal pairs: {len(pairs)} pairs from {notes['subject_id'].nunique()} patients")

        note_dataset = build_note_level_dataset(notes, diagnoses)
        note_dataset.to_csv(args.output_dir / "notes_with_icd.csv", index=False)
        print(f"Note-level dataset: {len(note_dataset)} notes")
    else:
        print("\nWARNING: NOTEEVENTS is empty (demo dataset).")
        print("The full MIMIC-III dataset is required for note-based training.")
        print("Structured data (diagnoses, admissions) has been preprocessed.")
        print("To proceed with the full pipeline, obtain NOTEEVENTS from PhysioNet.")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
