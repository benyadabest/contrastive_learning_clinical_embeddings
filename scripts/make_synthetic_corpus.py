"""Generate a small synthetic corpus for smoke-testing the FastAPI frontend.

The real corpus (~23,657 anchor notes from the 500-patient MIMIC subset) is
gitignored and only exists after running the full embedding pipeline. This
script creates a tiny stand-in that the API can run against end-to-end:

  embeddings/anchor_embeddings_synthetic.npy   (N x 384)
  data/temporal_pairs_synthetic.json           (N records)
  data/icd_hierarchy_synthetic.json            (hadm_id -> list[icd9_code])

Uses sentence-transformers/all-MiniLM-L6-v2 (~80 MB, ~1s cold load) so the
similarity scores returned by the API are meaningful even on the synthetic
data: the same model encodes both the corpus and the query.

Run:
    python scripts/make_synthetic_corpus.py

Then boot the API against it:
    EMBEDDINGS_MODEL_SAFE_NAME=synthetic \\
    PAIRS_FILENAME=temporal_pairs_synthetic.json \\
    ICD_MAP_FILENAME=icd_hierarchy_synthetic.json \\
    HF_REPO=sentence-transformers/all-MiniLM-L6-v2 \\
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SAFE_NAME = "synthetic"


# Scenario -> per-note clinical-ish text. Each list is the chronological
# story for one admission so that PCA / trajectory has structure.
PATIENT_SCENARIOS: list[tuple[str, list[str]]] = [
    ("septic_shock", [
        "Admission note: 62-year-old man presents with fever 39.4, hypotension, lactate 4.5, started on broad-spectrum antibiotics meropenem and vancomycin. Sepsis workup underway.",
        "MICU progress note: Hypotension persists despite 4L crystalloid. Norepinephrine started, vasopressin added. Blood cultures pending. Lactate 3.8.",
        "Renal note: Acute kidney injury, creatinine up from 1.0 to 2.6. Urine output minimal at 15 mL/hr. CVVH being considered.",
        "Pulmonary update: Worsening hypoxemia, intubated for respiratory failure. ARDSnet ventilation initiated. PEEP 12, FiO2 70%.",
        "Day 5 progress: Pressors weaning, vasopressin off. Lactate trending down to 1.9. Cultures grew E. coli, narrowed to ceftriaxone.",
        "Discharge summary: Patient extubated, transferred to step-down. Completing 14-day antibiotic course for E. coli bacteremia.",
    ]),
    ("post_cabg_afib", [
        "Pre-op note: 68-year-old with three-vessel disease, scheduled for CABG x3. EF 45%, no prior MI.",
        "Post-op day 1: Stable post-CABG, extubated, sinus rhythm at 84. Pain controlled with morphine PCA. Chest tubes draining serosanguinous.",
        "Post-op day 2: New-onset atrial fibrillation with RVR to 145. Started on amiodarone load and metoprolol. Hemodynamically stable.",
        "POD 3: AFib persists despite rate control. Considering DCCV. Hemoglobin stable at 9.4. Drains pulled today.",
        "POD 4: Successful cardioversion to sinus rhythm. Anticoagulation with apixaban for AFib. Discharge planning underway.",
    ]),
    ("newborn_respiratory", [
        "Admission note: 33-week GA newborn delivered via C-section for fetal distress. Apgars 5 and 7. Respiratory distress noted at birth.",
        "NICU day 1: Infant placed on CPAP +5, FiO2 35%. Chest X-ray consistent with TTN. Maintaining oxygen saturation 92-96%.",
        "NICU day 2: Weaning FiO2 to 25% on CPAP. Mild grunting noted on exam. Feeding via OG tube every 3 hours.",
        "NICU day 4: Off CPAP, on nasal cannula 1L. Continues to gain weight. Bilirubin trending down with phototherapy.",
        "Discharge note: Off oxygen for 48 hours. Feeding ad lib by bottle. Discharged home with parents trained in CPR.",
    ]),
    ("contrast_nephropathy", [
        "ED note: Patient with CKD stage 3 (baseline creatinine 1.8) underwent CT angiogram with contrast for chest pain workup.",
        "Inpatient day 2: Creatinine up to 2.6, urine output oliguric. Started on IV fluids and bicarbonate. Nephrology consulted.",
        "Inpatient day 3: Creatinine peaked at 3.4. Urine output improving to 30 mL/hr. Continuing hydration. No dialysis indicated.",
        "Discharge: Creatinine trending down to 2.1, near baseline. Nephrology follow-up scheduled. Avoid future contrast.",
    ]),
    ("stroke_eval", [
        "ED presentation: Sudden right-sided weakness and dysarthria, NIH stroke scale 12. CT head negative for hemorrhage. tPA window evaluated.",
        "Stroke unit admission: MRI confirms left MCA territory ischemic stroke. tPA given within window. Heparin gtt started.",
        "Day 2: Mild improvement in motor function, NIHSS now 8. Started on aspirin and atorvastatin. Speech therapy evaluating.",
        "Day 4: Continued improvement. NIHSS 5. PT/OT working on transfers and ADLs. Swallow eval cleared for puree diet.",
        "Discharge to rehab: Patient stable, ambulating with walker. Outpatient therapy and neurology follow-up arranged.",
    ]),
    ("liver_failure", [
        "Hepatology consult: 55-year-old with Hep C cirrhosis, presents with encephalopathy. Ammonia 187. Started on lactulose.",
        "ICU day 2: Worsening encephalopathy, intubated for airway protection. Total bilirubin up to 18. INR 2.8.",
        "ICU day 3: Family meeting today. Patient on transplant list. Discussed prognosis and goals of care.",
        "ICU day 5: Mental status improving on lactulose and rifaximin. Extubated. INR 2.0 with vitamin K.",
        "Step-down transfer: Awaiting liver transplant evaluation. MELD score 28. Outpatient follow-up arranged.",
    ]),
    ("gi_bleed", [
        "ED: Patient with cirrhosis presents with hematemesis and melena. Hemoglobin 6.8. Transfused 2 units PRBC.",
        "GI consult: EGD shows grade III esophageal varices, banded x4. Octreotide drip started. NPO.",
        "ICU day 2: Stable hemodynamically, no further bleeding. Hemoglobin 8.4 post-transfusion. Diet advanced to clears.",
        "Discharge: Tolerating regular diet. Started on propranolol for primary prophylaxis. GI follow-up scheduled.",
    ]),
]

# Real-looking ICD-9 codes per scenario for the synthetic ICD map.
SCENARIO_ICD = {
    "septic_shock":          ["99592", "0389", "5849", "5990", "5180"],
    "post_cabg_afib":        ["41401", "42731", "4280", "4111", "V4581"],
    "newborn_respiratory":   ["7706", "V290", "76519", "76527", "V3001"],
    "contrast_nephropathy":  ["5849", "5859", "4019", "2724", "V5867"],
    "stroke_eval":           ["43411", "78039", "4019", "2724", "V1259"],
    "liver_failure":         ["5715", "5722", "5723", "07054", "5728"],
    "gi_bleed":              ["5780", "4560", "5712", "2851", "5722"],
}

CATEGORIES = ["Nursing/other", "Radiology", "Physician", "Discharge summary", "ECG", "Echo"]


def main(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    pairs: list[dict] = []
    hadm_to_codes: dict[str, list[str]] = {}
    subject_counter = 1000
    hadm_counter = 100_000

    # 1-3 patients per scenario so cohort discovery has neighbors within and
    # across scenarios.
    for scenario, scenario_notes in PATIENT_SCENARIOS:
        n_patients = random.randint(1, 3)
        codes_for_scenario = SCENARIO_ICD[scenario]
        for _ in range(n_patients):
            subject_counter += 1
            hadm_counter += 1
            sid = subject_counter
            hadm = hadm_counter
            hadm_to_codes[str(hadm)] = codes_for_scenario.copy()

            n_notes = random.randint(max(2, len(scenario_notes) - 2), len(scenario_notes))
            base_day = random.randint(1, 20)
            month = (hadm % 12) + 1
            for i in range(n_notes):
                pairs.append({
                    "subject_id": sid,
                    "anchor_hadm_id": hadm,
                    "anchor_category": random.choice(CATEGORIES),
                    "anchor_date": f"2150-{month:02d}-{base_day + i:02d}",
                    "anchor_text": scenario_notes[i],
                })

    # One synthetic "busy patient" with 20 notes so trajectory has a long curve.
    busy_sid = 9999
    busy_hadm = 999999
    hadm_to_codes[str(busy_hadm)] = SCENARIO_ICD["septic_shock"]
    busy_notes = [
        "Daily progress: continued mechanical ventilation, weaning sedation, hemodynamically stable on low-dose pressors.",
        "Progress note: extubated successfully, on high-flow nasal cannula 40L 50%. Saturating 95%.",
        "Renal update: creatinine improving from peak 3.4 to 2.1. Urine output adequate. CVVH discontinued.",
        "Neuro check: alert and oriented x3 after extubation. No focal deficits. Family at bedside.",
        "Cardiology: telemetry shows occasional PVCs, otherwise NSR. Echo shows preserved EF 55%.",
        "ID rounds: blood cultures NTD x48h. Antibiotics narrowed to ceftriaxone for E. coli sensitivity profile.",
        "Pulmonary: weaning HFNC to 4L NC. Chest X-ray shows improving infiltrates.",
        "Nutrition: tolerating oral diet, advancing as tolerated. PO intake 60% of goal.",
        "Wound care: central line removed, no signs of infection at site. Peripheral access maintained.",
        "PT/OT: working on transfers, sitting at edge of bed. Tolerating chair x1 hour.",
        "Pharmacy: medication reconciliation complete. Home meds resumed where appropriate.",
        "Social work: discharge planning, discussing rehab options with family.",
        "Acute change: sudden tachycardia to 140s, hypotension. Rapid response called. Fluids bolused.",
        "Workup for acute change: lactate 2.8, repeat blood cultures drawn. Empiric antibiotics broadened to vanc and zosyn.",
        "Recovery: hemodynamics improved with 2L bolus. Source unclear, blood cultures pending. Continuing broad coverage.",
        "Stable overnight: vitals within normal limits, no recurrence of hypotension. Reducing fluid maintenance.",
        "Cardiology repeat echo: unchanged from prior, EF preserved. No new wall motion abnormalities.",
        "GI consult: mild transaminitis, attributed to sepsis and medication. Trending labs.",
        "Discharge planning: rehab facility identified. Family meeting scheduled for tomorrow.",
        "Final note: stable for discharge to subacute rehab. Outpatient follow-up with PCP and pulmonary scheduled.",
    ]
    for i, txt in enumerate(busy_notes):
        pairs.append({
            "subject_id": busy_sid,
            "anchor_hadm_id": busy_hadm,
            "anchor_category": random.choice(CATEGORIES),
            "anchor_date": f"2151-01-{i + 1:02d}",
            "anchor_text": txt,
        })

    random.shuffle(pairs)

    n_patients = len({p["subject_id"] for p in pairs})
    print(f"Generated {len(pairs)} note records across {n_patients} patients across {len(PATIENT_SCENARIOS)} scenarios")

    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [p["anchor_text"] for p in pairs]
    print(f"Encoding {len(texts)} texts...")
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    embeddings_dir = ROOT / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    emb_path = embeddings_dir / f"anchor_embeddings_{MODEL_SAFE_NAME}.npy"
    pairs_path = data_dir / "temporal_pairs_synthetic.json"
    icd_path = data_dir / "icd_hierarchy_synthetic.json"

    np.save(emb_path, embs)
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    with open(icd_path, "w") as f:
        json.dump(hadm_to_codes, f, indent=2)

    print("\nWrote:")
    print(f"  {emb_path}  (shape={embs.shape})")
    print(f"  {pairs_path}")
    print(f"  {icd_path}")
    print("\nRun the API with:")
    print(f"  EMBEDDINGS_MODEL_SAFE_NAME={MODEL_SAFE_NAME} \\")
    print(f"  PAIRS_FILENAME=temporal_pairs_synthetic.json \\")
    print(f"  ICD_MAP_FILENAME=icd_hierarchy_synthetic.json \\")
    print(f"  HF_REPO={MODEL_NAME} \\")
    print(f"  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
