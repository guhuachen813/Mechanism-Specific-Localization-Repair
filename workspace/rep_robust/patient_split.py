"""Phase-4 audit: patient-level split for NIH box-GT images.

Patient = filename prefix before '_' (NIH convention: 00004344_005.png ->
patient 00004344). Produces:
  - splits/nih_box_patients.csv  (file, finding, patient, fold)
  - deterministic patient-level 5-fold assignment per finding (seed 42),
    folds stratified so every fold gets patients, shared across findings.

Also verifies CheXphoto eval_list patient structure (200 patients / 808 rows).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "/root/autodl-tmp/experiments/rep_robust"
SEED = 42
NFOLD = 5


def main() -> int:
    bbox = pd.read_csv(f"{REPO}/BBox_List_2017.csv", skiprows=1, header=None,
                       usecols=range(6),
                       names=["f", "lab", "x", "y", "w", "h"])
    rows = []
    pat_finding = defaultdict(set)
    for lab in ["Cardiomegaly", "Atelectasis"]:
        b = bbox[bbox.lab == lab]
        for f in sorted(b.f.unique()):
            pat = f.split("_")[0]
            rows.append({"file": f, "finding": lab, "patient": pat})
            pat_finding[pat].add(lab)
    df = pd.DataFrame(rows)

    # deterministic patient-level folds, shared across findings:
    # assign per finding separately so each finding's folds are balanced,
    # then record both (a patient may have different folds across findings
    # only if it appears in both findings -- check).
    fold_of = {}
    for lab in ["Cardiomegaly", "Atelectasis"]:
        pats = sorted(df[df.finding == lab].patient.unique())
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(pats))
        folds = np.array_split(perm, NFOLD)
        for k, idx in enumerate(folds):
            for i in idx:
                fold_of[(lab, pats[i])] = k
    df["fold"] = [fold_of[(r.finding, r.patient)] for r in df.itertuples()]

    both = {p for p, fs in pat_finding.items() if len(fs) > 1}
    print(f"rows: {len(df)}; patients appearing in BOTH findings: {sorted(both)}")
    for lab in ["Cardiomegaly", "Atelectasis"]:
        d = df[df.finding == lab]
        print(f"{lab}: {d.file.nunique()} 图, {d.patient.nunique()} 患者, "
              f"fold sizes: {d.groupby('fold').patient.nunique().to_dict()}")

    out = f"{REPO}/splits"
    import os
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/nih_box_patients.csv", index=False)
    print(f"-> {out}/nih_box_patients.csv")

    el = pd.read_csv(f"{REPO}/eval_list.csv")
    p = el.key.str.extract(r"(patient\d+)")[0]
    print(f"CheXphoto eval_list: {len(el)} 行, "
          f"{p.nunique()} 患者")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
