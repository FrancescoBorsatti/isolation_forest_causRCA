#!/usr/bin/env python3
"""Isolation Forest on causRCA: load → tabular snapshots → fit → report.

Each recording is sampled at regular time intervals (INTERVAL_S).
Labels are derived from cause_start_at: 0 = pre-fault / normal, 1 = fault active.
IF is trained unsupervised (no labels); labels are only used for evaluation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

INTERVAL_S = 30.0  # snapshot every N seconds

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "causRCA" / "data"
DIG_TWIN   = DATA_DIR / "dig_twin"
REAL_OP    = DATA_DIR / "real_op"
ENCODING_F = DATA_DIR / "categorical_encoding.json"

with open(ENCODING_F) as f:
    CAT_ENCODING: dict = json.load(f)


# ── Value conversion ───────────────────────────────────────────────────────────
def to_numeric(node: str, value: str, vtype: str) -> float:
    if vtype in ("Binary", "Alarm"):
        return 1.0 if str(value).strip().lower() in ("true", "1") else 0.0
    if vtype == "Categorical":
        enc = CAT_ENCODING.get(node, {})
        return float(enc.get(str(value), np.nan))
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ── Snapshot from a pre-loaded dataframe ──────────────────────────────────────
def snapshot_from_df(df: pd.DataFrame, snapshot_time: float) -> pd.Series:
    """Last known state of every node at or before snapshot_time."""
    last = df[df["time_s"] <= snapshot_time].groupby("node").last()
    return pd.Series(
        {node: to_numeric(node, row["value"], row["type"])
         for node, row in last.iterrows()}
    )


# ── Load fault runs ────────────────────────────────────────────────────────────
def load_fault_records() -> list[dict]:
    records = []
    for sub_dir in sorted(DIG_TWIN.iterdir()):
        if not sub_dir.is_dir():
            continue
        for exp_dir in sorted(sub_dir.glob("exp_*")):
            desc_files = list(exp_dir.glob("*_description.json"))
            if not desc_files:
                continue
            with open(desc_files[0]) as f:
                desc = json.load(f)
            manip_vars = desc.get("manipulatedVars", [])

            for run_dir in sorted(exp_dir.glob("run_*")):
                causes_f = run_dir / "causes.json"
                csvs     = list(run_dir.glob("*.csv"))
                if not causes_f.exists() or not csvs:
                    continue
                with open(causes_f) as f:
                    causes = json.load(f)

                cause_start    = causes["cause_start_at"]
                alarms_resolved = causes["alarms_resolved_at"]
                df = pd.read_csv(csvs[0]).sort_values("time_s")
                max_time = df["time_s"].max()

                for t in np.arange(0, max_time + INTERVAL_S, INTERVAL_S):
                    records.append({
                        "label"           : int(cause_start <= t <= alarms_resolved),
                        "subsystem"       : sub_dir.name,
                        "exp_id"          : desc["exp_id"],
                        "run"             : run_dir.name,
                        "t"               : t,
                        "cause_start_at"  : cause_start,
                        "alarms_resolved" : alarms_resolved,
                        "manipulated_vars": manip_vars,
                        "snapshot"        : snapshot_from_df(df, t),
                    })

    n_fault  = sum(r["label"] == 1 for r in records)
    n_normal = sum(r["label"] == 0 for r in records)
    print(f"  Fault run snapshots  : {len(records)}  "
          f"({n_fault} fault window label=1, {n_normal} pre/post label=0)")
    return records


# ── Load normal runs ───────────────────────────────────────────────────────────
def load_normal_records() -> list[dict]:
    records = []
    for csv_path in sorted(REAL_OP.glob("*.csv")):
        df = pd.read_csv(csv_path).sort_values("time_s")
        max_time = df["time_s"].max()
        for t in np.arange(0, max_time + INTERVAL_S, INTERVAL_S):
            records.append({
                "label"           : 0,
                "subsystem"       : "real_op",
                "exp_id"          : csv_path.stem,
                "run"             : "",
                "t"               : t,
                "cause_start_at"  : None,
                "manipulated_vars": [],
                "snapshot"        : snapshot_from_df(df, t),
            })
    print(f"  Normal run snapshots : {len(records)}")
    return records


# ── Build X, y, root-cause matrix ─────────────────────────────────────────────
def build_matrices(fault_records, normal_records):
    all_records = fault_records + normal_records
    X = pd.DataFrame([r["snapshot"] for r in all_records]).fillna(0.0)
    y = np.array([r["label"] for r in all_records])
    cols = list(X.columns)
    rc = pd.DataFrame(
        [{col: int(col in r["manipulated_vars"]) for col in cols}
         for r in fault_records],
        columns=cols,
    )
    return X, y, rc, all_records


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    fault_records  = load_fault_records()
    normal_records = load_normal_records()

    X, y, rc_full, _ = build_matrices(fault_records, normal_records)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features  "
          f"(label=1: {y.sum()}, label=0: {(y==0).sum()})")

    contamination = float(y.mean())
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)

    scores = -clf.score_samples(X)
    y_pred = (clf.predict(X) == -1).astype(int)

    print(classification_report(y, y_pred, target_names=["normal/pre-fault", "fault"]))
    print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}")


if __name__ == "__main__":
    main()
