import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

#  Paths
DATA_DIR = Path(__file__).parent / "causRCA" / "data"
DIG_TWIN = DATA_DIR / "dig_twin"
REAL_OP = DATA_DIR / "real_op"
ENCODING_F = DATA_DIR / "categorical_encoding.json"

with open(ENCODING_F) as f:
    CAT_ENCODING: dict = json.load(f)


#  Value conversion
def to_numeric(node: str, value: str, vtype: str) -> float:
    """Convert a raw string event-log value to a float."""
    if vtype in ("Binary", "Alarm"):
        return 1.0 if str(value).strip().lower() in ("true", "1") else 0.0
    if vtype == "Categorical":
        enc = CAT_ENCODING.get(node, {})
        return float(enc.get(str(value), np.nan))
    try:  # Continuous / Counter
        return float(value)
    except (ValueError, TypeError):
        return np.nan


#  Snapshot from event-log CSV
def get_snapshot(csv_path: Path, snapshot_time: float) -> pd.Series:
    """
    Return the last known state of every node at or before snapshot_time
    as a numeric Series indexed by node name.
    """
    df = pd.read_csv(csv_path)
    df = df[df["time_s"] <= snapshot_time].sort_values("time_s")
    last = df.groupby("node").last()  # last event per node
    return pd.Series(
        {
            node: to_numeric(node, row["value"], row["type"])
            for node, row in last.iterrows()
        },
        name=str(csv_path),
    )


#  Load fault runs
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
                csvs = list(run_dir.glob("*.csv"))
                if not causes_f.exists() or not csvs:
                    continue
                with open(causes_f) as f:
                    causes = json.load(f)
                t = causes.get("diagnosis_at") or causes.get("cause_start_at")

                records.append(
                    {
                        "label": 1,
                        "subsystem": sub_dir.name,  # exp_coolant etc.
                        "exp_id": desc["exp_id"],
                        "run": run_dir.name,
                        "manipulated_vars": manip_vars,
                        "snapshot": get_snapshot(csvs[0], t),
                    }
                )
    print(f"  Fault records loaded : {len(records)}")
    return records


#  Load normal runs
def load_normal_records() -> list[dict]:
    records = []
    for csv_path in sorted(REAL_OP.glob("*.csv")):
        t = pd.read_csv(csv_path, usecols=["time_s"])["time_s"].max()
        records.append(
            {
                "label": 0,
                "subsystem": "real_op",
                "exp_id": csv_path.stem,
                "run": "",
                "manipulated_vars": [],
                "snapshot": get_snapshot(csv_path, t),
            }
        )
    print(f"  Normal records loaded: {len(records)}")
    return records


#  Build X, y, root-cause matrix
def build_matrices(fault_records, normal_records):
    all_records = fault_records + normal_records

    # feature matrix: rows = samples, cols = nodes, NaN → 0
    X = pd.DataFrame([r["snapshot"] for r in all_records]).fillna(0.0)
    y = np.array([r["label"] for r in all_records])
    cols = list(X.columns)

    # root-cause one-hot: only for fault samples (same row order as fault_records)
    rc = pd.DataFrame(
        [
            {col: int(col in r["manipulated_vars"]) for col in cols}
            for r in fault_records
        ],
        columns=cols,
    )
    return X, y, rc, all_records


#  Main
def main():
    fault_records = load_fault_records()
    normal_records = load_normal_records()

    X, y, rc_full, _ = build_matrices(fault_records, normal_records)
    print(
        f"Dataset: {X.shape[0]} samples, {X.shape[1]} features  "
        f"(fault={y.sum()}, normal={(y == 0).sum()})"
    )

    contamination = float(y.mean())
    clf = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    clf.fit(X)

    scores = -clf.score_samples(X)
    y_pred = (clf.predict(X) == -1).astype(int)

    print(classification_report(y, y_pred, target_names=["normal", "fault"]))
    print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}")


if __name__ == "__main__":
    main()
