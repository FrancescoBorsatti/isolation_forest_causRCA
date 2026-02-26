#!/usr/bin/env python3
"""
Isolation Forest anomaly detection on the causRCA dataset.

Straightforward tabular approach:
  - Each sample = state snapshot of all system variables at the diagnosis time
    (fault runs) or end of recording (normal runs).
  - Labels:     0 = normal,  1 = fault
  - Root cause: one-hot vector per fault run — feature is 1 if it appears in
                the dataset's `manipulatedVars` ground-truth list.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "causRCA" / "data"
DIG_TWIN    = DATA_DIR / "dig_twin"
REAL_OP     = DATA_DIR / "real_op"
ENCODING_F  = DATA_DIR / "categorical_encoding.json"

with open(ENCODING_F) as f:
    CAT_ENCODING: dict = json.load(f)


# ── Value conversion ───────────────────────────────────────────────────────────
def to_numeric(node: str, value: str, vtype: str) -> float:
    """Convert a raw string event-log value to a float."""
    if vtype in ("Binary", "Alarm"):
        return 1.0 if str(value).strip().lower() in ("true", "1") else 0.0
    if vtype == "Categorical":
        enc = CAT_ENCODING.get(node, {})
        return float(enc.get(str(value), np.nan))
    try:                        # Continuous / Counter
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ── Snapshot from event-log CSV ────────────────────────────────────────────────
def get_snapshot(csv_path: Path, snapshot_time: float) -> pd.Series:
    """
    Return the last known state of every node at or before snapshot_time
    as a numeric Series indexed by node name.
    """
    df = pd.read_csv(csv_path)
    df = df[df["time_s"] <= snapshot_time].sort_values("time_s")
    last = df.groupby("node").last()          # last event per node
    return pd.Series(
        {node: to_numeric(node, row["value"], row["type"])
         for node, row in last.iterrows()},
        name=str(csv_path),
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
                t = causes.get("diagnosis_at") or causes.get("cause_start_at")

                records.append({
                    "label"          : 1,
                    "subsystem"      : sub_dir.name,          # exp_coolant etc.
                    "exp_id"         : desc["exp_id"],
                    "run"            : run_dir.name,
                    "manipulated_vars": manip_vars,
                    "snapshot"       : get_snapshot(csvs[0], t),
                })
    print(f"  Fault records loaded : {len(records)}")
    return records


# ── Load normal runs ───────────────────────────────────────────────────────────
def load_normal_records() -> list[dict]:
    records = []
    for csv_path in sorted(REAL_OP.glob("*.csv")):
        t = pd.read_csv(csv_path, usecols=["time_s"])["time_s"].max()
        records.append({
            "label"          : 0,
            "subsystem"      : "real_op",
            "exp_id"         : csv_path.stem,
            "run"            : "",
            "manipulated_vars": [],
            "snapshot"       : get_snapshot(csv_path, t),
        })
    print(f"  Normal records loaded: {len(records)}")
    return records


# ── Build X, y, root-cause matrix ─────────────────────────────────────────────
def build_matrices(fault_records, normal_records):
    all_records = fault_records + normal_records

    # feature matrix: rows = samples, cols = nodes, NaN → 0
    X = pd.DataFrame([r["snapshot"] for r in all_records]).fillna(0.0)
    y = np.array([r["label"] for r in all_records])
    cols = list(X.columns)

    # root-cause one-hot: only for fault samples (same row order as fault_records)
    rc = pd.DataFrame(
        [{col: int(col in r["manipulated_vars"]) for col in cols}
         for r in fault_records],
        columns=cols,
    )
    return X, y, rc, all_records


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    fault_records  = load_fault_records()
    normal_records = load_normal_records()

    print("Building feature matrix …")
    X, y, rc_full, all_records = build_matrices(fault_records, normal_records)
    n_feat = X.shape[1]
    print(f"  Shape: {X.shape}  |  fault={y.sum()}  normal={(y==0).sum()}")

    # Root-cause columns that are actually used by at least one experiment
    rc_active_cols = [c for c in rc_full.columns if rc_full[c].any()]
    rc = rc_full[rc_active_cols]
    print(f"  Active root-cause features: {len(rc_active_cols)}")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    print("Fitting Isolation Forest …")
    contamination = float(y.mean())
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)

    # score_samples returns negative average depth; negate so higher = more anomalous
    scores  = -clf.score_samples(X)
    y_pred  = (clf.predict(X) == -1).astype(int)

    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=["normal", "fault"]))
    print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    #   Top row  : score distribution | confusion matrix | ROC curve
    #   Bottom   : root-cause one-hot heatmap (full width)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Isolation Forest on causRCA Dataset\n"
        "Tabular snapshot at diagnosis time (fault) / end of recording (normal)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.8],
                           hspace=0.42, wspace=0.35)

    # ── 1. Anomaly score distribution ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores[y == 0], bins=40, alpha=0.65, color="steelblue", label="Normal")
    ax1.hist(scores[y == 1], bins=40, alpha=0.65, color="tomato",    label="Fault")
    thresh = np.percentile(scores, (1 - contamination) * 100)
    ax1.axvline(thresh, color="k", linestyle="--", linewidth=1.2,
                label=f"IF threshold\n({thresh:.3f})")
    ax1.set_xlabel("Anomaly score  (higher → more anomalous)")
    ax1.set_ylabel("Count")
    ax1.set_title("Anomaly Score Distribution")
    ax1.legend(fontsize=8)

    # ── 2. Confusion matrix ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
        xticklabels=["Pred: Normal", "Pred: Fault"],
        yticklabels=["True: Normal", "True: Fault"],
        annot_kws={"size": 14},
    )
    ax2.set_title("Confusion Matrix")

    # ── 3. ROC curve ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    RocCurveDisplay.from_predictions(y, scores, ax=ax3,
                                     name=f"IF  (AUC={roc_auc_score(y, scores):.3f})")
    ax3.set_title("ROC Curve")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    # ── 4. Root-cause one-hot heatmap ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])

    # y-axis labels: "coolant exp_22 run_1"
    row_labels = [
        f"{r['subsystem'].replace('exp_', '')[:5]}  {r['exp_id']}  {r['run']}"
        for r in fault_records
    ]

    sns.heatmap(
        rc.values,
        ax=ax4,
        cmap=sns.color_palette(["#f0f0f0", "#e74c3c"], as_cmap=True),
        cbar=False,
        linewidths=0.3,
        linecolor="white",
        xticklabels=rc_active_cols,
        yticklabels=row_labels,
    )
    ax4.set_title(
        "Ground Truth Root Causes per Fault Run  "
        "(red = root-cause feature,  grey = not a root cause)",
        fontsize=11,
    )
    ax4.set_xlabel("System Feature (manipulated variable)")
    ax4.set_ylabel("Fault Run")
    ax4.tick_params(axis="x", rotation=45, labelsize=8)
    ax4.tick_params(axis="y", labelsize=6.5)

    out = Path(__file__).parent / "isolation_forest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
