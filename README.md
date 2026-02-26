# Setup

We use `uv` to manage the virtual environment and dependencies for this project.

To set up the environment, run the following command: `uv sync`. 

To add a new dependency, use `uv add <package-name>`.

To run any script, use `uv run script.py`.

# Objective

Investigate whether the causRCA dataset can be used to do unsupervised anomaly detection with simple Isolation Forest.

https://zenodo.org/records/15876410

## Dataset ELI5

The raw data comes from a CNC lathe machine. Sensors on the machine continuously log events: every time a variable changes its value, a new row is written to a CSV file. So the raw format looks like this:

```
time_s,  node,          value,  type
0.0,     CBC_Closed,    True,   Binary
0.0,     CBC_close,     True,   Binary
106.9,   HP_Pump_Ok,    False,  Binary
155.5,   HP_A_700304,   True,   Alarm
...
```

Each `node` is one sensor or signal on the machine (e.g. "is the high-pressure pump OK?", "is the coolant level alarm active?"). There are 96 such nodes in total.

### One CSV per run, one run = one row in our dataset

The raw data is split into **runs**:
- **Normal runs** (`real_op/`): 170 CSVs, each recording the machine during normal operation.
- **Fault runs** (`dig_twin/exp_*/run_*/`): 100 CSVs, each recording a simulated fault scenario (coolant leak, hydraulic failure, probe issue, etc.). Faults are grouped into 19 experiments; each experiment is repeated 3–8 times to get multiple runs.

Each CSV is one recording session → one row in our tabular dataset.

### How we turn a CSV into a row (feature vector)

A CSV is a time-ordered log of changes. To get a fixed-size feature vector we take a **snapshot**: we freeze the state of every node at one reference timestamp and ask "what was the last known value of each node at that moment?"

- For **fault runs**: the snapshot time is `diagnosis_at` from `causes.json` — the moment a human technician would diagnose the fault.
- For **normal runs**: the snapshot time is the end of the recording.

We then forward-fill: if a node hasn't changed yet by the snapshot time, we use its last recorded value. If it was never recorded at all, we fill with 0.

The result is one row with 96 columns (one per node), all numeric:
- `Binary` / `Alarm` nodes → 1.0 (True) or 0.0 (False)
- `Continuous` / `Counter` nodes → the numeric value as-is
- `Categorical` nodes → a numeric code from `categorical_encoding.json`

### Labels and root causes

| | normal runs | fault runs |
|---|---|---|
| **anomaly label** | 0 | 1 |
| **root cause vector** | all zeros | one-hot: 1 for each node listed in `manipulatedVars` (the variables that were physically manipulated to simulate the fault) |

The root cause vector has the same 96 columns as the feature vector. For example, a coolant leak experiment has `HP_Pump_Ok=1` and `LT_Level_Ok=1` and all other 94 entries are 0.

### Final tabular dataset

```
270 rows × 96 columns
  170 normal (label=0, root cause = all zeros)
  100 fault  (label=1, root cause = one-hot, 1–2 non-zero entries per row)
```

