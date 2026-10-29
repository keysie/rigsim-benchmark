# rigsim predictions

What [rigsim](https://github.com/keysie/rigsim) predicts for each measured run, next to what was measured. Derived
data, kept apart from the runs (which never change): `predictions/<rigsim tag>/<run>.json`. Licence: CC BY 4.0, as
the runs.

- A daily job (`.github/workflows/predictions.yml`, `tools/predict.py`) takes rigsim's newest engine release
  (tag `vX.Y.Z`) and predicts every run that has no file for that tag yet. A new rigsim release therefore predicts
  all runs once; a new run gets the newest release once.
- When the set of measured values grows (tools/measurements.py `VERSION`), the newest release predicts every run again,
  so new values get predicted too.
- Files of older releases are never recomputed or removed: the data page shows the newest release's prediction and,
  on hover, the earlier ones.
- A failed prediction is kept as a file with `error` (so it is not retried every day; the next release tries again).

## Contract with rigsim

At a release tag, rigsim provides `backend/app/predict.py`, run from `backend/` (after
`pip install -r backend/requirements.txt` when that file exists):

```
python3 -m app.predict --run <run folder> --measurements <measurements.json> --out <out.json>
```

- **`--run`**: the run folder in the run format ([results/FORMAT.md](../results/FORMAT.md)); rigsim builds the
  machine from its inventory and manifest.
- **`--measurements`**: what `tools/measurements.py` extracts: `{"version": "1", "run": …, "measurements": [{"key",
  "value", "unit", "spread", "source", "conditions"}]}`. The keys and their meaning are fixed (listed in
  `tools/measurements.py`); new kinds of values get new keys.
- **`--out`**, written by rigsim:
  ```json
  {"predictions": [{"key": "llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128", "value": 158.1, "unit": "tok/s",
                    "calibration": false, "basis": "free text: what the prediction rests on"}],
   "unsupported": [{"key": "…", "reason": "GPU not in rigsim's catalogue"}]}
  ```
  Only keys that were measured, in the measured unit. `calibration: true` marks a value whose measurement went into
  one of the release's calibration constants — not an independent test; the page says so.
- Exit code 0 on success. Anything else is recorded as the run's error for that release.

## File

```json
{"rigsim": {"tag": "v0.1.0", "commit": "…"}, "run": "…", "measurements_version": "1", "computed_utc": "…",
 "predictions": [...], "unsupported": [...]}        or  "error": "…"  instead of the two lists
```
