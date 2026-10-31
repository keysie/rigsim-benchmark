# Findings — rigsim-benchmark (curated)

Current facts only. Stale or superseded items are corrected or removed here and recorded in the journal.
Tag: *docs* (read from a repo document or source) / *measured* / *inferred* / *decision*. Last curated: 2026-10-31.

## Data and format

- State 2026-10-31 (docs, counted): stick 0.13.2, format 1.4, 15 run folders, 12 machine hashes, predictions v0.1.0/v0.2.0/v0.3.0.
- Runs are never changed after publication; there is no list of known errors; judging a result against a tool bug is up
  to the user of the data, from what the run records (decision, README).
- One submission/PR may only add new run folders exactly as the stick wrote them; a `stopped` run keeps only finished
  steps; nothing identifying; credit sits in `results/attribution/<run>.json` and may be absent (page: "Submitted by someone").
- `<machine>` = salted hash of board identifiers: groups runs of the same machine without identifying it.
- Runs without `format` (stick builds before 0.9.0) are not accepted. The validator knows every format up to the newest
  released with it; a newer run is rejected until the repo is updated.
- Shared runs are CC BY 4.0; the submitter confirms the machine owners agreed.

## Stick decisions (docs)

- 2026-10-29 decided: llama.cpp is built from source, not taken from release archives, because the archives lack
  `test-backend-ops` on the GPU backends; archives would save only ~5–10 % of the llama.cpp build, which runs only when
  the version changes. All backends come from one commit; rule, tag and commit are recorded.
- `LLAMA_REF` default `release` = newest patch of the previous release line; `latest-release` = newest release.
- Build is reproducible by recorded versions (`tools/manifest.py`): packages, image digests, llama.cpp commits and
  binary hashes, models' SHA-256, MLC and Blender versions; each run adds what actually ran.
- Screens are switched off while measuring (scanout costs integrated graphics a few percent of RAM bandwidth).
- Blender runs by the official offline route (build default script 3.1.1; the 0.13.1 runs used script 3.1.2 — check which is real), one device type at a time.
- Format 1.1–1.4 added, in order: GPU identity per result (`pci`, `uuid`), `pcie.json`, `operator_skips`, `powercap.json`.

## Interface with rigsim (docs: predictions/README.md, tools/predict.py)

- Predictions are keyed by value key, in the measured unit; only measured and supported keys are predicted; the rest is
  `unsupported` with a reason. Failed predictions are stored with `error` and retried only for the next tag.
- When `tools/measurements.py` `VERSION` changes, the newest tag predicts every run again.
- Timeout per prediction run in `predict.py`: 600 s.
- rigsim's `predict.py` (2026-10-31) supports GPUs by nvidia-smi name and memory from its catalogue, power limits, PCIe
  link from `pcie.json`, CPU memory values (MLC idle latency, all-core read bandwidth) for Intel Core 6th–13th gen and AMD
  Ryzen 1000–7000; no prediction is marked `calibration` (no run went into a constant).
- Pitfall: Windows checkouts convert line endings to CRLF and break the build scripts; `.gitattributes` forces LF but a
  wrong checkout keeps its endings (repair: `git rm -rq --cached . && git reset -q --hard` from WSL).

## New CPU questions: first results (stick 0.13.1, two runs, 2026-10-31; measured)

- `cpu-gemm-ladder`: the fixed grid (1…64 threads) fails every step above the machine's logical CPU count (rc 1), so
  every run ends `done with failures` (7600U: 8 steps, 13700K: 3). Harmless but noisy.
- `cpu-power-time`: works; `package_w` is −1 in both runs (energy counter unreadable inside the container), so only
  throughput, mean clock and temperature are recorded over time. The idle-start reading is taken at the start of the
  idle period, when a laptop can still be warm (73 °C).
- Hybrid/SMT parts: OpenBLAS static partition makes the ladder beyond the physical cores measure the slowest thread
  (see rigsim findings, same date).
- `tools/measurements.py` does not yet expose gemm, ladder or power-time values, so rigsim predicts none of them.

## Audit of all 15 runs (measured; 2026-10-31, by reading `run.json` of every run; unit tests: 133 pass)

Works in the runs: inventory, cpu-mem-latency, cpu-mem-ladder, cpu-gemm, NVIDIA questions, vkprobe, llama.cpp (CPU, CUDA,
Vulkan), tool versions and `stick_check`. Does **not** deliver data:
- **Blender: 0 of 15 runs have a successful result.** 8 runs: "not in the seed" (stick builds without it); 5 runs (0.11.0 ..
  0.13.1, both newest): rc 127 — the Blender binary cannot start, `libXfixes.so.3` missing in the run environment; the
  first two runs (0.9.0, 0.11.0): rc 1, cause not checked.
- **oneAPI:** on 9 of 15 runs "the runtime sees no GPU" (older Intel iGPUs; likely unsupported by Level Zero — *inferred*), recorded
  as rc 1 instead of skipped. On the 13700K (UHD 770) it works, but 5 of 92 steps abort (rc 134, the SYCL llama runs).
- **Vulkan on older Intel iGPUs:** test-backend-ops `mul_mat_id` and `flash_attn_ext` abort (rc −6) in all 8 runs on
  such machines — driver or ggml limit, not investigated.
- **Seed content differs between builds:** MLC missing in 2 runs ("not in the seed"), Blender in 8. A stick built without
  a component silently produces incomplete runs; the run records it, the data page does not warn.
- **cpu-gemm-ladder grid** fails above the CPU's logical count; **package energy** reads −1 (see above).
- Consequence: nearly every run ends `done with failures`, so the status carries no information about real problems.
- The unit tests cannot catch any of this: they check the stick's logic, not the contents of the seed or the run environment.
