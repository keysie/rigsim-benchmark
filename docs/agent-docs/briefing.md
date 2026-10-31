# Briefing — rigsim-benchmark (entry point for every new agent)

Hard limit: **16k tokens**. If this file grows beyond ~12k, move detail to `findings.md` or other docs.
Last revised: 2026-10-31. Facts here are *read from the repository* unless marked otherwise.

## 1. What this project is

**rigsim-benchmark** measures real machines with a **bootable USB stick** and publishes the results as open data.
Plug the stick in, boot, walk away (1.5–3 h): it measures CPU, memory and every GPU (low-level probes, llama.cpp,
Blender), stores results on the stick and powers off. Nothing is written to the machine's disks, no serials are kept.
Data page: https://keysie.github.io/rigsim-benchmark/ . Like its sibling it is largely AI-written in chat sessions on the
direction of one person (keysie); the data records everything that can influence a result so users can judge it.

Current state (2026-10-31): stick **0.13.2**, run format **1.4**, **15 run folders from 12 distinct machines** (one machine
has 4 runs), 13 attribution files, predictions for tags v0.1.0 (2 files), v0.2.0 (7), v0.3.0 (15).

## 2. The two repositories and how they interact

| | **rigsim** | **rigsim-benchmark** (this repo) |
|---|---|---|
| Is | the simulator (browser, static site): model, catalogs, UI | the measurement stick + open data |
| Licence | Apache 2.0 | code Apache 2.0, data in `results/` and `docs/data.json` CC BY 4.0 |
| Changes | fast, evolving model | runs immutable; format only grows |

**Data flow.**
1. **Stick → run.** Each boot writes `runs/<UTC time>-<machine hash>-<random>/` on the stick (`run.json`, `manifest.json`,
   `inventory/`, `q/<question>/`, tool outputs, `stick.log`). `<machine>` is a salted hash of board identifiers.
2. **Run → repo.** Contributors submit via a "Submit runs" issue (zip/tar.gz → `tools/submission.py` → PR from branch
   `submission/issue-<n>`) or `tools/contribute.sh`. CI (`tools/validate.py`) allows only new run folders, exactly as the
   stick wrote them, from finished (or stopped, with the interrupted step pruned) runs, nothing identifying. After merge the
   table is rebuilt (`tools/index.py` → `docs/data.json`) and Pages rebuilt.
3. **Run → values.** `tools/measurements.py` extracts stable named values: `llama-bench/<backend>/<dev>/<model>/<test>`,
   `gpu-vram/<gpu>/<op>`, `cpu-mem-ladder/<kernel>/<threads>`, `mlc/idle_latency`, `mlc/max_bandwidth/<traffic>`.
   Its `VERSION` (now "2") bumps when the set of values changes; the next daily job then predicts every run again.
4. **Values → predictions.** Daily 03:17 UTC (`predictions.yml`): `tools/predict.py` finds rigsim's newest tag `vX.Y.Z`,
   and for each run without a file for that tag runs rigsim's `python3 -m app.predict --run … --measurements … --out …`
   from rigsim's `backend/`. Result: `predictions/<tag>/<run>.json` (never recomputed or removed; failures kept with
   `error`). The data page shows measured value, newest prediction, deviation; earlier tags on hover. A prediction with
   `calibration: true` went into a constant of that release (not an independent test).
5. **The rigsim tag is the only interface** (plus key names and the output format in `predictions/README.md`).
   rigsim consumes no code from here; this repo consumes no rigsim code beyond that CLI at a tag.
6. **Runner/toolkits:** `bench/hw/` (the bench runner and toolkits the stick builds on) lives **only here** since 2026-10-31;
   rigsim's earlier copy was removed (rigsim PR #25).

**Why separate** (stated in the repos: different licences and immutable data; the data "stands on its own"; runner lives
only here. *Inferred, confirm with user:* keeps validation data out of the model's repo so it cannot silently become
fitting data; independent release cadence — stick versions vs rigsim tags; contributors submit machines without
touching simulator code; heavy build machinery and bot-written files stay out of the simulator's history).

Do not confuse: `rigsim-benchmark/results/runs/` = **hardware** runs from the stick; `rigsim/measurements/` =
user-submitted **engine** measurements (tokens/s of one engine via the app).

## 3. These documents

| File | Content | Read when |
|---|---|---|
| `briefing.md` | this file | always, first |
| `instructions.md` | permanent working rules (commits, branches, propose-first, secrets, immutability) | always, second |
| `open-points.md` | what is open, roughly by priority | before proposing work |
| `findings.md` | curated facts about the stick, the data and decisions; stale items removed | before touching stick, format or tools |
| `journal.md` | dated log of decisions and findings; rarely read | to learn how a state came to be |

How to work: read briefing + instructions; read open-points; read findings for the area at hand; propose; after
clearance branch, build, test, update these files in the same branch, open a PR. At the end of every answer re-check
the files (instruction 3). Journal entries are appended and never rewritten; findings are rewritten to stay current.

## 4. Repository layout

```
stick/            the measurement stick
  build.sh        builds stick/out/rigsim-stick.iso + seed (Docker; first build hours, later minutes)
  flash.sh        writes the stick: system, data partition RIGSIM (ext4), seed; update/erase keep runs (backup first)
  VERSION         stick release (semver); bump in the commit that changes what is measured/written
  payload/stick.py  on-stick orchestrator;  tools/ (manifest.py, verify.py, collect.sh, deps.sh, ticker.sh)
  questions/*.toml  one file per measurement question: cpu-gemm, cpu-gemm-ladder, cpu-mem-latency, cpu-mem-ladder,
                    cpu-power-time, gpu-gemm, gpu-launch, gpu-p2p, gpu-vram, pcie-copy
  builder/, ubuntu/  build container, GRUB and chroot setup
bench/hw/         bench runner and toolkits (only home)
results/          runs/<run>/ (CC BY 4.0), attribution/<run>.json, README.md, FORMAT.md (versioned format + changelog)
predictions/      <rigsim tag>/<run>.json, README.md (contract)
docs/             GitHub Pages site (legacy build from main:/docs): index.html, data.json (generated)
tools/            contribute.sh, submission.py, validate.py, index.py, measurements.py, predict.py, import.sh
tests/ stick/tests/   pytest
.github/workflows/    ci.yml (Tests), commits.yml, predictions.yml (daily), runs.yml (validate + rebuild table),
                      submission.yml (issue → PR)
```

## 5. What the stick measures (stick/README.md)

Inventory (CPU, board, memory incl. SPD, GPUs, PCIe links with the path through root ports/switches in
`inventory/pcie.json`, screens; serials removed). CPU and memory: latency and bandwidth by core count, Intel MLC, matrix
speed (fp32, fp64, int8 VNNI/AMX), package power limits (RAPL, format 1.4). NVIDIA: cuBLAS GEMM, VRAM bandwidth, PCIe and
card-to-card copies, launch cost, clocks and power. Any GPU: Vulkan probe (bandwidth, dispatch, vector/matrix speed),
OpenCL via clpeak. AMD ROCm and Intel oneAPI: vendor GEMM, bandwidth, launch cost. llama.cpp: llama-bench (TinyLlama 1.1B,
Llama 2 7B) and per-kernel timings (`test-backend-ops`) on CUDA, Vulkan, ROCm, SYCL and CPU, **all from one llama.cpp
commit built from source**. Blender Benchmark on CPU and every GPU device type. Screens are switched off while measuring.
Requirements: 128 GB stick (64 GB with `SKIP_ROCM`/`SKIP_ONEAPI`), x86_64 targets, UEFI or legacy, Secure Boot may stay on.
Operator can skip a step (Enter twice within 5 s; recorded as `operator_skips`). Details, build variables and
troubleshooting: `stick/README.md`.

## 6. Run format (results/FORMAT.md)

`run.json → format = {"name": "rigsim-stick-run", "version": "1.4"}`. Minor versions only add fields/files/folders; readers
ignore what they do not know; a non-additive change would be 2.0 and is avoided. Changelog: 1.0 (stick 0.9.0) first
versioned; 1.1 (0.10.0) card behind every GPU result; 1.2 (0.11.0) `pcie.json`; 1.3 (0.12.0) `operator_skips`;
1.4 (0.13.0) `powercap.json`. Runs without `format` are rejected. Status of a run: `done`, `done with failures` (complete),
`stopped`, `error: …` (incomplete).

## 7. Commands

```
python -m pip install pytest && python -m pytest -q stick/tests tests     # what CI runs (Python 3.12)
python3 tools/validate.py results/runs/<run>                              # check a run
python3 tools/measurements.py results/runs/<run>                          # values of a run
python3 tools/index.py                                                    # rebuild docs/data.json (bots do this)
stick/build.sh [-v]    sudo stick/flash.sh /dev/sdX                       # build / flash (Linux, Docker)
```
Windows builds only run inside WSL 2 with the repo cloned in WSL's own filesystem (CRLF breaks the scripts; `.gitattributes`
forces LF).

## 8. Conventions and gotchas

- Conventional Commits (CI + `.githooks/commit-msg`). Bots commit as `chore(runs)`, `chore(table)`, `chore(predictions)`.
- The repo has several automatic writers to `main`; fetch `main` before branching and before opening a PR, and never touch bot output
  (instruction 12).
- Pages serves everything under `docs/` — including this folder (see open-points).
- Third-party software (Ubuntu, NVIDIA driver/CUDA images, ROCm, oneAPI, llama.cpp, Blender, Intel MLC, GGUF models)
  is downloaded at build time under its own licence and is not part of the repo; building the stick accepts Intel's MLC licence.

## 9. Session start checklist

1. Ask the user for a fine-grained GitHub token (only `keysie/rigsim` and `keysie/rigsim-benchmark`, Contents + Pull
   requests read/write, short expiry). Secret (instruction 9).
2. Clone, `git config core.hooksPath .githooks`, `git log --oneline -15`, open PRs and branches.
3. Read this folder; run the tests; then propose.
