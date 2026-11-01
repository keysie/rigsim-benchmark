# Journal — rigsim-benchmark

Dated log of decisions, findings and events, newest at the bottom. Append only; never rewrite old entries (correct
by adding a new entry). Rarely read: it explains how the current state came to be. Current facts live in
`findings.md`. History before 2026-10-31 is reconstructed from the repository (about 92 commits, 2026-10-28 to
2026-10-31) and its documents, not from the original sessions.

- **2026-10-28** — First versioned run format 1.0 with stick 0.9.0 (`format`, `stick` object, per-step `tools`,
  `manifest.json`). First runs published. (FORMAT.md, git log)
- **2026-10-28 .. 2026-10-29** — Formats 1.1 (stick 0.10.0), 1.2 (0.11.0), 1.3 (0.12.0) added GPU identity per result,
  `pcie.json`, `operator_skips`. (FORMAT.md)
- **2026-10-29** — Decided: llama.cpp built from source (needs `test-backend-ops`). (stick/README)
- **2026-10-30 .. 2026-10-31** — Format 1.4 (stick 0.13.0, `powercap.json`); stick 0.13.1 added `cpu-gemm-ladder` and
  `cpu-power-time` to the profile (they had been missing); 0.13.2 fixed the progress counter's total from the start.
  Submissions of runs by the user (#11, #15) merged. (git log)
- **2026-10-31** — `bench/hw` lives only here; rigsim's copy removed. (git log)
- **2026-10-31** — agent-docs created in both repositories on branch `docs/agent-docs`. Open points tagged [agent] are
  the agent's reading and need the user's confirmation.
- **2026-10-31** — User confirmed (decision): the reasons for the two-repo split in the briefing are correct; changes to
  agent-docs never need clearing with the user; the pre-journal history here is inferred from the repos (an older chat
  exists but kept no journal) — from now on the journal is kept. Because Pages serves `docs/`, the folder moved from
  `docs/agent-docs` to `agent-docs/` at the repository root.
- **2026-10-31** — The two newest runs (stick 0.13.1, 7600U laptop and 13700K desktop) are the first with the ladder and
  power-time questions. Findings and fixes proposed in findings and open-points (grid fails above the CPU count, package
  power unreadable, no value keys yet). No code changed. (measured, analysis by agent)
- **2026-10-31** — User decision: stick first, to make sure it works and collects the right data. Audit of all 15 runs
  found: Blender never delivered a result (missing library `libXfixes.so.3` in the newest runs, not in the seed in others),
  oneAPI and Vulkan failures on older Intel iGPUs recorded as failures, seed content differing between builds. Plan
  proposed in open-points; nothing changed in code. (measured)
- **2026-10-31** — User cleared A and B of the stick plan. Found that branch `blender-libs` is the user's open PR #19
  (Blender libraries, stick 0.13.3), not a stale branch — an earlier open point said otherwise and was removed. Wrote PR #21
  (ladder cap, 0.13.4; `[limits]` plan section, `cap_to_cpus` in the runner; 134 tests pass, not run on hardware).
  New standing rule: the briefing's section 0 always holds the next steps and the last solved items.
- **2026-10-31** — User merged PR #19 (Blender libraries). User decisions: all stick fixes go into one branch (`fix/stick`;
  PR #21's branch renamed from `fix/ladder-grid`, conflict on `stick/VERSION` and the tests resolved, 135 tests pass);
  agent-docs live on `main` only, no docs branches (the docs PRs #20 and rigsim #27 were merged onto main, their branches
  deleted). Rule 7 rewritten: branches per component, agent-docs committed directly to `main`.
- **2026-10-31** — Direct pushes of agent-docs to `main` are refused by the ruleset (required status "Check Conventional
  Commits"). Workaround recorded in rule 7 (throwaway ref `agent-docs-sync`, wait for the check, push `main`). Asked the
  user whether to add a bypass for the repository admin instead.
- **2026-10-31** — Renaming `fix/ladder-grid` to `fix/stick` closed PR #21 (GitHub does not carry a PR across a branch
  rename here); reopened as rigsim-benchmark PR #22. Lesson: never rename a branch with an open PR.
- **2026-10-31** — User decision: branches are sprints, and agent-docs live on the branch being worked on (replaces the
  "notes only on main" rule and its `agent-docs-sync` workaround). Rule 7 rewritten. PR #22 (stick fixes, stick 0.13.4)
  merged by the user. New sprint `sprint/stick-data` opened in both repositories.
- **2026-10-31** — User decision: a missing part stops the build (not a warning). Built the seed self-test on
  `sprint/stick-data` (stick 0.13.5): `stick/tools/selftest.py`, wired into `build.sh` before the seed manifest, option
  `ALLOW_INCOMPLETE`; 140 tests pass; not yet run in a real build.
- **2026-10-31** — First real build with the seed self-test: everything ok except llama.cpp CUDA, `libcuda.so.1` missing in
  the CUDA image — a false alarm (the NVIDIA container toolkit injects driver libraries from the host). Fixed: driver
  libraries are checked in the system image instead. 141 tests pass. (measured by the user's build)
- **2026-10-31** — User report: the stick does not boot on some machines unless CSM / legacy mode is on. Causes read from
  the build (MBR-only hybrid, no active flag) recorded as inferred in open-points; asked for the symptom first.
- **2026-10-31** — UEFI boot: symptom = stick seen, no boot option. Switched to the GPT hybrid layout of current Ubuntu
  ISOs (`stick/tools/layout.sh`), FAT16 ESP; `flash.sh` appends the data partition as GPT entry after relocating the
  backup header and refuses in-place updates of old-layout sticks. While building a QEMU boot test found that the BIOS
  GRUB core lacked `configfile` (legacy boot ended at `grub>`), fixed. Test green: UEFI, UEFI + Secure Boot, BIOS;
  added as CI workflow "Boot test". 142 unit tests pass.
- **2026-10-31** — User report: for an old-layout stick `flash.sh` refused the update but kept offering it. Fixed: the
  blocker is checked before the menu; [u]pdate is offered only when it can work.
- **2026-10-31** — Branches `sprint/old-gpus` and `ddr-timing-improvements` opened off this one. User's order: this
  sprint first (its stick is running on hardware), then old GPUs, then DDR (parked).
- **2026-10-31** — First hardware run of this sprint's stick (0.13.5): self-test build, Blender and ladder cap verified;
  remaining noise: oneAPI SYCL aborts, Blender oneAPI with no render, gpu-p2p on one GPU. UEFI on an affected machine open.
- **2026-10-31** — User: UEFI boot verified on an affected machine. D built (stick 0.13.6): skipped-with-reason for
  gpu-p2p on one card, vendor runtime without GPU, Blender device types not found; traceback behind rc 0 = failure.
  145 tests pass.
- **2026-10-31** — E: SYCL logs read; the aborts come from the oneDNN path of llama.cpp's SYCL mul_mat on the UHD 770
  (inferred from the message). Proposed: retry once without oneDNN, recorded. Waiting for the user.
- **2026-10-31** — User chose variant 1 (retry without oneDNN, recorded) and asked for the plan before building.
- **2026-11-01** — Manual test confirms the SYCL hypothesis for llama-bench on the UHD 770 (without oneDNN it runs:
  pp512 220 tok/s, tg128 38 tok/s). Ops test still running.
- **2026-10-31** — Manual SYCL test, first result: one test-backend-ops op without oneDNN ends "2/2 backends passed, OK" on the
  UHD 770 (which op not stated); llama-bench with / without oneDNN still to see.
- **2026-11-01** — Manual test complete: FLASH_ATTN_EXT and MUL_MAT_ID also pass without oneDNN. Hypothesis confirmed for
  all four aborted SYCL steps.
