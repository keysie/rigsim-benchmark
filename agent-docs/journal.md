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
