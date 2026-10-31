# Instructions (permanent) — rigsim-benchmark

This file holds the standing rules for working in this repository. Read it fully at the start of every session.
Companion repository: [rigsim](https://github.com/keysie/rigsim), which has its own agent-docs.

## Permanent instructions

These apply to every agent in every session. The user may override one for a single task; it stays in force afterwards.

### Working with the user

1. **Propose first, build after clearance.** Never just write code. Describe the change (what, where, why, risks,
   tests) in the answer and wait for the user's go-ahead. A proposal is short, not an essay.
   Exception: the agent-docs themselves (rule 3) are updated without asking, on the same branch as the work.
2. **Say what you are doing** in one line before any task that takes several tool calls.
3. **Keep agent-docs updated with every answer.** Before finishing an answer, check each file in this folder:
   journal (new decision or finding, dated), open-points (added, reprioritised, closed), findings (new facts added,
   stale ones corrected or removed), briefing (still true? still under 16k tokens?). If nothing changed, change nothing.
   Never leave a statement standing that you know is superseded.
   The briefing's section 0 ("Now") must always hold the direct next steps and the few most recently solved items
   (about five, newest first, merged or closed only; open PRs belong under next steps). Update it in every branch that
   changes the state, and drop the oldest solved item when adding a new one.
4. **Say what is verified and what is not.** Mark claims as *measured*, *read from source/docs*, *inferred* or
   *user decision*. Never present an inference as a fact.
5. **Keep answers short.** The user prefers short answers. Details belong in these files, not in the chat.

### Git

6. **Conventional commit messages**: `<type>[(<scope>)][!]: <description>`, type one of feat, fix, docs, style,
   refactor, perf, test, build, ci, chore, revert. CI and `.githooks/commit-msg` reject anything else. Once per clone:
   `git config core.hooksPath .githooks`.
7. **Branches and pull requests for all building.** Never commit to `main` directly. Before starting, decide:
   - the work belongs to a feature or fix that already has an open branch → add commits to that branch;
   - it is a different feature or fix → open a new branch named after it (`feat/…`, `fix/…`, `docs/…`);
   - unrelated changes are never mixed in one branch. One topic per commit, one topic per PR.
   Open a PR when the branch is ready for review. The user merges; agents do not merge their own PRs.
8. **Run the tests before and after** a change (commands in the briefing). Never hand over a red branch without saying so.

### Secrets and data

9. **The GitHub token is a secret.** Use it only in shell commands as an HTTP header; never write it into a file, commit,
   git config, remote URL, journal or reply. Each bash call is a fresh shell: re-export it per command. Remind the user
   to revoke it at the end of the session.
   ```
   AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
   git -c http.extraheader="Authorization: Basic $AUTH" push origin <branch>
   ```
10. **No personal data, hostnames, serial numbers, usernames or other identifying data** in files, commits or docs.
    If the user pastes such data, say so and ask for test data instead.

## Repository-specific rules

11. **Published runs are immutable.** Never edit, rename, delete or "fix" anything under `results/runs/`; there is no
    list of known errors by design. Corrections happen in newer stick versions and are recorded in the data.
12. **Do not edit generated or bot-written files by hand:** `docs/data.json` (from `tools/index.py`), `predictions/<tag>/*`
    (from `tools/predict.py`), `results/attribution/*` (written by the submission pipeline). Bots push to `main` with
    their own commit-message check; do not imitate them.
13. **The run format only grows** (`results/FORMAT.md`): minor versions add, nothing is removed, renamed or re-typed. A
    format change bumps the format version, the changelog row, `stick/VERSION` (in the same commit that changes what the
    stick measures or writes) and the validator (`tools/validate.py`) together.
14. **Value keys are stable** (`tools/measurements.py`): a new kind of value gets a new key and bumps `VERSION` there;
    an existing key never changes meaning. This is the contract with rigsim's predictions.
15. **Everything that can influence a result is recorded** in the run (tool versions, decisions, command lines,
    fallbacks). New measurement code must record its versions and any decision it takes.
16. **The stick must stay unattended and harmless:** nothing written to the measured machine's disks, no power limits
    changed, no serial numbers kept, the CPU clock restored after each step.
17. **Cross-repo changes** (prediction contract, keys, run format that rigsim reads): open the paired PRs, say which
    merges first, and never rely on an unreleased rigsim tag.
