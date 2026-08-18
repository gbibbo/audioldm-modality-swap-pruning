# AudioLDM Modality-Swap Pruning - Progress

> Living state document. Keep this compact. The SessionStart hook injects only the bounded state block below. Detailed experimental provenance belongs in `docs/experiment_ledger.md` and milestone artifacts.

## CURRENT STATE

* Repository established at `/teamspace/studios/this_studio/audioldm-modality-swap-pruning`, branch `main`, initialized fresh on 2026-08-18.
* No prior research repository was recoverable: no `audioldm*` / modality-swap / pruning repo exists under any `gh`-accessible account (`gbibbo` + `Rhysbv/panns_ndi`), and none exists anywhere on this Lightning filesystem. The M0/M1 work described in the master plan (repo bootstrap, frozen references, LoRA CPU scaffold/tests) is **local-only on the author's machine and unpushed**; it is NOT present here.
* `docs/master_plan_v3.md` is in place and is the scientific execution contract.
* Minimal agent kit installed and verified. Repo-local `.claude/` contains only kit hooks/skills; no Edge Audio Labs, Clockify, meeting-transcript, Jira, or publication tooling in this repository.
* Frozen reference SHAs verified to exist upstream (read-only API check, not yet imported):
  * `haoheliu/AudioLDM-training-finetuning` @ `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` (2024-12-13).
  * `Arshdeep-Singh-Boparai/PruningAudioLDM` @ `6f65f628fabc4ad27770753698fc81944e820f9f` (2026-07-16).
* No scientific source tree yet: `audioldm_train/`, `audioldm_peft/`, `research_pruning/`, `configs/`, `scripts/`, `tests/` are all absent. There is no CPU test suite to reproduce in this repository; `pytest` is not installed in the active env.
* M0 is NOT complete. M1 is NOT verifiable here. First real Lightning GPU benchmark and numeric compute budget are pending.
* M3 remains blocked until the benchmark is recorded, Compute Gate CG is resolved, and `docs/pilot_protocol.md` is frozen and committed.

## OPEN ITEMS

1. Import upstream history: add `haoheliu/AudioLDM-training-finetuning` as a remote, fetch `702a638d`, and preserve it as the `upstream-frozen` branch so `git diff upstream-frozen -- audioldm_train/` becomes meaningful.
2. Recover the local-only M1 LoRA/PEFT CPU scaffold and its passing tests from the author's machine into this repository, then re-run those tests here before modifying them.
3. Record the PruningAudioLDM reference clone under `_external/` and capture environment metadata for `artifacts/m0_baseline_reproduction/`.
4. Complete the M0 public-artifact inventory, including the full-FT `(1,2,3,1)` checkpoint search; if unavailable, request it from Arshdeep.
5. Implement M2 audio/text conditioning instrumentation and prepare the single reproducible GPU benchmark command that records all Section 7.2 variables.
6. Resolve Compute Gate CG before M3.

## RUN RECIPES

* Agent kit verification: `python3 .claude/verify_agent_kit.py .`
* Progress structure: `python3 .claude/hooks/check_progress.py PROGRESS.md`
* Git state: `git status --short --branch`
* Claude config: `claude doctor`
* Add project build/test/benchmark commands only after they are verified in this repository.

<!-- FIN-ESTADO -->

## LOG

### 2026-08-18 | Bootstrap of the dedicated research repository

* Searched for the prior research repository before creating anything: `gh repo list` / `gh api /user/repos` (owner+collaborator+org) and a filesystem-wide search returned no AudioLDM, modality-swap, or pruning repository. Repository therefore initialized from scratch, not recovered.
* Installed `audioldm-agent-kit-minimal-v2` against this repository root.
* Verifications passed: `check_progress.py` OK, `verify_agent_kit.py` OK, `settings.json` valid JSON, `bash -n` on all three hooks OK, all three hooks execute cleanly, kit `tests/test_install.sh` self-test OK.
* Contamination audit: repository clean. External scopes clean except for account-level claude.ai MCP connectors (Atlassian/Jira, Gmail, Google Calendar, Google Drive, Slack) reachable from any session in this environment; these are outside the repository and cannot be removed by the kit.
* No scientific milestone is marked complete by this bootstrap.
