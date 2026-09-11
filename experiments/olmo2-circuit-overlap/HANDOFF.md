# Fresh-session handoff — OLMo 2 circuit overlap

## Latest instruction and current goal

Scope amendment (2026-09-11): the user explicitly requested **no further probe
activation saving and no further linear-probe fitting**. Keep full accuracy,
answer scores, circuit top nodes and interventions. The first five checkpoints
finished all components; RLVR1 finished behavior and circuits before interruption,
plus three probe tasks. Preserve those artifacts. Resume with
`resume_without_probes.py`, which validates and reuses completed behavior/circuits
and records `probe_policy_by_stage` in metadata. The original sampling plan and
runner remain unchanged. The final audit and report recognize intentionally
skipped probes while still requiring complete behavior/circuits/interventions.
Inspect live status before acting; do not restart the original all-components
controller. RLVR2 is the remaining checkpoint requiring inference.

Active full run (2026-09-11): instance50602969, RTX6000 Ada48GB, approximately
$0.722222/hour,100GBdisk. Do not rent or start a duplicate run. Source commit
`a46ca264a8dcb23ed06460afb55ec2d7e22f7253` was pushed and pulled on the host.
Full evaluation started at2026-09-11T15:19:32Z. Inspect live state in
`geode-store/olmo2-authorized-20260911/` and remote
`/workspace/olmo2-resume/results/full/` before taking any action.

Both pilots passed. All448 saved checkpoint/example evaluations retained their
correctness outcomes at coding192/math384; trained parse-failure counts were
unchanged. Generated tokens fell155,032→40,728; cap pilot elapsed135.38s.
The full-sized RLVR2 pilot passed all eight probe audits and behavior/circuit
smokes in778.54s. Full projection:7.99h/$5.77, or10.39h/$7.50 with a30% allowance,
excluding final reporting/backup overhead. This is an estimate, not a bound.
The full controller reuses that audited pilot; its remaining evaluation guard
is$22.6034, with$2 reserved for subsequent rental overhead and a$25 rental watchdog.

Unattended local automation is running: incremental rsync backup, completion
monitor, and rental watchdog. `/tmp/olmo2-authorized-finish-local.py` waits for
full completion, runs the committed CPU finalizer on the host, verifies every
local backup hash, then STOPS the GPU while preserving disk. It sends urgent
failure notifications to the user-authorized ntfy topic stored in that local
script. Script copies and records are under the local artifact directory.
`completion_supervisor.json`, `backup_status.json`, `completion_stopped.json`
and `cleanup_complete.json` distinguish running, failed and stopped states.

Automatic approval review rejected a combined HF-upload/destruction monitor.
The active safer monitor performs NEITHER HF upload NOR instance destruction.
The user explicitly authorized Vast.ai artifact transfer; do not ask that again.
Separate HF publishing and final disk deletion remain pending. Storage billing
continues after stopping until the instance is destroyed. The prior instruction
below to destroy automatically is superseded by this active monitor policy.

Resume authorization (2026-09-11): the user explicitly authorized transferring
experiment files to Vast.ai and instructed: "Next time just commit push and pull
from the instance." Use the experiment Git branch for source deployment; transfer
ignored saved outputs, public datasets and the frozen plan separately. This
supersedes the earlier stop/export block. No additional export approval is needed.

CRUXEval192/GSM-Symbolic384 overrides are implemented in the runner, full-plan
pilot and controller; all357 CPU experiment tests pass. Native prompts and parsers
are unchanged. The rebuilt plan is
`geode-store/olmo2-full-preflight/full-plan-192-384.json`, fingerprint
`3c6c860ff14a799e7a3d2bfccb7327dd918f88dae7053227236a66d7b78a8ade`.
Its entire payload matches the earlier full plan and its new source binding
validates. All327 original sanity artifacts were rehashed and passed.

First validate caps on saved examples across all seven checkpoints, then audit
the full-sized probe pilot, re-estimate cost, and run the full evaluation behind
the technical gate. Preserve the$25 budget guard. Back up/hash-verify results and
destroy the task instance at completion; stop to preserve disk if backup fails.

The blocked rental50600891 was destroyed without uploaded experiment files or
new inference; estimated upper rental cost$0.061. Its historical records are in
`geode-store/olmo2-resume-20260911/`. The authorized attempt uses fresh records in
`geode-store/olmo2-authorized-20260911/`; consult these before any instance action.

Historical shutdown record:
The user explicitly said: **"stop everything and save everything and terminate gpu box"**.
All experiment work is stopped. Do not launch or rent anything automatically on
resume; first read this handoff and the user's new instruction. GPU instance
50543356 was destroyed and verified absent at2026-09-11T03:12:44Z; the account
has no remaining instances. Final estimated rental cost was$0.68 (not invoice).
See `geode-store/olmo2-sanity-20260910/shutdown/instance_shutdown.json`.
The earlier instruction to keep the instance is superseded.

Scientific goal: compare OLMo 2 1B initialization, Stage-1/2 pretraining, SFT,
DPO, RLVR-1/2 on ethical judgments, math and coding, using the existing node
attribution method, Jaccard@16, native accuracy/likelihoods, layerwise probes,
and held-out activation interventions. Overlap alone cannot prove acquisition
or elicitation. User wants three TLDR lines, then each plot with a takeaway,
caption, actual identified data example and confusing points.

## Saved work

- Branch `olmo2-circuit-overlap`, from `fig2nl2` at
  `08ba917bee6459ec19c87739b58a69a7e2f0ca02`. Source and tests are now authorized for commit/push;
  source archives are also preserved. No merge was requested.
- Code: `geode/circuits/`; experiment scripts: `experiments/olmo2-circuit-overlap/`;
  CPU tests: `tests/lib/circuits/`; interface properties: `specs/00-interfaces.md`.
- Completed original two-checkpoint pilot:
  `geode-store/olmo2-pilot-20260910/`, summary `PILOT.md`.
- Completed seven-checkpoint sanity:
  `geode-store/olmo2-sanity-20260910/`, summary `SANITY.md`, full six-plot
  `report.md` under the artifact directory. It took39.7 minutes at a2048-token
  generation cap. All seven checkpoints loaded, weight hashes matched, no
  missing/unexpected/mismatched keys, and technical audits passed.
- SHA256 backup verification:327 original files,1,507,524,187 bytes.
  See `backup_verification.json`, `remote_artifact_manifest.json`,
  `technical_audit.json`, and `load_integrity.json` in the sanity directory.
-299 experiment tests passed before rental and299 on the remote CPU environment.
  Later reporting/audit/controller/pilot tests also passed their targeted suites.
  The pre-rental full repository suite had one confirmed pre-existing failure:
  `test_overlay_directory_is_fully_discovered[ts1b_fig2ts]` expects38 YAMLs,
  while baseline fig2nl2 contains41. Do not fix this unrelated guard.
- Private HF repository: `mhieuuu/olmo2-circuit-overlap`, authenticated owner
  `mhieuuu`, private verified. Seven-stage artifacts uploaded under
  `sanity/20260910-seven-checkpoints`. A final save may add a later commit;
  consult local shutdown/HF records. No HF credentials were sent to the host.

## Actual generation-length findings

Read `geode-store/olmo2-sanity-20260910/token_budget/concise.md` and
`pilot_lengths.md` plus their detailed JSON and reproducible audit scripts.

| Task | Actual native reference lengths | Candidate cap |
|---|---|---:|
| ETHICS | Every original/control label is1 token |1 |
| CRUXEval |800 functions/direction; required full assertion + closing tag max101 tokens |192 |
| GSM-Symbolic |5,000 worked references; median77, max192 tokens |384 |

User update (2026-09-11): raise the proposed caps to192 for both CRUXEval
directions and384 for GSM-Symbolic; ETHICS remains1. These are wired into
production but still require GPU validation. Historical saved audits describe
the earlier128/256 candidates. Resumption is authorized as recorded above.

CRUX prompts request direct assertion completion. GSM currently uses eight
worked examples and explicitly asks "Let's think step by step." Final numbers
alone use at most2 tokens, but switching to final-number-only prompting changes
the protocol and has NOT been implemented or approved as the chosen protocol.
The earlier user instruction was to keep datasets native, with ETHICS controls.

All trained sanity CRUX outputs fit within61 tokens. Trained GSM outputs fit
within192 except one incorrect Stage-2 run-on that hit2048. Approximate offline
prefix screening preserves GSM correctness at256;128 loses some correct
answers. This screening is not an actual GPU rerun and does not guarantee full
benchmark parity. Initialization generated64 capped, wrong outputs.

**The shorter-cap GPU pilot was NOT run.** Its script and9 local tests passed;
script/tests were copied to the host immediately before the user stopped work,
but no calibration command was launched. The full-size probe GPU pilot and
full experiment were also NOT run.

## Prepared next-stage work (not executed)

- `benchmark_caps.py`: repeat identical saved native prompts on init/Stage2/RLVR2
  at coding192/math384, BF16 SDPA batch8/context4096; this resume uses all seven
  checkpoints. Saves lost/gained answers,
  truncation and timings. Requires `--confirm-cost`, fresh output, cached models;
  internal900-second budget, use an external timeout too.
- `pilot_full_plan.py`: locally tested (13 tests) full-size probe pilot on RLVR2,
  all eight tasks, actual full probe rows and five shuffled-label fits; finite/
  identity/split/scaler audit; tiny behavior and circuit/intervention smoke.
  Has hard timeout supervisor and `pilot_status.json` gate.
- `next_stage.py`: locally tested (10 tests) pilot-gated full-run controller,
  all seven stages, explicit generation cap, $25 evaluation wall-time guard,
  process-group termination on timeout, no automatic instance actions.
  It accepts a global fallback plus coding/math generation-cap overrides,
  propagated through both the full-plan pilot and production runner.
- Full native CPU plan: `geode-store/olmo2-full-preflight/full-plan.json`,
  prepared236.35s,106,772,993 bytes. Fingerprint
  `cadc9ef9cbca6ce2b6c7de6402f1a353c37905fb39cbae9bdd33ba17cebce853`;
  file SHA256 `b83e35fb7534a037fcb7d4fdba0a97603bffa921cb11de21171af71510c1558f`.
  All prior workload counts matched;15 plan tests passed. Changing runner/data
  source invalidates its binding and requires a new plan. Generation cap alone
  is not in its sample binding, but must still be recorded in run metadata.
- Full counts per checkpoint:186,312 behavior rows;5,109 attribution rows;
 1,181 intervention rows;17,484 probe rows +1,212 answer-only. GSM attribution
  has only38 independent source-pair clusters, interventions8 pairs, probes47
  groups. Repeated rows must not be treated as independent.
- Old full-runtime projection21.74h/$13.24 assumes2048 tokens and should NOT be
  reused for the shorter caps. Full-size probe fitting remains an unmeasured
  nonlinear runtime risk. GPU reference: tested RTX6000 Ada48GB,100GBdisk;
  after-load allocation peaks around6.1GB do not validate a smaller GPU.

## If the user asks to resume

1. Verify final shutdown state and saved artifacts; do not assume an active GPU.
2. Reuse actual prompt/length findings; do not rerun completed sanity checks.
3. Finish cap validation on the same saved examples before accepting192/384.
4. Implement/test any per-task cap change, then rebuild the full frozen plan
   if its source binding changes. Keep fixed batch8 for comparability.
5. Run full-size probe pilot and audit; revise runtime/cost from its measurements.
6. Only when resumed/authorized, run the full experiment behind the tested gate,
   save incremental backups, verify hashes, create the final report and privately
   upload artifacts. Arrange explicit cleanup; stopping preserves disk, destroying
   deletes it. The latest user instruction ended the previous rental completely.

Useful local paths: source data `/tmp/olmo-circuit-data`; pinned metadata/tokenizers
`geode-store/olmo2-sanity-preflight/checkpoints/`; exact original runtime archive
`geode-store/olmo2-sanity-20260910/runtime_source.tar.gz`.
No active processes need resuming. Any old exec session IDs are stale after shutdown.
