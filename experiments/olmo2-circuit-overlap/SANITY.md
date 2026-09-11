All seven pinned OLMo 2 checkpoints completed the same technical sanity workload in 39.7 minutes.  
Checkpoint load/hash checks, artifact checks, and cross-checkpoint consistency checks passed; the full experiment has not run.  
The 2,048-token cap produced 65 truncated answers, mostly at initialization; the next generation-cap setting is still being decided.

## Completed run

Run: `olmo2-sanity-20260910`, completed September 10, 2026, local time
(`2026-09-11T02:49:21Z`). Hardware: NVIDIA RTX 6000 Ada Generation.
The same frozen plan covered all seven endpoints: 16 behavioral source groups
per task, two GSM instances per template, eight attribution source pairs per
task, eight ETHICS controls, twenty target probe groups, and four independent
intervention pairs per task. Every checkpoint produced 2,224 behavioral rows.

| Checkpoint | Stage time | Peak allocated GPU memory | Truncated answers |
|---|---:|---:|---:|
| Initialization | 9.88 min | 6.10 GB | 64 |
| Stage 1 | 5.13 min | 5.67 GB | 0 |
| Stage 2 | 5.62 min | 6.06 GB | 1 |
| SFT | 4.58 min | 5.67 GB | 0 |
| DPO | 5.03 min | 5.67 GB | 0 |
| RLVR-1 | 4.77 min | 5.67 GB | 0 |
| RLVR-2 | 4.61 min | 5.67 GB | 0 |

Stage times include checkpoint startup and the behavioral, circuit,
intervention, and probe work. Overall runner time was 2,381.29 seconds;
subsequent CPU load auditing, backup, and local reporting are outside that
measurement. No behavioral prompt overflowed the 4,096-token context limit.
Source: [run metadata](../../geode-store/olmo2-sanity-20260910/run_metadata.json)
and each checkpoint's `hardware.json`/`behavior.jsonl` in the same directory.

## Integrity and backup

- **299 experiment tests passed before rental and 299 on the rented host**,
  using CPU fixtures before pretrained model loading. The repository-wide
  check retained the documented pre-existing `fig2nl2` configuration-count
  failure: 38 expected YAMLs versus 41 present.
- The [technical audit](../../geode-store/olmo2-sanity-20260910/technical_audit.json)
  passed all seven checkpoints with no failures or warnings, including matched
  inputs, grouped probe splits, and ordinary and type-matched intervention controls.
- The cached [CPU load/hash audit](../../geode-store/olmo2-sanity-20260910/load_integrity.json)
  passed all seven checkpoints: verified weight hashes and no missing,
  unexpected, or mismatched parameter keys or loader errors.
- The [local backup verification](../../geode-store/olmo2-sanity-20260910/backup_verification.json)
  passed SHA256 checks for **327 files totaling 1,507,524,187 bytes**.

## Results and limits

The [generated report](../../geode-store/olmo2-sanity-20260910/report.md)
contains six plots with takeaways, captions, identified real examples, and
interpretation caveats. These are small-sample findings: GSM uses only 32
problems from 16 templates per checkpoint. Confidence intervals concern source
groups, not independent training runs; circuit overlap and probe accuracy do
not establish acquisition or elicitation by themselves.

This sanity run retained the native benchmark generation prompts. ETHICS was
scored through single-token choices; GSM requested step-by-step reasoning;
CRUXEval used its published assertion-completion prompts. The 2,048-token
setting was a maximum generation budget, not a required answer length. All
truncated generations remain in the saved scores. A shorter native-generation
cap is being considered for the next stage; a change to the prompt protocol
would require a new frozen plan.

At final shutdown, the user instructed stopping all work, saving everything and
terminating the GPU box. **Instance50543356 was destroyed and verified absent**
at2026-09-11T03:12:44Z; no instances remain. Total rental estimate: approximately
$0.68, including setup, sanity, backup and the time it was held running.
No shorter-cap GPU pilot, full-size probe GPU pilot, or full experiment ran.
Actual references and saved outputs support testing ETHICS1 / coding128 /
math256 tokens; see the token-budget audit and [HANDOFF.md](HANDOFF.md).
The earlier two-checkpoint pilot remains documented in [PILOT.md](PILOT.md).
