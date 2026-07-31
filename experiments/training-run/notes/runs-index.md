# Runs index

Archive policy: nothing is deleted; lifecycle lives in each manifest
(spec 00 §2); relay = the archive; this file is the human index.

Relay = HF `mhieuuu/geode-store` (private), mirroring `runs/<run_id>/…`.
Residency hub-verified 2026-07-26. Artifacts column: **both** = local
store + relay; **relay** = relay only (no local run dir); **local** =
local store only, not yet pushed (new-phase runs, 2026-07-26); **lost** =
no run dir anywhere (results survive in decisions.md / EXPERIMENTS.md).
For rows with no locally edited manifest, the lifecycle shown is
index-only (the manifest, if any, predates the field; absent ⇒ canonical
per spec 00 §2).

| run_id | role | lifecycle | artifacts | pointer |
|---|---|---|---|---|
| evt-run1-base | first production pretrain, G0 FAIL (min val 1.1464) | superseded (→ v3-ext) | lost — metrics only | decisions.md 2026-07-19 "Gate G0: FAIL"; EXPERIMENTS.md §2 |
| evt-run1-base-v1 | evt-run1-base under later `-v1` naming (same min val 1.1464); metrics-only history | superseded (→ v3-ext) | lost — metrics only | decisions.md 2026-07-22 "runs 2–4 closed" (v1 note) |
| evt-run1-base-v2 | base pretrain, cosine retrain of failed evt-run1-base | superseded (→ v3-ext) | both | decisions.md 2026-07-19 "G0 fix: cosine retrain" |
| evt-run1-base-v2-ext | v2 warm-start extension to convergence | superseded (→ v3-ext) | both | decisions.md 2026-07-20 "run 1 CLOSED: ext converged" |
| evt-run1-base-v3 | constant-LR retrain after run-1 re-open | superseded (→ v3-ext) | both | decisions.md 2026-07-20 "run 1 RE-OPENED: v3 constant-LR retrain" |
| evt-run1-base-v3-ext | canonical base pretrain; floor 1 = 1.0718 | canonical | both | EXPERIMENTS.md §2 run 1; decisions.md 2026-07-21 "v3 hit the ceiling" |
| evt-run2-armA-algo | Arm A capability installer (D_algo) | canonical | both | EXPERIMENTS.md §2 run 2; decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run2-sweep-lr3e-5 | run-2 installer LR re-sweep point | pilot | relay | decisions.md 2026-07-21 "run-2 re-sweep (post-V5.43)" |
| evt-run2-sweep-lr1e-4 | run-2 installer LR re-sweep point | pilot | relay | decisions.md 2026-07-21 "run-2 re-sweep (post-V5.43)" |
| evt-run2-sweep-lr3e-4 | run-2 re-sweep winner (lr pinned) | pilot | relay | decisions.md 2026-07-21 "run-2 re-sweep (post-V5.43)" |
| evt-run2-sweep-lr1e-3 | run-2 installer LR re-sweep point | pilot | relay | decisions.md 2026-07-21 "run-2 re-sweep (post-V5.43)" |
| evt-run3-armA-inst | Arm A format installer (D_inst, lr 3e-6) | canonical | both | EXPERIMENTS.md §2 run 3; decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr3e-6 | run-3 installer sweep — retention winner (run-3 pin) | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr1e-5 | run-3 installer LR sweep point | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr3e-5 | run-3 installer LR sweep point | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr1e-4 | run-3 installer LR sweep point | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr3e-4 | run-3 installer LR sweep point | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run3-sweep-lr1e-3 | run-3 installer LR sweep point | pilot | relay | decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run4-armB-inst | Arm B format installer (matched to run 3) | canonical | both | EXPERIMENTS.md §2 run 4; decisions.md 2026-07-22 "runs 2–4 closed" |
| evt-run5-armA-target | elicit target @500K | canonical | both — model weights LOST | decisions.md 2026-07-22 "Runs 5–6 complete"; 2026-07-24 "old box deleted" |
| evt-run5-pilot-n50k | OPEN(2) dataset-size pilot, Arm A ref | pilot | both | decisions.md 2026-07-22 "OPEN(2) closed — target n = 500K" |
| evt-run6-armB-target | teach target @500K (matched to run 5) | canonical | both — model weights LOST | decisions.md 2026-07-22 "Runs 5–6 complete"; 2026-07-24 "old box deleted" |
| evt-run6-pilot-n10k | OPEN(2) dataset-size pilot, Arm B recipe | pilot | both | decisions.md 2026-07-22 "OPEN(2) closed — target n = 500K" |
| evt-run6-pilot-n50k | OPEN(2) dataset-size pilot, Arm B recipe | pilot | both | decisions.md 2026-07-22 "OPEN(2) closed — target n = 500K" |
| evt-run6-pilot-n200k | OPEN(2) dataset-size pilot, Arm B recipe | pilot | both | decisions.md 2026-07-22 "OPEN(2) closed — target n = 500K" |
| evt-run6-pilot-n500k | OPEN(2) dataset-size pilot, Arm B recipe | pilot | both | decisions.md 2026-07-22 "OPEN(2) closed — target n = 500K" |
| evt-run6-sweep-lr3e-5 | target-LR sweep point (OPEN(2) phase 1) | pilot | relay | decisions.md 2026-07-22 "target LR pinned 1e-3" |
| evt-run6-sweep-lr1e-4 | target-LR sweep point (OPEN(2) phase 1) | pilot | relay | decisions.md 2026-07-22 "target LR pinned 1e-3" |
| evt-run6-sweep-lr3e-4 | target-LR sweep point (OPEN(2) phase 1) | pilot | relay | decisions.md 2026-07-22 "target LR pinned 1e-3" |
| evt-run7-armA-target-1m | canonical elicit target @1M; internals source (snapshots on relay) | canonical | both | EXPERIMENTS.md §2 run 7; decisions.md 2026-07-23 "1M rerun pair" |
| evt-run7-pilotA-lr3e-3 | Arm A 1M LR pilot at 3e-3 (high plateau) | pilot | lost | decisions.md 2026-07-24 "Arm-A 3e-3 pilot" |
| evt-run7-pilotA-lr1e-3 | Arm A 1M tie-break pilot — pinned 1e-3 | pilot | lost | decisions.md 2026-07-24 "Arm-A 3e-3 pilot" |
| evt-run8-armB-target-1m | canonical teach target @1M; internals source (snapshots on relay) | canonical | both | EXPERIMENTS.md §2 run 8; decisions.md 2026-07-23 "1M rerun pair" |
| evt-run8-sweep-lr3e-4 | 1M B-arm LR sweep point | pilot | lost | decisions.md 2026-07-24 "1M B-arm LR sweep" |
| evt-run8-sweep-lr1e-3 | 1M B-arm LR sweep point | pilot | lost | decisions.md 2026-07-24 "1M B-arm LR sweep" |
| evt-run8-sweep-lr3e-3 | 1M B-arm sweep winner (3e-3 stands) | pilot | lost | decisions.md 2026-07-24 "1M sweep extension result: 3e-3 stands" |
| evt-run8-sweep-lr1e-2 | 1M B-arm edge extension ("converged" at 1.867 nats) | pilot | lost | decisions.md 2026-07-24 "1M sweep extension result: 3e-3 stands" |
| evt-run9-llama1b-inst | Llama format installer — retention destroyed (LR scope leak) | invalid (→ inst-v2) | both | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run9-llama1b-inst-v2 | canonical Llama installer (lr 3e-6); parent of run 10-v2 | canonical | both — weights relay-only | EXPERIMENTS.md §2 run 9-v2; decisions.md 2026-07-25 "runs 9-v2 / 10-v2" |
| evt-run9-smoke | Llama chain smoke test | pilot | lost | decisions.md 2026-07-24 "third batch: full-chain launcher" |
| evt-run9-sweep-lr3e-6 | run-9 installer retention sweep — winner (pin 3e-6) | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run9-sweep-lr1e-5 | run-9 installer retention sweep point | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run9-sweep-lr3e-5 | run-9 installer retention sweep point | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run9-sweep-lr1e-4 | run-9 installer retention sweep point | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run9-sweep-lr3e-4 | run-9 installer retention sweep point (1e-3 point = run 9 v1 itself) | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-llama1b-base-ref | base Llama lr=0 reference (base retention 0.3271) | pilot | relay | decisions.md 2026-07-25 "run 9's installer LR was a scope leak" |
| evt-run10-llama1b-target | Llama target on the invalid parent — measured teaching, not elicitation | invalid (→ target-v2) | both | EXPERIMENTS.md §2 run 10; decisions.md 2026-07-25 "runs 9-v2 / 10-v2" |
| evt-run10-llama1b-target-v2 | canonical Llama elicitation target (min_val 0.01323) | canonical | both | EXPERIMENTS.md §2 run 10-v2; decisions.md 2026-07-25 "runs 9-v2 / 10-v2" |
| evt-run10-smoke | Llama chain smoke test (memory worst case) | pilot | lost | decisions.md 2026-07-24 "third batch: full-chain launcher" |
| evt-run10-sweep-lr3e-3 | Llama target LR sweep point (plateau; min at step 1) | pilot | both | decisions.md 2026-07-25 "Llama target LR sweep" |
| evt-run10-sweep-lr1e-3 | Llama sweep 1e-3 slot; final table scored the incumbent run 10-v2 instead | pilot | lost | decisions.md 2026-07-25 "Llama target LR sweep" |
| evt-run10-sweep-lr3e-4 | Llama sweep winner (0.00118; stage 2 not launched) | pilot | both | decisions.md 2026-07-25 "Llama target LR sweep" |
| evt-run10-sweep-lr1e-4 | Llama sweep point (max_steps cap, still descending) | pilot | both | decisions.md 2026-07-25 "Llama target LR sweep" |
| evt-p2-cal-dose1 | new-phase dose ε/k calibration pilot, n=1 (eps 0.0; 8.9963 → 0.00026 nats, stopped by hand at 1065 of 3000) | pilot | local — logs only, no checkpoint | decisions.md 2026-07-26 "dose stopping-rule calibration" |
| evt-p2-cal-dose16 | new-phase dose ε/k calibration pilot, n=16 (eps 0.0; 15.9081 → 0.0709, killed at step 948 by a shell teardown, still descending — RERUN on the box before ε/k is pinned) | pilot | local — logs only, no checkpoint | decisions.md 2026-07-26 "dose stopping-rule calibration" |
| evt-p3-teach-inst | Phase-3 teach format/shape installer: TinyStories base → permuted-label NL addition; behavior stop @75 | canonical | relay — analysis metadata local | decisions.md 2026-07-28 "Phase-3 teaching arm built"; EXPERIMENTS.md §6 |
| evt-p3-teach-target | Phase-3 teaching EDL target @500K NL addition; G7-matched to elicit control; converged @16K | canonical | relay — analysis metadata local | decisions.md 2026-07-28 "Phase-3 teaching arm built"; EXPERIMENTS.md §6 |
| evt-p3-warm-sum-lr1e-3 | practical Phase-3 warm-start candidate, `sum` rows 261+492, LR 1e-3 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-lr1e-2 | practical Phase-3 warm-start candidate, `sum` rows 261+492, LR 1e-2 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-lr1e-1 | selected `sum` parent, LR 1e-1 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-lr1e0 | practical Phase-3 warm-start candidate, `sum` rows 261+492, LR 1e0 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-target | selected-sum residual EDL target, fixed first 100K once | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-colon-lr1e-3 | practical Phase-3 warm-start candidate, broad `:` row 27 control, LR 1e-3 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-colon-lr1e-2 | practical Phase-3 warm-start candidate, broad `:` row 27 control, LR 1e-2 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-colon-lr1e-1 | practical Phase-3 warm-start candidate, broad `:` row 27 control, LR 1e-1 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-colon-lr1e0 | selected broad `:` parent, LR 1e0 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-colon-target | selected-colon residual EDL target, fixed first 100K once | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-colon-lr1e-3 | practical Phase-3 warm-start candidate, `sum` + `:` rows, LR 1e-3 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-colon-lr1e-2 | practical Phase-3 warm-start candidate, `sum` + `:` rows, LR 1e-2 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-colon-lr1e-1 | practical Phase-3 warm-start candidate, `sum` + `:` rows, LR 1e-1 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-colon-lr1e0 | selected `sum` + `:` parent, LR 1e0 | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-p3-warm-sum-colon-target | selected-sum-colon residual EDL target, fixed first 100K once | diagnostic | both — weights relay-only | decisions.md 2026-07-28 "practical Phase-3 embedding warm-start RESULTS" |
| evt-llama-fig2-noinst-n1000 … -n681292 (18 runs: n=1000, 1468, 2154, 3162, 4642, 6813, 10000, 14678, 21544, 31623, 46416, 68129, 100000, 146780, 215443, 316228, 464159, 681292) | fig-2 Llama dataset-size sweep, noinst arm, one run per size, all converged | canonical | both — metadata only (weights pruned per plan; owner declined rerun 2026-07-31) | decisions.md 2026-07-31 "Fig-2 Llama sweep"; EXPERIMENTS.md §6.10 |
| evt-llama-fig2-noinst-n1000000 | fig-2 sweep top size, n=1M (min_val 0.005282 nats, G5 0.9951) | canonical | both — full weights + 90MB adapter sidecar on relay, sha-verified | decisions.md 2026-07-31 "Fig-2 Llama sweep"; EXPERIMENTS.md §6.10 |
| evt-llama-fig2-installer | fig-2 1-example full-FT format installer @2.0e-5 — G4 1.0000 PASS recorded, G2 0.0732 FAIL unrecorded, arm discarded (the 3.53e-4 divergence artifact was rm'd; its manifest/logs archived laptop-side) | diagnostic | relay — manifest/logs/weights | decisions.md 2026-07-31 "Fig-2 Llama sweep" installer post-mortem |
