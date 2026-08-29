#!/usr/bin/env bash
# ts1b op-install premise program (owner + teammate, 2026-08-28): build the
# op-notation-installed TS-1B (the paper's App. I.2.1 intervention parent),
# then measure whether that capability is LATENT for natural language —
# diagnosing WHERE any transfer failure lives (format? lexical binding?) and
# testing the targeted fixes. Stages:
#
#   0  datagen           op sets + translate doses (deterministic, skip if present)
#   1  op-install        full FT on "23 + 45 = 68" until convergence
#   2  corpus census     are "sum"/"difference" even in TinyStories? (CPU)
#   3  battery baseline  9-format probe battery on ts1b-base (control, ~0 rows)
#   4  battery           same on the op-installed model  <- the premise readout
#   5  format dose       16-example bare-NL mult dose (G3-proven answer-free)
#   6  bridge doses      answer-free op<->NL rewriting: both-ops, then ADD-ONLY
#                        (word-specificity control) — each followed by the battery
#   7  small-n contrast  NL add/sub at n=100/316/1000 from op-install vs blank TS
#   8  circuits          op circuit of the install; NL circuit of the n=1000
#                        fine-tune; Jaccard between them (the reuse referee)
#
# Every stage is skip-if-done, so rerunning after a crash resumes. Branches 5,
# 6a, 6b all start from the FROZEN op-install checkpoint (never from each
# other) — the comparison arms must never see NL target data upstream.
#
# Cost: ~4-8 h GPU for the install (1M rows full FT, converges ~epoch 1);
# doses+battery+small-n+circuits add ~2-3 h. $0 on owner hardware.
#
# Usage:  bash launch_ts1b_op_program.sh --confirm-cost
#   env:  TS_CORPUS=/path/to/TinyStoriesV2-GPT4-train.txt  (stage 2; optional)
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1b-op

echo "[ts1b-op] estimated cost: ~6-11 h GPU total (install dominates); no HF uploads"
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts1b_op_program.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

TS_BASE_RID=evt-ts1b-base
TS_BASE_MODEL=$GEODE_STORE/runs/$TS_BASE_RID/model
OP_RID=evt-ts1b-op-install
OP_MODEL=$GEODE_STORE/runs/$OP_RID/model
DATA=$REPO_ROOT/experiments/training-run/data/full
ANALYSIS=$REPO_ROOT/experiments/training-run/analysis
CONFIGS=../configs

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE"
[[ -f $TS_BASE_MODEL/model.safetensors ]] ||
  fail "parent $TS_BASE_MODEL missing — pull evt-ts1b-base first"

train_or_skip() {
  local rid=$1 status
  shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
    return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid"
}

battery() { # battery <rid> — 9-format probe battery, skip if json exists
  local rid=$1
  local out="$ANALYSIS/premise_${rid}.json"
  if [[ -f $out ]]; then
    milestone "battery_skip run=$rid (exists: $out)"
    return 0
  fi
  milestone "battery_start run=$rid"
  (cd "$ANALYSIS" && python3 premise_checks.py --run-id "$rid" \
    --out "premise_${rid}.json") || fail "battery $rid"
  milestone "battery_done run=$rid -> $out"
}

# ---- stage 0: datagen (deterministic; skip if present) ---------------------
if [[ ! -f $DATA/D_algo_op.parquet || ! -f $DATA/D_translate_dose.parquet ]]; then
  (cd ../datagen &&
    python3 make_op_sets.py --out ../data/full &&
    python3 make_translate_dose.py --out ../data/full) || fail "datagen"
fi
milestone "datagen_ready (D_algo_op, D_algo_eval_op, D_translate_dose{,_add})"

# ---- stage 1: op-notation add/sub install (full FT to convergence) ---------
train_or_skip "$OP_RID" \
  python3 train_sft.py --config $CONFIGS/ts1b_op_install.yaml \
    --init-from "$TS_BASE_MODEL" --confirm-cost

# ---- stage 2: corpus lexeme census (CPU; optional, non-fatal) --------------
if [[ -n ${TS_CORPUS:-} && -f ${TS_CORPUS:-} ]]; then
  (cd "$ANALYSIS" && python3 corpus_check.py --corpus "$TS_CORPUS") ||
    milestone "corpus_check FAILED (non-fatal)"
else
  milestone "corpus_check skipped (set TS_CORPUS=/path/to/TinyStoriesV2-GPT4-train.txt)"
fi

# ---- stages 3+4: probe battery — blank control, then the measurement -------
battery "$TS_BASE_RID"   # expect ~0 in every cell: the zero line
battery "$OP_RID"        # the premise readout (bare_op high is the sanity row)

# ---- stage 5: format-dose demonstration (answer-free, G3-proven) -----------
train_or_skip evt-ts1b-op-dose \
  python3 train_sft.py --config $CONFIGS/ts1b_op_install_dose.yaml \
    --init-from "$OP_MODEL" --confirm-cost
battery evt-ts1b-op-dose

# ---- stage 6: lexical-binding bridges (each branches from op-install) ------
train_or_skip evt-ts1b-op-bridge \
  python3 train_sft.py --config $CONFIGS/ts1b_op_bridge.yaml \
    --init-from "$OP_MODEL" --confirm-cost
battery evt-ts1b-op-bridge

train_or_skip evt-ts1b-op-bridge-add \
  python3 train_sft.py --config $CONFIGS/ts1b_op_bridge_add.yaml \
    --init-from "$OP_MODEL" --confirm-cost
battery evt-ts1b-op-bridge-add
milestone "bridge_readout: compare premise_evt-ts1b-op-bridge{,-add}.json — " \
  "add-only unlocking NL '+' but not NL '-' = per-word binding (the sharp result)"

# ---- stage 7: small-n unlock contrast (elicit arm vs blank TS) -------------
for n in 100 316 1000; do
  train_or_skip "evt-ts1b-op-nl-n${n}" \
    python3 train_target.py --config $CONFIGS/ts1b_op_nl.yaml \
      --override $CONFIGS/sweeps/ts1b_op_nl/ts1b_op_nl_n${n}.yaml \
      --init-from "$OP_MODEL" --confirm-cost
done
for n in 100 316; do  # blank comparators (n>=1000 already trained in fig2ts)
  train_or_skip "evt-ts1b-fig2ts-noinst-n${n}" \
    python3 train_target.py --config $CONFIGS/ts1b_fig2ts_noinst.yaml \
      --override $CONFIGS/sweeps/ts1b_fig2ts/ts1b_fig2ts_noinst_n${n}.yaml \
      --init-from "$TS_BASE_MODEL" --confirm-cost
done
milestone "smalln_done: held-out EM/EDL per run are in each run's manifest/logs;" \
  " op-parent >> blank at n<=1000 = elicitation demonstrated (teammate's criterion)"

# ---- stage 8: circuit referee — does NL fine-tuning reuse the op circuit? --
(cd "$ANALYSIS" &&
  { [[ -f circ_ts_op.parquet ]] ||
      python3 circuit_nodes.py --run-id "$OP_RID" \
        --eval-config "$REPO_ROOT/experiments/training-run/configs/eval_op_algo_data_ts.yaml" \
        --out circ_ts_op; } &&
  { [[ -f circ_ts_opnl_n1k.parquet ]] ||
      python3 circuit_nodes.py --run-id evt-ts1b-op-nl-n1000 --out circ_ts_opnl_n1k; } &&
  python3 circuit_compare.py circ_ts_op circ_ts_opnl_n1k) || fail "circuit referee"
milestone "circuits_done: high Jaccard = NL fine-tune reuses the installed op" \
  " circuit (latent, elicited); new circuit = format-bound install"

milestone "TERMINAL_SUCCESS ts1b-op program complete — paste the battery JSONs" \
  " + circuit_compare output back for interpretation"
