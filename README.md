# RulE - Graduation Project Implementation

This repository contains my analyses on top of the RulE knowledge graph reasoning approach described in https://github.com/XiaojuanTang/RulE as part of my graduation project. The folders ``data/``, ``config/`` as well as Datasets section below are borrowed from the authors' repository. The implementation of the baseline is kept under ``RulE_original/src/`` and the paper is found under ``RulE_paper/``.

## Datasets

Datasets used are contained in the folder ``data/``. The format is as follows:

train.txt, valid.txt, test.txt: training, valid, test set with triplets; Format: (h,r,t).

mined_rules.txt: These rules are mined by RNNLogic. Format: [rule_head, rule_body_list]. For example, $r_1 \land r_2 \rightarrow r_3$ can be represented as $[r_3,r_1,r_2]$

## Reproducing the analyses

Every analysis in this README is a short, self-contained command block, runnable in a plain shell on any machine with a GPU (some steps `cd` into `RulE_original/src` or `src_additive` first).

### Shared variables

Every command below expands these. Values shown are for UMLS; swap them per dataset.

```bash
export REPO=$PWD
export DATASET=umls
export CONFIG=config/umls_config.json
export DATA_PATH=data/umls
export RULE_FILE=data/umls/mined_rules.txt
export OUT_DIR=outputs/$DATASET/rq          # one pipeline shares a single --out_dir across its steps

# Two DIFFERENT pre-trained artifacts (not interchangeable):
export CHECKPOINT=outputs/$DATASET/checkpoint   # bare KGE+rule embeddings from RulE pre-training; RQ2/RQ3
export GROUNDING=outputs/$DATASET/grounding.pt  # full RulE model incl. the MLP grounding head; RQ1
```

### Configs location

`config/` holds one JSON per dataset. Each file carries the baseline model hyper-parameters (`hidden_dim`,
`gamma_fact`, `gamma_rule`, `p_norm`, ...), the fixed mixing weight `alpha`, and
three path fields:

| field | meaning |
| --- | --- |
| `data_path` | directory holding `train/valid/test.txt` |
| `rule_file` | the mined rule list |
| `save_path` | where the training stages write checkpoints |
| `alpha` | the dataset's published fixed mixing weight, `score = rule + alpha*kge` |

These are the values behind the reported adaptive-weight results:

| | UMLS | WN18RR | FB15k-237 |
| --- | --- | --- | --- |
| `--alpha` (from config) | 2.0 | 1.0 | 3.0 |
| `--feature` | `kge_max` | `kge_entropy` | `top_rule_confidence` |
| `--epochs` | 10 | 10 | 6 |
| `--lr` | 0.001 | 0.001 | 0.001 |
| `--feature_lr` | 0.001 | 0.01 | 0.01 |


## RQ1 Adaptive mixing weight

### 1. Headroom diagnostics and feature-validity screen

`scripts/feature_validity_screen.py` computes the filtered rank of
the true tail under the rule scorer alone and the KGE scorer alone, and defines

```
rule_advantage(q) = RR_rule(q) - RR_kge(q)
```

The screen prints:

**Per-relation and per-query ceiling** — the MRR available at each level of mixing granularity, all
computed on one shared convex grid `beta*rule + (1-beta)*kge` (`--beta_grid_steps`
points in `[0, 1]`)

**Feature table** — one row per feature, sorted by `|r_adv|`:

`r_adv` is the Pearson correlation with `rule_advantage` — the headline number —
and the 95% CI is its Fisher-z interval. `r:RRrule` and `r:RRkge`
decompose it: whether the feature tracks rule strength or (inversely) KGE
strength.

**Per-bucket advantage** — for the single top feature by `|r_adv|` only, mean
`rule_advantage` across `--num_buckets` quantile bins. This is the shape check
behind the correlation: a router is usable when the advantage column moves
monotonically across buckets.

The feature table is also written to `--output` as CSV
(`feature,n,r_adv,ci_lo,ci_hi,r_RRrule,r_RRkge`).

The screen imports `data` / `model` / `utils` from `RulE_original/src`, so run this
with that directory as the working directory:

```bash
cd "$REPO/RulE_original/src"
python -u "$REPO/scripts/feature_validity_screen.py" \
    --config      "$REPO/$CONFIG" \
    --checkpoint  "$REPO/$GROUNDING" \
    --split       valid \
    --features    "density,num_rules,top_rule_confidence,kge_max,kge_entropy" \
    --num_buckets 5 \
    --output      "$REPO/$OUT_DIR/feature_validity.csv" 
cd "$REPO"
```

### 2. Train the mixing weight

`scripts/train_beta_grounding_chunked.py` computes the mixing gate:

```
beta(q) = sigmoid( beta[r] + beta_feature * feature(q) )
score(q) = beta(q) * rule + (1 - beta(q)) * kge
```

trained in two stages: stage 3 fits the per-relation intercept `beta[r]`, then
stage 6 fits the single slope `beta_feature` (learning rate `--feature_lr`) on
the standardized feature. The valid split is halved per relation — one half trains beta, the other selects the
best epoch — so test is never used for selection. `--loss ce` is filtered softmax
cross-entropy; `--loss margin` is pairwise margin ranking over sampled negatives
(`--neg_sampling`, `--num_negatives`, `--mixed_hard_frac`; `--label_smoothing` is
CE-only and inert under margin).

The sign of `beta_feature` says which way the
feature routes.

Cross-entropy run (reported in results). `--epochs`, `--feature`, `--feature_lr`
below are the UMLS values; substitute per the table in [Configs location](#configs-location)
for WN18RR / FB15k-237:

```bash
for SEED in 42 43 44; do
    cd "$REPO/RulE_original/src"
    python -u "$REPO/scripts/train_beta_grounding_chunked.py" \
        --config          "$REPO/$CONFIG" \
        --checkpoint      "$REPO/$GROUNDING" \
        --epochs 10 --lr 0.001 --feature_lr 0.001 \
        --loss ce --label_smoothing 0 --beta_l2 0.0 \
        --feature kge_max --standardize_feature \
        --seed            "$SEED" \
        --beta_checkpoint "$REPO/$OUT_DIR/beta_ce/seed${SEED}/beta.pt" \
        > "$REPO/$OUT_DIR/beta_ce_seed${SEED}.log" 2>&1
done
cd "$REPO"
```

## RQ2 Fixed-weight rule aggregators (conf x count, conf x binary, precision)

Three scorers that need **no fitting at all**. Each replaces RulE's MLP grounding
head with a plain weighted sum over the rules that fired for the query, and they
differ only in the per-rule weight `w_R` and in whether a rule's activation is its
raw path count or a 0/1 indicator:

| Aggregator | score(t) | flags to `src_additive/main.py` |
| --- | --- | --- |
| Conf x count | `sum_R conf_R * count_R(t)` | `--conf_count --no_bias` |
| Conf x binary | `sum_R conf_R * 1[count_R(t) > 0]` | `--conf_binary --no_bias` |
| Precision | `sum_R prec_R * 1[count_R(t) > 0]` | `--precision_binary --precision_file <rule_precision.pt>` |

`conf_R` is the frozen RulE rule confidence read straight out of the pre-trained
checkpoint (`gamma_rule - d`, Eq. 6 of the paper); `prec_R` is the rule's empirical
train PCA precision.
`--conf_count` is the paper's
"sum (w/o MLP)" baseline. In all three the per-entity bias is dropped and
frozen, so **no parameter is trainable** and the run is a pure evaluation:
`--num_iters 1` is enough, and the result is deterministic given the checkpoint.

These three are reported **rule-only**, with no KGE fusion. Each run's log prints both:
the `Evaluating on test` block is the rule-only score (what the table reports), the
`Evaluating on test_kge` block is `rule + alpha*kge` and is not used here.

The checkpoint comes from RulE's own pre-training stage, run once per dataset:

```bash
cd RulE_original/src
python -u main.py --init_checkpoint_config "../../$CONFIG" --save_path "../../outputs/$DATASET"
cd -
```

Everything below reuses that one checkpoint (`--skip_pretrain`), so pre-training is never repeated.

### 1. Precision only: per-rule train PCA precision

`rule_precision.pt` depends only
on the data and the rule list — no checkpoint, no training:

```bash
python scripts/rule_precision_train.py \
    --data_path "$DATA_PATH" \
    --rule_file "$RULE_FILE" \
    --out_dir   "$OUT_DIR" \
    --device    cuda
```

On WN18RR/FB15k-237 add `--chunk_size 256`: it grounds each relation's heads in
blocks for computational feasibility.

### 2. Evaluate the aggregator

```bash
cd src_additive

# Conf x count
python -u main.py --cuda --skip_pretrain \
    --pretrain_checkpoint     "../$CHECKPOINT" \
    --init_checkpoint_config  "../$CONFIG" \
    --conf_count --no_bias \
    --clamp_negative_confidence \
    --num_iters 1 --seed 42 \
    --save_path "../outputs/$DATASET/conf_count"

# Conf x binary
python -u main.py --cuda --skip_pretrain \
    --pretrain_checkpoint     "../$CHECKPOINT" \
    --init_checkpoint_config  "../$CONFIG" \
    --conf_binary --no_bias \
    --clamp_negative_confidence \
    --num_iters 1 --seed 42 \
    --save_path "../outputs/$DATASET/conf_binary"

# Precision x binary
python -u main.py --cuda --skip_pretrain \
    --pretrain_checkpoint     "../$CHECKPOINT" \
    --init_checkpoint_config  "../$CONFIG" \
    --precision_binary \
    --precision_file "../$OUT_DIR/rule_precision.pt" \
    --num_iters 1 --seed 42 \
    --save_path "../outputs/$DATASET/precision_binary"

cd -
```

### On `--clamp_negative_confidence` flag

`conf_R = gamma_rule - d` is unbounded below, so some rules carry a negative
confidence and *subtract* from a candidate they support.
`--clamp_negative_confidence` sets `w_R = max(0, w_R)` before aggregation (it
affects the two confidence-weighted rows only; precision is non-negative by
construction).

## RQ2 Fitted rule aggregators (logistic regression / OLS / linear SVM)

On top of the pretrained KGE+rule checkpoint, `scripts/rule_logreg_train.py` fits an
interpretable **additive** per-rule aggregator: one weight `beta_R` per rule plus one
shared intercept, so that

```
score(t) = sum_R beta_R * 1[rule R fired h -> t]   (+ intercept, constant per query)
```

`--loss` selects what `beta_R` means: `bce` (default) = **logistic regression**
(beta = log-odds, calibrated), `mse` = **OLS / linear probability model**, `squared_hinge`
= **linear SVM** (beta = margin coefficient). All three share the identical score form,
are fit with L2-regularised L-BFGS, and are swept over an L2 grid and selected
offline. 
Because each objective is convex, `beta` is
zero-initialised, and the whole fit is deterministic, re-running reproduces
identical coefficients.

### 1. Fit the per-rule weights over an L2 grid

```bash
python scripts/rule_logreg_train.py \
    --data_path  "$DATA_PATH" \
    --rule_file  "$RULE_FILE" \
    --out_dir    "$OUT_DIR" \
    --loss       bce \
    --l2         1e-4 \
    --l2_grid    1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 \
    --design_cache "$OUT_DIR/design_matrix.pt" \
    --device     cuda
```

Writes `rule_logreg.pt`, holding one `beta` vector per grid value (step 4 selects
among them), plus `design_matrix.pt` — the exact leave-one-out firing matrix the
fit used, reused by the RQ3 collinearity check. Only the train graph is read
here; no held-out data. For OLS pass `--loss mse`; for linear SVM pass
`--loss squared_hinge`. Every fit prints a convergence certificate.

### 2. Dump per-query rule firing counts (needed for offline selection/eval)

```bash
python scripts/dump_rule_counts.py \
    --checkpoint "$CHECKPOINT" \
    --config     "$CONFIG" \
    --data_path  "$DATA_PATH" \
    --rule_file  "$RULE_FILE" \
    --out_dir    "$OUT_DIR" \
    --splits     valid test \
    --dump_kge
```

`--dump_kge` writes `kge_{valid,test}.pt`, which step 4 needs for the alpha
sweep. Without them `select_logreg.py` silently runs rule-only and pins
`alpha = 0`.

### 3. Optional verification

```bash

# score identity: score_model(t) == sum_R beta_R * 1[rule R fired]
python scripts/verify_score_identity.py \
    --data_path   "$DATA_PATH" \
    --config      "$CONFIG" \
    --counts      $OUT_DIR/counts_valid.pt \
    --logreg_file $OUT_DIR/rule_logreg.pt
```

### 4. Select L2 and alpha on validation MRR

```bash
python scripts/select_logreg.py \
    --logreg_file  $OUT_DIR/rule_logreg.pt \
    --analysis_dir "$OUT_DIR" \
    --data_path    "$DATA_PATH" \
    --out_dir      "$OUT_DIR" \
    --select_split valid \
    --report_split test \
    --splits       valid test \
    --alpha_grid 0 0.5 1 2 3 5
```

Two stages, both on the validation split: L2 is picked on **rule-only** MRR
(`alpha = 0`), then `alpha` on the **combined** (`rule + alpha*kge`) MRR at that
fixed L2; the test split is scored once at the selected pair. Needs
`kge_{valid,test}.pt` from step 2. Writes `rule_logreg_selected.pt` and
`logreg_selection.json` (selected `l2`/`alpha`, plus `valid_rule_only_MRR` and
the combined `test_metrics`).

### 5. Dump final in-model ranks at the selected hyperparameters

```bash
cd src_additive
python main.py \
    --cuda --skip_pretrain \
    --pretrain_checkpoint "$CHECKPOINT" \
    --init_checkpoint_config "$CONFIG" \
    --logreg_binary \
    --logreg_file $OUT_DIR/rule_logreg_selected.pt \
    --dump_ranks --num_iters 1 \
    --save_path "$OUT_DIR"
```

`--dump_ranks` writes the **rule-only** in-model ranking (KGE fusion is off in
this path), so recomputing MRR from `ranks_{valid,test}.csv` reproduces
`logreg_selection.json`'s rule-only figure (`valid_rule_only_MRR` / the
`alpha = 0` test row) from a real model forward pass. The combined
`rule + alpha*kge` headline is the offline number in `logreg_selection.json`
(`test_metrics`); `--alpha` here only changes the `test_kge` log line, not the
dumped CSV.

### Selected hyper-parameters (what the reported numbers use)

| Dataset | LogReg (`bce`) | OLS (`mse`) | Linear SVM (`squared_hinge`) |
| --- | --- | --- | --- |
| Family | L2 `1e-4`, alpha `2.0` | L2 `1e-3`, alpha `0.5` | L2 `1e-3`, alpha `0.5` |
| UMLS | L2 `1e-4`, alpha `1.0` | L2 `1e-3`, alpha `0.5` | L2 `1e-3`, alpha `0.5` |
| WN18RR | L2 `1e-6`, alpha `1.0` | L2 `1e-5`, alpha `0.5` | L2 `1e-5`, alpha `0.5` |

## RQ3 Interpretability trust table

To characterise, per rule, whether its `beta` is individually trustworthy (enough
support, not separated, not collinear with another rule). `analyze_rule_collinearity.py`
reads `design_matrix.pt` from `--logreg_dir`, so run the fitted-aggregator step 1
with `--design_cache "$OUT_DIR/design_matrix.pt"` first (as shown above).

```bash
python scripts/analyze_rule_collinearity.py --logreg_dir "$OUT_DIR"

python scripts/build_rule_trust_table.py --logreg_dir "$OUT_DIR"

# ECDF figure comparing datasets; repeat --trust_table per dataset (LABEL:PATH)
python scripts/plot_trust_table_ecdfs.py \
    --trust_table "$DATASET:$OUT_DIR/rule_trust_table.csv" \
    --out "$OUT_DIR/rule_trust_ecdfs.png"
```

Writes `rule_trust_table.csv` (per-rule `support`, `true_fired`,
`minority_support`, a derived `separated` flag, `precision`, `max_jaccard`,
`vif`) and `rule_trust_summary.json`.
