#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# UMLS: learned per-rule LOGISTIC-REGRESSION aggregator (additive, in-model)
#
# Fits one weight beta_R per rule + one shared intercept b by binary
# cross-entropy on a leave-one-out train design matrix, then evaluates in-model
# EXACTLY like the precision_binary baseline (binary activation, no per-entity
# bias) but with the learned beta_R in place of frozen PCA precision.
#
# UNIFIED RULE-ONLY PIPELINE (identical logic in the umls / wn18rr / family
# scripts; only the SBATCH scale knobs and the CHUNK_* values differ):
#
#   1. rule_logreg_train.py --l2_grid ...   -> rule_logreg.pt (beta_grid over L2)
#   2. dump_rule_counts.py                  -> counts_{valid,test}.pt
#   3. select_logreg.py --no_kge            -> rule-only OFFLINE L2 selection;
#                                              writes rule_logreg_selected.pt
#   4. main.py --logreg_binary --dump_ranks -> ranks_{valid,test}.csv AT THE
#                                              SELECTED L2 (headline number)
#
# Why offline L2 selection: the expensive grounding runs ONCE (step 2); the whole
# L2 sweep in step 3 is pure CPU arithmetic. Re-fitting/evaluating in-model per L2
# would be G full eval passes (days on WN18RR) -- this is one pass.
#
# Why rule-only: the in-model KGE fusion is disabled (src_additive/trainer.py),
# so a KGE-fused number would not correspond to any ranks_*.csv. Rule-only offline
# (alpha=0) reproduces the in-model --logreg_binary ranks EXACTLY, so the number
# step 3 reports == the number recomputed from the step-4 ranks. KGE fusion is an
# optional, separate experiment: set USE_KGE=1 (needs a working KGE dump).
#
# Compare against precision_binary / MLP offline by joining ranks on (h, r, t).
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=logreg_umls
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

SEED=42

# ── Python environment ──────────────────────────────────────────────────────
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

VENV="$HOME/RulE/rule_env"
if [ -f "${VENV}/bin/activate" ]; then
    echo "Activating virtualenv: ${VENV}"
    source "${VENV}/bin/activate"
else
    echo "ERROR: Virtualenv not found at ${VENV}."
    echo "Run setup_env_hpc.sh on the login node first: bash ~/RulE/setup_env_hpc.sh"
    exit 1
fi

echo "Python  : $(which python3)"
echo "Version : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
REPO="$HOME/RulE"
DATA_PATH="data/umls"                                   # relative to REPO (scripts chdir to repo root)
RULE_FILE="data/umls/mined_rules.txt"
CONFIG="${REPO}/config/umls_hpc.json"
KGE_CHECKPOINT="${REPO}/outputs/umls/checkpoint"        # KGE+rule pretrain (instantiates additive model)
LOGREG_DIR="${REPO}/outputs/additive_umls_aggregation_2x2/logreg"
LOGREG_FILE="${LOGREG_DIR}/rule_logreg.pt"
SELECTED_FILE="${LOGREG_DIR}/rule_logreg_selected.pt"
ANALYSIS_DIR="${REPO}/outputs/additive_umls_aggregation_2x2/analysis"

# ── Pipeline knobs (the ONLY things that differ across umls/wn18rr/family) ─────
USE_KGE=1                                          # 1 = optional KGE-fusion experiment (needs working KGE dump)
L2="1e-4"                                               # primary beta (always in the grid)
L2_GRID="1e-5 1e-4 1e-3 1e-2 1e-1 1"                    # offline L2 log grid
ALPHA_GRID="0 0.5 1 2 3 5"                              # only used when USE_KGE=1
MAX_ITER="200"
CHUNK_SIZE="0"                                          # step-1 design-matrix head/fact blocks (0 = none)
DUMP_EXTRA=()                                           # step-2 scale args (wn18rr: --device cuda --grounding_chunk 512)
KGE_EXTRA=()                                            # step-2 KGE scale args (wn18rr: --kge_chunk 16)
RUN_VERIFY=1                                            # 1 = hard-fail the job if grounding/score parity breaks
VERIFY_PARITY_ARGS=()                                   # step-0 sampling (wn18rr: --num_relations 4 --max_heads 32 --max_rules 50)
VERIFY_SCORE_ARGS=()                                    # step-2b sampling (e.g. --num_queries 32)

mkdir -p "${LOGREG_DIR}" "${ANALYSIS_DIR}"

echo "Seed           : ${SEED}"
echo "Config         : ${CONFIG}"
echo "KGE checkpoint : ${KGE_CHECKPOINT}"
echo "Logreg dir     : ${LOGREG_DIR}"
echo "Use KGE        : ${USE_KGE}"
echo "Started        : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for f in "${KGE_CHECKPOINT}" "${CONFIG}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: required input not found: ${f}"
        exit 1
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Step 0 (CPU): PARITY GATE. The offline stack (MinimalGraph) selects the L2 in
#   step 3; the in-model eval (step 4) uses the real KnowledgeGraph. This asserts
#   they ground IDENTICALLY -- exact integer path counts, forward+inverse
#   relations, with and without leave-one-out -- so the offline selection is a
#   valid stand-in for the in-model sweep. Needs only data + rules; hard-fails on
#   any mismatch. (The score-identity half runs as step 2b, once counts exist.)
# ══════════════════════════════════════════════════════════════════════════════
if [ "${RUN_VERIFY}" = "1" ]; then
    echo "[step 0] grounding parity gate (MinimalGraph == real KnowledgeGraph)..."
    cd "${REPO}"
    python3 -u scripts/verify_grounding_parity.py \
        --data_path "${DATA_PATH}" \
        --rule_file "${RULE_FILE}" \
        "${VERIFY_PARITY_ARGS[@]}"
    S0_RC=$?
    echo "[step 0] exit code: ${S0_RC}"
    [ "${S0_RC}" -eq 0 ] || { echo "ABORT: grounding parity FAILED -> offline selection would be fit against a graph the model never sees."; exit "${S0_RC}"; }
else
    echo "[step 0] RUN_VERIFY=0; skipping grounding parity gate."
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 (GPU): fit per-rule logistic regression over the L2 grid -> rule_logreg.pt
#   Design matrix built once; each L2 is a cheap refit (beta_grid [G, R]).
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${LOGREG_FILE}" ]; then
    echo "[step 1] ${LOGREG_FILE} exists; skipping logreg fit."
else
    echo "[step 1] fitting per-rule logistic regression over L2 grid (GPU, chunk_size=${CHUNK_SIZE})..."
    cd "${REPO}"
    python3 -u scripts/rule_logreg_train.py \
        --data_path  "${DATA_PATH}" \
        --rule_file  "${RULE_FILE}" \
        --out_dir    "${LOGREG_DIR}" \
        --device     cuda \
        --chunk_size "${CHUNK_SIZE}" \
        --l2         "${L2}" \
        --l2_grid    ${L2_GRID} \
        --max_iter   "${MAX_ITER}"
    S1_RC=$?
    echo "[step 1] exit code: ${S1_RC}"
    [ "${S1_RC}" -eq 0 ] || exit "${S1_RC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 (GPU): dump per-query rule firing counts (independent of beta/L2/alpha,
#   so dumped ONCE). With USE_KGE=1 also dump frozen KGE scores for the optional
#   fusion experiment.
# ══════════════════════════════════════════════════════════════════════════════
DUMP_KGE_FLAG=()
[ "${USE_KGE}" = "1" ] && DUMP_KGE_FLAG=( --dump_kge "${KGE_EXTRA[@]}" )
if [ -f "${ANALYSIS_DIR}/counts_valid.pt" ] && [ -f "${ANALYSIS_DIR}/counts_test.pt" ] \
   && { [ "${USE_KGE}" != "1" ] || { [ -f "${ANALYSIS_DIR}/kge_valid.pt" ] && [ -f "${ANALYSIS_DIR}/kge_test.pt" ]; }; }; then
    echo "[step 2] counts (+kge if requested) exist in ${ANALYSIS_DIR}; skipping dump."
else
    echo "[step 2] dumping rule firing counts (GPU, USE_KGE=${USE_KGE})..."
    cd "${REPO}"
    python3 -u scripts/dump_rule_counts.py \
        --checkpoint "${KGE_CHECKPOINT}" \
        --config     "${CONFIG}" \
        --data_path  "${DATA_PATH}" \
        --rule_file  "${RULE_FILE}" \
        --out_dir    "${ANALYSIS_DIR}" \
        --splits     valid test \
        "${DUMP_EXTRA[@]}" "${DUMP_KGE_FLAG[@]}"
    S2_RC=$?
    echo "[step 2] exit code: ${S2_RC}"
    [ "${S2_RC}" -eq 0 ] || exit "${S2_RC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 2b (CPU): SCORE-IDENTITY GATE. Confirms the real RulE.forward score ==
#   sum_R beta_R * 1[rule R fired], reconstructed from BOTH the dumped counts and
#   a fresh grounding, and that gold ranks agree -- i.e. the offline alpha=0
#   number the selection uses IS the in-model number. Uses the fitted beta from
#   step 1 over the step-2 counts. Hard-fails on any mismatch.
# ══════════════════════════════════════════════════════════════════════════════
if [ "${RUN_VERIFY}" = "1" ]; then
    echo "[step 2b] score-identity gate (offline alpha=0 == in-model)..."
    cd "${REPO}"
    python3 -u scripts/verify_score_identity.py \
        --data_path   "${DATA_PATH}" \
        --config      "${CONFIG}" \
        --counts      "${ANALYSIS_DIR}/counts_valid.pt" \
        --logreg_file "${LOGREG_FILE}" \
        "${VERIFY_SCORE_ARGS[@]}"
    S2B_RC=$?
    echo "[step 2b] exit code: ${S2B_RC}"
    [ "${S2B_RC}" -eq 0 ] || { echo "ABORT: score identity FAILED -> the offline number would not match the in-model dump."; exit "${S2B_RC}"; }
else
    echo "[step 2b] RUN_VERIFY=0; skipping score-identity gate."
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 (CPU): offline L2 selection on rule-only valid MRR (--no_kge). Writes
#   rule_logreg_selected.pt (the single beta the pipeline commits to) and
#   logreg_selection.json. With USE_KGE=1 the alpha sweep is added on top.
# ══════════════════════════════════════════════════════════════════════════════
echo "[step 3] offline L2 selection (CPU, USE_KGE=${USE_KGE})..."
NOKGE_FLAG=( --no_kge )
[ "${USE_KGE}" = "1" ] && NOKGE_FLAG=()
cd "${REPO}"
python3 -u scripts/select_logreg.py \
    --logreg_file  "${LOGREG_FILE}" \
    --analysis_dir "${ANALYSIS_DIR}" \
    --data_path    "${DATA_PATH}" \
    --out_dir      "${LOGREG_DIR}" \
    --alpha_grid   ${ALPHA_GRID} \
    --select_split valid \
    --report_split test \
    --splits       valid test \
    "${NOKGE_FLAG[@]}"
S3_RC=$?
echo "[step 3] exit code: ${S3_RC}"
[ "${S3_RC}" -eq 0 ] || exit "${S3_RC}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 (GPU): dump in-model ranks AT THE SELECTED L2 -> ranks_{valid,test}.csv.
#   Recomputing MRR from these matches the rule-only number step 3 reported, so
#   the headline metric corresponds to a real, reproducible in-model dump.
#   Re-runs whenever the selection is newer than the existing ranks.
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${LOGREG_DIR}/ranks_valid.csv" ] && [ -f "${LOGREG_DIR}/ranks_test.csv" ] \
   && [ ! "${SELECTED_FILE}" -nt "${LOGREG_DIR}/ranks_valid.csv" ]; then
    echo "[step 4] ranks_{valid,test}.csv already reflect the current selection; skipping."
else
    echo "[step 4] in-model rank dump at the selected L2 (GPU)..."
    cd "${REPO}/src_additive"
    python3 -u main.py \
        --cuda \
        --skip_pretrain \
        --pretrain_checkpoint "${KGE_CHECKPOINT}" \
        --init_checkpoint_config "${CONFIG}" \
        --logreg_binary \
        --logreg_file "${SELECTED_FILE}" \
        --dump_ranks \
        --num_iters 1 \
        --seed "${SEED}" \
        --save_path "${LOGREG_DIR}"
    S4_RC=$?
    echo "[step 4] exit code: ${S4_RC}"
    [ "${S4_RC}" -eq 0 ] || exit "${S4_RC}"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "UMLS learned logistic-regression aggregator complete (rule-only)"
echo "  weights      : ${LOGREG_FILE}  (beta_grid over L2)"
echo "  selected     : ${SELECTED_FILE}  (single beta at the chosen L2)"
echo "  selection    : ${LOGREG_DIR}/logreg_selection.json"
echo "  L2 sweep     : ${LOGREG_DIR}/logreg_alpha_sweep_{valid,test}.csv"
echo "  ranks        : ${LOGREG_DIR}/ranks_{valid,test}.csv  (in-model, AT selected L2)"
echo "  The headline number in logreg_selection.json == MRR recomputed from ranks_*.csv."
echo "  Compare vs precision_binary / MLP OFFLINE by joining on (h, r, t)."
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
