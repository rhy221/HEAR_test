#!/bin/bash
# Usage: sbatch scripts/job_lava.slurm
#        OVERRIDES="data.sample=20" sbatch scripts/job_lava.slurm   # quick smoke test
#
# Two-phase pipeline:
#   Phase 1 — pymupdf parse + Dense encode + ColQwen page/query encode → disk cache  (~15 min)
#   Phase 2 — vLLM Qwen2.5-VL-32B-AWQ inference, MaxSim retrieval from cache         (~95 min)
#   Total budget: ~110 min < 2h ✓
#
# Model: Qwen/Qwen2.5-VL-32B-Instruct-AWQ (awq_marlin, ~20GB VRAM on A100 40GB)

#SBATCH --job-name=lava_2026
#SBATCH --output=/datastore/%u/LAVA2026_2/lava26/runs/slurm-%j-early.out
#SBATCH --error=/datastore/%u/LAVA2026_2/lava26/runs/slurm-%j-early.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=mps:a100:1
#SBATCH --time=02:30:00

set -euo pipefail

# ── 1. LOAD MODULES ───────────────────────────────────────────────────────────
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# ── 2. KHAI BÁO BIẾN ─────────────────────────────────────────────────────────
REQUIRED_VRAM=40000
PROJECT_DIR=/datastore/$USER/LAVA2026_2/lava26
PYTHON=/datastore/$USER/miniconda3/envs/vqa-jv/bin/python
VLLM=/datastore/$USER/miniconda3/envs/vqa-jv/bin/vllm
MODEL_DIR=/datastore/$USER/models/Qwen2.5-VL-32B-Instruct-AWQ
COLQWEN_DIR=/datastore/$USER/models/colqwen2.5-v0.2
DENSE_DIR=/datastore/$USER/models/gte-multilingual-base
RUN_DIR="$PROJECT_DIR/runs/$SLURM_JOB_ID"
VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
VLLM_LOG="$RUN_DIR/vllm_server.log"

# ── 3. KIỂM TRA GPU ──────────────────────────────────────────────────────────
unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

# ── 4. KHỞI TẠO MPS ──────────────────────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"

# ── 5. SETUP DIRS & LOGGING ───────────────────────────────────────────────────
export TORCH_HOME=/datastore/$USER/.torch_cache
export HF_HOME=/datastore/$USER/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export COLQWEN_DIR="$COLQWEN_DIR"
export DENSE_DIR="$DENSE_DIR"
export MODEL_DIR="$MODEL_DIR"
export VLLM_PORT="$VLLM_PORT"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$RUN_DIR"

# Per-job log goes into runs/<job_id>/
exec > >(tee -a "$RUN_DIR/job.out") 2> >(tee -a "$RUN_DIR/job.err" >&2)

echo "Run dir    : $RUN_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Model dir  : $MODEL_DIR"
echo "ColQwen    : $COLQWEN_DIR"
echo "Dense      : $DENSE_DIR"

# ── 6. HF TOKEN ──────────────────────────────────────────────────────────────
HF_TOKEN_FILE="/datastore/$USER/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

# ── 7. KIỂM TRA MÔI TRƯỜNG ───────────────────────────────────────────────────
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$PROJECT_DIR" ]; then echo "ERROR: Project not found: $PROJECT_DIR"; exit 1; fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# ── 8. DOWNLOAD MODEL (nếu chưa có) ──────────────────────────────────────────
HF_CLI=/datastore/$USER/miniconda3/envs/vqa-jv/bin/hf

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Model not found. Downloading Qwen/Qwen2.5-VL-32B-Instruct-AWQ ..."
    mkdir -p "$MODEL_DIR"
    "$HF_CLI" download Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
        --local-dir "$MODEL_DIR"
    echo "Download complete."
    ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null | tail -5 || echo "(no safetensors found)"
else
    echo "Model already exists at $MODEL_DIR — skipping download."
    ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null | tail -3 || true
fi

# ── 9. PHASE 1: PREPROCESSING (ColQwen only, Qwen-VL chưa start) ─────────────
# ColQwen (~7 GiB) loads alone — plenty of room, no OOM risk.
# Encodes: page images → cache/visual/*.pt
#          all queries → cache/query_embs.pt   (MaxSim in Phase 2, no model needed)
#
# --clear-cache wipes all cache dirs before Phase 1 → accurate end-to-end timing.
OVERRIDES="${OVERRIDES:-}"

if [ ! -d "$COLQWEN_DIR" ]; then
    echo "WARNING: ColQwen model not found at $COLQWEN_DIR — visual retrieval disabled"
    COLQWEN_OVERRIDE="retriever.visual.enabled=false"
else
    COLQWEN_OVERRIDE="retriever.visual.enabled=true retriever.visual.retriever_model=$COLQWEN_DIR"
fi

if [ ! -d "$DENSE_DIR" ]; then
    echo "WARNING: Dense model not found at $DENSE_DIR — using HF hub"
    DENSE_OVERRIDE=""
else
    DENSE_OVERRIDE="retriever.dense.model=$DENSE_DIR"
fi

PREPROCESS_OVERRIDES="model.qwen_vl.path=$MODEL_DIR \
    $COLQWEN_OVERRIDE \
    $DENSE_OVERRIDE"

echo ""
echo "============================================================"
echo "  PHASE 1 — Preprocessing (pymupdf + Dense + ColQwen)"
echo "============================================================"
"$PYTHON" -u run.py \
    --config config.yaml \
    --output "runs/$SLURM_JOB_ID/results.json" \
    --preprocess \
    --clear-cache \
    $PREPROCESS_OVERRIDES $OVERRIDES
echo "Phase 1 done."

# ── 10. KHỞI ĐỘNG vLLM SERVER (sau khi preprocessing xong) ──────────────────
# ColQwen unloaded after Phase 1 — Qwen2.5-VL-32B-AWQ can use ~0.85 × 40GB.
# --enforce-eager disables CUDA graph capture to avoid memory spikes.
# awq_marlin quantization: ~20GB VRAM, leaving headroom for KV cache.
echo "Starting vLLM server (Qwen2.5-VL-32B-Instruct-AWQ)..."
mkdir -p "$RUN_DIR"

# Kill any stale process holding the port before starting fresh
STALE_PID=$(lsof -ti tcp:$VLLM_PORT 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    echo "Killing stale process on port $VLLM_PORT (PID: $STALE_PID)"
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 2
fi

"$VLLM" serve "$MODEL_DIR" \
    --port $VLLM_PORT \
    --quantization awq_marlin \
    --dtype float16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 4}' \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID (log: $VLLM_LOG)"

# Wait for server to be ready (up to 5 min)
echo "Waiting for vLLM server to start..."
VLLM_READY=0
for i in $(seq 1 100); do
    if curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}×3s"
        VLLM_READY=1
        break
    fi
    sleep 3
done

if [ $VLLM_READY -eq 0 ]; then
    echo "ERROR: vLLM server failed to start. Check: $VLLM_LOG"
    cat "$VLLM_LOG" | tail -30
    exit 1
fi

# ── 11. PHASE 2: INFERENCE (cache + MaxSim, ColQwen không load lại) ──────────
# Visual retrieval dùng pre-computed query embs + MaxSim tensor math (no model).
BASE_OVERRIDES="model.qwen_vl.path=$MODEL_DIR \
    model.qwen_vl.backend=vllm \
    model.qwen_vl.vllm_url=http://localhost:$VLLM_PORT/v1 \
    retriever.visual.enabled=false \
    $DENSE_OVERRIDE"

echo ""
echo "============================================================"
echo "  PHASE 2 — Inference (Qwen2.5-VL-32B-AWQ vLLM + MaxSim cache)"
echo "============================================================"
"$PYTHON" -u run.py \
    --config config.yaml \
    --output "runs/$SLURM_JOB_ID/results.json" \
    $BASE_OVERRIDES $OVERRIDES

# ── 12. DỌN DẸP ──────────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── 13. VERIFY SUBMISSION FORMAT ─────────────────────────────────────────────
SUBMISSION_CSV="$RUN_DIR/submission.csv"
if [ -f "$SUBMISSION_CSV" ]; then
    echo ""
    echo "Verifying submission format..."
    "$PYTHON" -m pytest tests/test_format.py \
        --csv="$SUBMISSION_CSV" \
        -v --tb=short || echo "WARNING: Format check found issues — review before submitting."
else
    echo "WARNING: submission.csv not found at $SUBMISSION_CSV"
fi

echo ""
echo "============================================================"
echo "  Job $SLURM_JOB_ID completed"
echo "  Results: $RUN_DIR"
echo "  Submission: $SUBMISSION_CSV"
echo "============================================================"
