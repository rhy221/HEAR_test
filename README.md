# lava26 — LAVA 2026 Document VQA Pipeline

HEAR-inspired multi-agent Document VQA pipeline for ACM MM 2026 Grand Challenge (LAVA 2026).  
Supports Japanese + Vietnamese, open-ended answers, evidence page grounding.

## Architecture Overview

```
Stage 1: Holistic Parsing   → PyMuPDF + (MinerU) + VLM Reconciliation
Stage 2: Multimodal Retrieval → Dense (gte-multilingual) + BM25 + ColQwen + RRF Fusion
Stage 3: Multi-Agent Reasoning → Coordinator + Textual + Visual + Verification + Synthesis
Stage 4: Output Formatting  → answer + evidence_page_number (1-indexed)
```

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate vqa-jv
```

### 2. Install MeCab (optional, for Japanese tokenization)

```bash
# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3

# macOS
brew install mecab mecab-ipadic
pip install mecab-python3
```

### 3. Download models

```bash
# Main VLM (required)
hf download Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --local-dir /datastore/$USER/models/Qwen2.5-VL-32B-Instruct-AWQ

# ColQwen visual retriever (required for visual retrieval)
hf download vidore/colqwen2.5-v0.2 \
    --local-dir /datastore/$USER/models/colqwen2.5-v0.2

# Dense retriever (auto-downloads from HuggingFace if not specified)
hf download Alibaba-NLP/gte-multilingual-base \
    --local-dir /datastore/$USER/models/gte-multilingual-base
```

---

## Local Smoke Test (no GPU required)

```bash
cd /path/to/lava26

# Generate synthetic test PDFs
python tests/fixtures/create_test_pdfs.py

# Run unit tests (no models needed)
pytest tests/test_lang_detect.py tests/test_fusion.py tests/test_format.py -v

# Smoke test with mock data (5 questions, no VLM/retrievers)
python run.py \
    --config config.yaml \
    --preprocess \
    data.test_questions=tests/fixtures/questions.json \
    data.pdf_dir=tests/fixtures/pdfs \
    data.sample=5 \
    retriever.dense.enabled=false \
    retriever.sparse.enabled=false \
    retriever.visual.enabled=false \
    parsing.vlm_reconcile=false \
    parsing.use_mineru=false
```

---

## SLURM Submission

```bash
# Standard run
sbatch scripts/job_lava.slurm

# Quick smoke test (20 questions)
OVERRIDES="data.sample=20" sbatch scripts/job_lava.slurm

# Custom model dirs
MODEL_DIR=/path/to/model COLQWEN_DIR=/path/to/colqwen sbatch scripts/job_lava.slurm
```

---

## CLI Reference

```
python run.py [OPTIONS] [key=value ...]

Options:
  --config CONFIG     Config file path (default: config.yaml)
  --output OUTPUT     Override results JSON path
  --preprocess        Phase 1 only: parse + encode, skip inference
  --clear-cache       Clear all caches before run
  --help              Show this help

OmegaConf overrides:
  data.sample=5                   Process only 5 questions (smoke test)
  agents.mode=shared_vlm          All agents share one VLM (default, A100-friendly)
  agents.mode=separate_text_model Textual agent uses dedicated text model (warning: slow)
  retriever.fusion.top_k_cap=3    Retrieve top-3 pages instead of 2
  parsing.dpi=200                 Higher DPI for better image quality
  retriever.dense.enabled=false   Disable dense retrieval
  retriever.visual.enabled=false  Disable visual retrieval
```

---

## Output Format

`submission.csv`:
```csv
id,answer,evidence_page_number
q_0016,Answer,[1]
q_0020,"['Câu trả lời 1', 'Answer 2']",[1]
q_0017,1,[1]
```

- `answer`: string, number, or Python list with single quotes
- `evidence_page_number`: `[1]` or `[2, 5]` — **1-indexed**

`results.json`: Full debug info per question (retrieval scores, agent claims, verification status, timings, token usage).

---

## Troubleshoot OOM on A100 40GB

| Symptom | Fix |
|---|---|
| OOM during Phase 2 vLLM | Reduce `gpu_memory_utilization`: `model.qwen_vl.gpu_memory_utilization=0.75` |
| OOM during ColQwen Phase 1 | Reduce `retriever.visual.batch_size=1` |
| vLLM max context too long | Reduce `model.qwen_vl.max_model_len=8192` |
| Too many retrieved pages | Reduce `retriever.fusion.top_k_cap=1` |
| Slow inference | Use `agents.mode=shared_vlm` (default) |
| Timeout on complex questions | Increase `agents.per_question_timeout_s=90` |

---

## Phase Split Rationale

Phase 1 and Phase 2 are intentionally split because:
- ColQwen (~7GB) + Qwen2.5-VL-32B-AWQ (~20GB) + KV cache simultaneously would exceed 40GB VRAM
- Phase 1 encodes all page images and query embeddings to disk
- Phase 2 loads only the VLM; visual retrieval uses cached embeddings (MaxSim only, no model)
