#!/bin/bash -l
# One resume-safe simulation unit on Empire AI Alpha (Nature Comms plan WP2). vLLM and the client run in the same
# job, so nothing leaves Empire and no tunnel is needed. Submitted by empire-dispatch.py, which sets account, QoS,
# GPUs and wall time on the sbatch line (free burst by default; paid `test` only after a 24 h wait, capped).
#   needs env TASKENV = a file setting MODEL WEIGHTS REQ OUT (optional MAXLEN VLLM_ARGS WORKERS); the dispatcher
#   writes it, so JSON vLLM arguments never pass through sbatch --export (which splits on commas).
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=%x-%j.log
set -euo pipefail
# shellcheck source=/dev/null
source "${TASKENV:?}"
: "${MODEL:?}" "${WEIGHTS:?}" "${REQ:?}" "${OUT:?}"
R=$HOME/r391
export PATH=$R/.venv-uv/bin:$PATH HF_HOME=$R/cache/hf XDG_CACHE_HOME=$R/cache TRITON_CACHE_DIR=$R/cache/triton \
       VLLM_CACHE_ROOT=$R/cache/vllm HF_HUB_OFFLINE=1
TP=$(nvidia-smi -L | wc -l)
PORT=$((8100 + SLURM_JOB_ID % 800))
# --max-num-seqs 256: Qwen3.5 is hybrid (Mamba); the default 1024 sequences exceeds its cache blocks on one
# 96 GB RTX PRO 6000 (job 191140 crashed at start, 2026-11-11). The client sends 16 at a time.
vllm serve "$WEIGHTS" --served-model-name "$MODEL" --dtype bfloat16 --tensor-parallel-size "$TP" \
  --max-model-len "${MAXLEN:-16384}" --max-num-seqs 256 --disable-custom-all-reduce --host 127.0.0.1 --port $PORT ${VLLM_ARGS:-} \
  > "$HOME/ncomms/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT TERM
for i in $(seq 180); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
  sleep 10
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "vLLM never became healthy"; exit 1; }
echo "serving $MODEL on $(hostname):$PORT TP=$TP"
python3 "$HOME/ncomms/sim.py" run --requests "$REQ" --url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --out "$OUT" --workers "${WORKERS:-16}"
