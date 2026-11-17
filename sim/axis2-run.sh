#!/bin/bash
# One GPU of axis2 serves one model and answers a list of request files (prereg v3 calibration-side arms).
# Shared lab box: only GPUs 6-7 are ours; a GPU that is busy (> 2 GB used) is never taken - we wait instead.
#   tmux new-session -d -s ncomms-g6 "bash ~/ncomms/code/axis2-run.sh 6 qwen3.5-27b runs/v3 a0-t0.8 a2-t0.6 ..."
# Resume-safe: sim.py run skips answered ids, so re-running the same command after a crash loses nothing.
set -uo pipefail
G=$1 MODEL=$2 DIR=$3; shift 3
N=~/ncomms; cd $N
LOG=$N/logs/run-g$G.log
exec >> "$LOG" 2>&1
log() { echo "$(date '+%F %T') g$G $*"; }
case "$G" in 6|7) ;; *) log "refusing GPU $G (only 6-7 are ours)"; exit 1 ;; esac
until [ "$(nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 2000 ]; do
  log "GPU $G busy - waiting"; sleep 300
done
PORT=$((18100 + G))
ARGS=""; [ "${MODEL#qwen}" != "$MODEL" ] && ARGS='--default-chat-template-kwargs {"enable_thinking":false}'
CUDA_VISIBLE_DEVICES=$G $N/.venv/bin/vllm serve $N/models/$MODEL --served-model-name "$MODEL" --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.90 --host 127.0.0.1 --port $PORT $ARGS \
  > $N/logs/vllm-g$G.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; log "exit"' EXIT
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $PID 2>/dev/null || { log "vLLM died"; exit 1; }; sleep 10; done
log "serving $MODEL on :$PORT"
for ARM in "$@"; do
  log "$ARM start"
  python3 $N/code/sim.py run --requests $DIR/$ARM-req.jsonl --url http://127.0.0.1:$PORT --model "$MODEL" \
    --out $DIR/$ARM-$MODEL.jsonl --workers 48
  log "$ARM done: $(grep -c '"failed": null' $DIR/$ARM-$MODEL.jsonl) / $(grep -c . $DIR/$ARM-req.jsonl) answered"
done
log "all done"
