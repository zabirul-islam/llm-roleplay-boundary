# Nature Comms simulation runner (plan WP2)

- `sim.py` — build requests from a dataset + frozen split; resume-safe client (one JSONL line per answer).
- `sim-job.sbatch` — one unit on Empire AI Alpha: vLLM + client in the same job, no tunnel.
- `empire-dispatch.py` — runs on the H100 every 30 min (`systemd/ncomms-dispatch.{service,timer}` in
  `~/.config/systemd/user/`). Free burst first (4 h, `--requeue`); after a 24 h wait a paid `test` copy (0.5x, 2 h),
  only while paid SU (from sacct) + reserved + new <= 200 SU (= $100, approved by Ge). Log:
  `~/alive/MIMIC/data/research/ncomms/dispatch.log` on the H100.
- Units: `~/ncomms/tasks.tsv` on the H100 (id, model, weights dir under ~/r391/models, gpus, maxlen, requests, out,
  vllm args). Data and outputs stay on Empire (`~/ncomms-data/`, `~/ncomms/runs/`); EEDI is CC BY-NC-ND.
- Needs the H100's Empire key (ssh host `empire-r391`). The R391 runner deletes the key line tagged `r391-h100`
  when it finishes; rename the tag first (owner, once):
  `ssh empire-alpha "sed -i 's/ r391-h100\$/ ncomms-h100/' ~/.ssh/authorized_keys"`

## Knob selection (prereg §2) — calibration side only
`knob.py build` → 9 request files in `~/ncomms/runs/knob/` on Empire (405,396 requests: 5 grouped folds, matched
administration, per-request temperature). The 9 cells are dispatcher tasks `knob-s<σ>-t<τ>-qwen` in
`~/ncomms/tasks.tsv` on the H100 (queued 2026-11-12). When all are answered:
`python3 knob.py score --data <all_data.csv> --split <split-eedi.json> --out-dir ~/ncomms/runs/knob --model qwen3.5-27b`
→ `knob-result-qwen3.5-27b.json` (chosen cell, fold losses, invalid rate, persona adherence per cell).
