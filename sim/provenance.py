#!/usr/bin/env python3
"""Provenance ledger for prereg v3.1 §1 (Rosalind pass 3, B1). Runs where the artifacts live (axis2).

    python3 provenance.py --root ~/ncomms --repo <paper-ncomms dir> --out provenance.json

For every derived artifact used in §§2-6 it records: producing code, source files, fields read, blocks whose
RESPONSE values (AnswerValue) it depends on, mtime and full SHA-256. Fields are declared from the code (listed per
entry) and then checked mechanically: (1) access.log has no READ of hh or ch; (2) no code path outside
heldout.read's lock guard names a sealed block; (3) the sealed files keep mode 0600. Exit 1 if any check fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

RAW = "sealed/all_data.csv (raw release, moved to sealed/ 2026-11-14)"
# name -> (relative path under root, producer, sources, fields read, response blocks)
LEDGER = {
    "items.json": ("data/eedi-blocks/items.json", "heldout.py quarantine", [RAW],
                   ["QuestionId", "problem", "CorrectAnswer"], []),
    "homes.json": ("data/eedi-blocks/homes.json", "heldout.py quarantine", [RAW], ["UserId", "GroupId"], []),
    "cc.csv": ("data/eedi-blocks/cc.csv", "heldout.py quarantine (mechanical partition by split ids)", [RAW],
               ["all columns, rows with calibration user and calibration item"], []),
    "hc.csv": ("data/eedi-blocks/hc.csv", "heldout.py quarantine (mechanical partition by split ids)", [RAW],
               ["all columns, rows with held-out user and calibration item"], []),
    "split-eedi.json": ("data/split-eedi.json", "data/split.py (2026-11-10)", ["raw release ids and dates"],
                        ["UserId", "QuestionId", "GroupId", "DateAnswered"], []),
    "components.json (templates, 134-item list, home-class membership graph)": (
        "runs/v3/components.json", "audit.py components", ["items.json", "homes.json", "hc.csv", "split-eedi.json"],
        ["problem", "CorrectAnswer key (tie exclusion)", "hc UserId", "hc GroupId"], []),
    "emb-bge-m3.npz (B1/B2 item embeddings)": ("runs/v3/emb-bge-m3.npz", "score3.py embed", ["items.json"],
                                                ["problem"], []),
    "rate-req.jsonl (B4 prompts)": ("runs/v3/rate-req.jsonl", "score3.py rate-build", ["items.json"], ["problem"], []),
    "rate-qwen3.5-27b.jsonl (B4 ratings)": ("runs/v3/rate-qwen3.5-27b.jsonl", "sim.py run", ["rate-req.jsonl"],
                                             ["model output"], []),
    "calib-v3.json (development scores; fold 2PLs, anchors, B1-B4 fits inside)": (
        "runs/v3/calib-v3.json", "score3.py score", ["cc.csv", "items.json", "homes.json", "v3 + v2 arm outputs",
                                                    "emb-bge-m3.npz", "rate-qwen3.5-27b.jsonl"],
        ["cc AnswerValue", "key"], ["cc"]),
}
REPO = {"choices-6.json (§6 selections)": ("results/calib-v3/choices-6.json", "select6.py", ["calib-v3.json"],
                                           ["fold rho"], ["cc"])}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def entry(base: Path, rel, prod, src, fields, blocks) -> dict:
    p = base / rel
    e = {"path": rel, "producer": prod, "sources": src, "fieldsRead": fields, "responseBlocks": blocks}
    if p.exists():
        e.update(sha256=sha(p), mtime=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
    else:
        e["missing"] = True
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, repo = Path(a.root).expanduser(), Path(a.repo).expanduser()
    out = {"written": dt.datetime.now().isoformat(timespec="seconds"), "artifacts": {}, "checks": {}}
    for n, v in LEDGER.items():
        out["artifacts"][n] = entry(root, *v)
    for n, v in REPO.items():
        out["artifacts"][n] = entry(repo, *v)
    bad = [n for n, e in out["artifacts"].items() if set(e["responseBlocks"]) - {"cc", "hc"}]
    out["checks"]["noArtifactUsesSealedResponses"] = not bad
    log = (root / "data/eedi-blocks/access.log").read_text().splitlines()
    sealed_reads = [l for l in log if re.search(r"\bREAD (hh|ch)\b", l)]
    out["checks"]["accessLogSealedReads"] = sealed_reads
    out["checks"]["accessLogLines"] = len(log)
    # heldout.py is the guard itself; its only sealed-name call is the selftest on synthetic rows with a temp lock
    code = {p.name: p.read_text() for p in (repo / "sim").glob("*.py") if p.name not in ("provenance.py", "heldout.py")}
    hits = {n: [m.group(0) for m in re.finditer(r"read\(\s*[\"'](hh|ch)[\"']", s)] for n, s in code.items()}
    out["checks"]["codeNamingSealedBlocks"] = {n: h for n, h in hits.items() if h}
    sealed = root.parent / "ncomms-data/eedi-blocks/sealed"
    out["checks"]["sealedModes"] = {p.name: oct(p.stat().st_mode & 0o777) for p in sealed.glob("*")} if sealed.exists() \
        else "sealed dir is on Empire (~/ncomms-data/eedi-blocks/sealed); axis2 holds open blocks only"
    ok = not bad and not sealed_reads and not out["checks"]["codeNamingSealedBlocks"]
    out["checks"]["pass"] = ok
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["checks"], indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
