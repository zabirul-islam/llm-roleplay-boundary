#!/usr/bin/env bash
# Export the public code repository for the NC paper (github.com/zabirul-islam/llm-roleplay-boundary).
# Allowlist copy of tracked files only, then an audit that fails closed. The manuscript, internal reviews and every
# dataset file stay private. Re-run at submission/publication, then commit + push from the target directory.
#   bash export-public.sh [TARGET]      (default ~/Desktop/Research/llm-roleplay-boundary)
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-$HOME/Desktop/Research/llm-roleplay-boundary}"
cd "$SRC"

# allowlist: tracked files under these paths only (git ls-files never lists untracked data)
FILES=$(git ls-files sim results supplementary prereg figures/make_figs.py figures/src make_numbers.py \
        data/split.py data/MANIFEST.md data/AUDIT.md export-public.sh public/README.md public/LICENSE)

# audit (fails closed): item text, per-item human values for EEDI/ASSIST, secrets
python3 - $FILES <<'EOF'
import json, re, sys
bad = []
sec = re.compile(r"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|api[_-]?key\s*[=:]\s*['\"]\S{8,})", re.I)
DBE_OK = {"results/explore/explore-postlock.json", "results/dbe/results-dbe.json"}  # DBE-KT22: unrestricted access with citation
def scan(f, o, path=""):
    if isinstance(o, dict):
        ks = list(o)
        if f not in DBE_OK and len(ks) >= 10 and sum(bool(re.fullmatch(r"\d+", str(k))) for k in ks) / len(ks) > 0.8:
            bad.append(f"{f}: item-keyed values at {path}")
        for k, v in o.items():
            scan(f, v, f"{path}/{k}")
    elif isinstance(o, list):
        for v in o[:200]:
            scan(f, v, path + "[]")
    elif isinstance(o, str) and len(o) > 200 and not path.startswith(("/historical", "/label")):
        bad.append(f"{f}: long string at {path}")
for f in sys.argv[1:]:
    if f.endswith(".json"):
        scan(f, json.load(open(f)))
    if f.endswith((".py", ".md", ".json", ".csv", ".sh", ".tsv", ".txt", ".service", ".timer", ".sbatch", ".html")):
        for m in sec.finditer(open(f, errors="ignore").read()):
            bad.append(f"{f}: secret-like {m.group(0)[:12]}")
    if re.search(r"(^|/)(all_data|Interactions|Transaction|items-assist|items)\.(csv|json)$", f) or "/sealed/" in f:
        bad.append(f"{f}: dataset file")
if bad:
    print("AUDIT FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
print(f"audit OK: {len(sys.argv) - 1} files")
EOF

mkdir -p "$DST"
for f in $FILES; do
  case "$f" in public/*) out="${f#public/}" ;; *) out="$f" ;; esac
  mkdir -p "$DST/$(dirname "$out")"; cp "$f" "$DST/$out"
done
echo "exported $(echo "$FILES" | wc -w | tr -d ' ') files to $DST"
