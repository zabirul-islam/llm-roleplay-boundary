#!/usr/bin/env python3
"""Keep the Nature Comms simulation units moving on Empire AI Alpha (plan WP2, docs/ncomms-paper-plan.md).

Runs on the H100 every 30 min (systemd --user timer `ncomms-dispatch`) and talks to Empire over the H100's own key
(ssh host `empire-r391`). For every unit in ~/ncomms/tasks.tsv on the H100:
  - finished (every request answered)          -> nothing;
  - a job RUNNING                               -> cancel its other pending copies;
  - no job                                       -> submit on FREE burst (account rpi, 0 SU), 4 h, --requeue;
  - burst PENDING > 24 h                         -> also submit a paid `test` copy (0.5x, 2 h), only if the total
                                                    paid SU (spent + reserved + this) stays <= CAP_SU (0 since 2026-11-14: never paid).
Jobs are resume-safe (sim.py run), so a cut-off or preempted job loses nothing. Paid SU is recomputed from sacct
on every run (no ledger to drift). Log: ~/alive/MIMIC/data/research/ncomms/dispatch.log

    python3 empire-dispatch.py [--dry-run]      # --dry-run prints what it would do, submits nothing
    python3 empire-dispatch.py selftest

tasks.tsv: id <TAB> model <TAB> weights_dir <TAB> gpus <TAB> maxlen <TAB> requests <TAB> out <TAB> vllm_args
(paths are on Empire; weights_dir relative to ~/r391/models).
"""
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path

CAP_SU, PAID_QOS, PAID_ACCOUNT, PAID_HOURS, QOS_FACTOR = 0.0, "test", "rp_sogmai_reasoningempowe", 2, {"test": 0.5, "long": 0.5, "standard": 1.0, "priority": 2.0}
WAIT_H, FREE_HOURS, SINCE = 24.0, 4, "2026-11-10"
# Owner 2026-11-14: no paid Empire at all (axis2 is back) -> CAP_SU = 0, so a paid copy is never submitted.
HOME = Path.home()
TASKS, LOG = HOME / "ncomms/tasks.tsv", HOME / "alive/MIMIC/data/research/ncomms/dispatch.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def E(cmd: str, timeout: int = 120) -> str:
    """Run on the Empire Alpha login node in a login shell (Slurm commands exist only there)."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "empire-r391", f"bash -lc {shlex.quote(cmd)}"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    return r.stdout


def paid_su(sacct_rows: list[str]) -> float:
    """sacct -P JobName|QOS|ElapsedRaw|AllocTRES -> SU used by paid ncomms jobs (Alpha base 1 SU/GPU-h)."""
    su = 0.0
    for row in sacct_rows:
        p = row.split("|")
        if len(p) < 4 or not p[0].startswith("ncomms-") or p[1] not in QOS_FACTOR:
            continue
        gpus = next((int(t.split("=")[1]) for t in p[3].split(",") if t.startswith("gres/gpu=")), 0)
        su += int(p[2] or 0) / 3600 * gpus * QOS_FACTOR[p[1]]
    return su


def decide(task: dict, jobs: list[dict], done: bool, now: dt.datetime, spent: float, reserved: float) -> list[tuple]:
    """Pure decision for one unit -> list of actions: ("cancel", jobid) / ("submit", qos) / ("log", text)."""
    if done:
        return [("cancel", j["id"]) for j in jobs] + [("log", f"{task['id']}: done")]
    running = [j for j in jobs if j["state"] == "RUNNING"]
    pending = [j for j in jobs if j["state"] == "PENDING"]
    if running:
        return [("cancel", j["id"]) for j in pending]
    if not jobs:
        return [("submit", "burst")]
    if any(j["qos"] == PAID_QOS for j in pending):
        return []
    oldest = min(j["submit"] for j in pending) if pending else now
    if (now - oldest).total_seconds() / 3600 < WAIT_H:
        return []
    est = task["gpus"] * PAID_HOURS * QOS_FACTOR[PAID_QOS]
    if spent + reserved + est > CAP_SU:
        return [("log", f"{task['id']}: free queue waited {WAIT_H:.0f}h+, but paid SU {spent + reserved:.1f} + {est:.1f} would pass the {CAP_SU:.0f} SU cap - staying on free")]
    return [("submit", PAID_QOS)]


def submit(task: dict, qos: str, dry: bool) -> str:
    env = HOME / "ncomms/tasks" / f"{task['id']}.env"
    lines = [f"MODEL={shlex.quote(task['model'])}", f"WEIGHTS=$HOME/r391/models/{shlex.quote(task['weights'])}",
             f"REQ={shlex.quote(task['req'])}", f"OUT={shlex.quote(task['out'])}", f"MAXLEN={task['maxlen']}",
             f"VLLM_ARGS={shlex.quote(task['vllm_args'])}"]
    acct, hours = ("rpi", FREE_HOURS) if qos == "burst" else (PAID_ACCOUNT, PAID_HOURS)
    cmd = (f"mkdir -p ~/ncomms/tasks ~/ncomms/logs && printf '%s\\n' {' '.join(shlex.quote(l) for l in lines)} > ~/ncomms/tasks/{task['id']}.env && "
           f"cd ~/ncomms && sbatch --parsable -J ncomms-{task['id']} --account={acct} --qos={qos} --gres=gpu:{task['gpus']} "
           f"--time={hours:02d}:00:00 --export=ALL,TASKENV=$HOME/ncomms/tasks/{task['id']}.env sim-job.sbatch")
    if dry:
        return f"(dry) {cmd}"
    return E(cmd).strip().split(";")[0]


def main(dry: bool) -> int:
    tasks = []
    for l in TASKS.read_text().splitlines() if TASKS.exists() else []:
        if l.strip() and not l.startswith("#"):
            p = l.split("\t") + [""] * 8
            tasks.append({"id": p[0], "model": p[1], "weights": p[2], "gpus": int(p[3]), "maxlen": int(p[4] or 16384),
                          "req": p[5], "out": p[6], "vllm_args": p[7]})
    if not tasks:
        log("no tasks")
        return 0
    try:
        q = E("squeue -h -u $USER -o '%i|%j|%T|%q|%V'")
        acct = E(f"sacct -u $USER -S {SINCE} -X -n -P -o JobName,QOS,ElapsedRaw,AllocTRES")
        counts = E("cd ~ && for f in " + " ".join(shlex.quote(t["req"]) + ":" + shlex.quote(t["out"]) for t in tasks) +
                   "; do r=${f%%:*}; o=${f#*:}; n=$(grep -c . \"$r\" 2>/dev/null || echo 0); "
                   "d=$(python3 -c 'import json,sys;print(len({json.loads(l)[\"id\"] for l in open(sys.argv[1]) if l.strip() and not json.loads(l)[\"failed\"]}))' \"$o\" 2>/dev/null || echo 0); echo \"$n $d\"; done")
    except Exception as e:  # noqa: BLE001
        log(f"Empire unreachable or the H100 key is missing ({e}). Nothing submitted.")
        return 0
    jobs = []
    for row in q.splitlines():
        i, name, state, qos, submit_t = (row.split("|") + [""] * 5)[:5]
        if name.startswith("ncomms-"):
            jobs.append({"id": i, "task": name[7:], "state": state, "qos": qos,
                         "submit": dt.datetime.fromisoformat(submit_t) if submit_t[:4].isdigit() else dt.datetime.now()})
    spent = paid_su(acct.splitlines())
    reserved = sum(t["gpus"] * PAID_HOURS * QOS_FACTOR[PAID_QOS] for t in tasks
                   for j in jobs if j["task"] == t["id"] and j["qos"] == PAID_QOS and j["state"] == "PENDING")
    now = dt.datetime.now()
    for t, c in zip(tasks, counts.splitlines()):
        n, d = (int(x) for x in c.split())
        for act, arg in decide(t, [j for j in jobs if j["task"] == t["id"]], n > 0 and d >= n, now, spent, reserved):
            if act == "cancel":
                log(f"{t['id']}: cancel pending copy {arg}") if not dry else print(f"(dry) scancel {arg}")
                if not dry:
                    E(f"scancel {arg}")
            elif act == "submit":
                jid = submit(t, arg, dry)
                if arg == PAID_QOS:
                    reserved += t["gpus"] * PAID_HOURS * QOS_FACTOR[PAID_QOS]
                log(f"{t['id']}: submitted on {arg} -> {jid} ({d}/{n} answered)")
            else:
                log(arg)
        st = ",".join(f"{j['id']}:{j['state']}:{j['qos']}" for j in jobs if j["task"] == t["id"]) or "-"
        log(f"{t['id']}: {d}/{n} answered, jobs {st}")
    log(f"paid SU used {spent:.2f} + reserved {reserved:.2f} of {CAP_SU:.0f} (= ${(spent + reserved) / 2:.2f} of $100)")
    return 0


def selftest() -> int:
    global CAP_SU
    t = {"id": "u1", "gpus": 2}
    cap, CAP_SU = CAP_SU, 200.0  # the decision logic is tested at the old cap, then at the live one
    now = dt.datetime(2026, 11, 12, 12)
    old, new = now - dt.timedelta(hours=30), now - dt.timedelta(hours=2)
    assert decide(t, [], False, now, 0, 0) == [("submit", "burst")]
    assert decide(t, [{"id": "1", "state": "PENDING", "qos": "burst", "submit": new}], False, now, 0, 0) == []
    assert decide(t, [{"id": "1", "state": "PENDING", "qos": "burst", "submit": old}], False, now, 0, 0) == [("submit", "test")]
    refused = decide(t, [{"id": "1", "state": "PENDING", "qos": "burst", "submit": old}], False, now, 199, 0)
    assert refused[0][0] == "log" and "cap" in refused[0][1], "must refuse past the 200 SU cap"
    assert decide(t, [{"id": "1", "state": "PENDING", "qos": "burst", "submit": old},
                      {"id": "2", "state": "PENDING", "qos": "test", "submit": new}], False, now, 0, 0) == []
    assert decide(t, [{"id": "1", "state": "RUNNING", "qos": "burst", "submit": old},
                      {"id": "2", "state": "PENDING", "qos": "test", "submit": new}], False, now, 0, 0) == [("cancel", "2")]
    assert decide(t, [{"id": "3", "state": "PENDING", "qos": "burst", "submit": new}], True, now, 0, 0)[0] == ("cancel", "3")
    CAP_SU = cap
    if CAP_SU == 0:
        never = decide(t, [{"id": "1", "state": "PENDING", "qos": "burst", "submit": old}], False, now, 0, 0)
        assert never[0][0] == "log", "with CAP_SU = 0 a long free-queue wait must never become a paid job"
    rows = ["ncomms-u1|test|7200|billing=2,cpu=16,gres/gpu=2,mem=160G,node=1", "ncomms-u1|burst|99999|gres/gpu=2",
            "sim-qwen|test|3600|gres/gpu=2", "ncomms-u2|long|3600|gres/gpu=1"]
    assert abs(paid_su(rows) - (2.0 + 0.5)) < 1e-9, paid_su(rows)
    print("dispatch selftest ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(selftest())
    sys.exit(main("--dry-run" in sys.argv))
