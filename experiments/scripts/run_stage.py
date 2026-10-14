"""Run every task of a stage inside ONE job with K concurrent worker processes.

For clusters that allow only a few running jobs per user (e.g. max_user_run = 2): one GPU job
runs many small trainings side by side on the same H100.

  python scripts/run_stage.py pretrain --workers 5
  python scripts/run_stage.py classifiers --workers 6 --shard 0 --nshards 2
Each task writes logs/<stage>/task_<i>.log; finished tasks are skipped by the scripts themselves.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import config_from  # noqa: E402

KILL_RETRIES, KILL_RETRY_DELAY_S, BUSY_RETRIES = 3, 600, 24

STAGES = {
    "pretrain": ["pretrain_clt.py", "extract_features.py"],
    "classifiers": ["train_classifiers.py"],
    "cls_baselines": ["train_cls_baselines.py"],
    "cpd": ["run_cpd.py"],
    "robustness": ["robustness.py"],
}


def n_tasks(stage, cfg):
    if stage == "pretrain":
        from pretrain_clt import tasks_pretrain as f
    elif stage == "classifiers":
        from train_classifiers import tasks_classifiers as f
    elif stage == "cls_baselines":
        from train_cls_baselines import tasks_cls_baselines as f
    elif stage == "cpd":
        from run_cpd import tasks_cpd as f
    else:
        from robustness import tasks_robustness as f
    return len(f(cfg))


def state_fn(stage, cfg):
    """task index -> 'done' | 'ready' | 'wait' for stages whose inputs appear while the job runs."""
    if stage == "classifiers":
        from train_classifiers import task_state, tasks_classifiers
        tasks = tasks_classifiers(cfg)
        return lambda i: task_state(cfg, tasks[i])
    if stage == "cpd":
        from run_cpd import tasks_cpd
        from clcpd.pipeline import run_dir
        tasks = tasks_cpd(cfg)
        res = Path(cfg.paths.work_dir) / "results" / "cpd"

        def cpd_state(i):
            method, variant, seed = tasks[i]
            name = f"clt_{variant}" if method == "clt" else method
            if (res / name / f"seed{seed}.json").exists():
                return "done"
            if method == "clt" and not (run_dir(cfg, variant, seed) / "summary.json").exists():
                return "wait"
            return "ready"
        return cpd_state
    return lambda i: "ready"


def jobs_pending(ids):
    """True while any of the PBS jobs is queued or running (finished jobs drop out of plain qstat)."""
    if not ids:
        return False
    import shutil
    try:
        qstat = shutil.which("qstat") or "/opt/pbs/bin/qstat"
        out = subprocess.run([qstat, *ids], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return any(j.split(".")[0] in out for j in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated task indices")
    ap.add_argument("--pass", dest="passthru", default="", help="extra arguments appended to every task command")
    ap.add_argument("--wait-jobs", default="",
                    help="PBS job ids producing this stage's inputs: tasks whose inputs are missing are retried "
                         "every few minutes until these jobs have finished")
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args()      # allows key=value overrides after the options
    cfg = config_from(args)
    total = n_tasks(args.stage, cfg)
    import re
    todo = [int(i) for i in re.split(r"[,+:]", args.only) if i] if args.only else list(range(total))
    todo = [i for i in todo if i % args.nshards == args.shard]
    wait_ids = [j for j in re.split(r"[,+:]", args.wait_jobs) if j]
    state = state_fn(args.stage, cfg) if wait_ids else (lambda i: "ready")
    ncpu = int(os.environ.get("NCPUS") or os.cpu_count() or 8)
    per = max(1, ncpu // max(1, args.workers))
    logdir = HERE.parent / "logs" / args.stage
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per), OPENBLAS_NUM_THREADS=str(per),
               CLCPD_THREADS=str(per), PYTHONUNBUFFERED="1")
    print(f"stage={args.stage} tasks={len(todo)}/{total} workers={args.workers} threads/worker={per}"
          + (f" waiting on {wait_ids}" if wait_ids else ""), flush=True)
    running, failed, done, deferred, retries = {}, [], 0, [], {}
    kills, busy, retry_at = {}, {}, {}        # killed by SIGKILL (e.g. a usage monitor) or locked elsewhere
    queue = list(todo)
    t0 = time.time()
    final_pass = False
    while True:
        for i, t in list(retry_at.items()):
            if t <= time.time():
                del retry_at[i]
                queue.append(i)
        while queue and len(running) < args.workers:
            i = queue.pop(0)
            st = state(i)
            if st == "done":
                continue
            if st == "wait":
                deferred.append(i)
                continue
            cmds = [[sys.executable, str(HERE / s), "--task-index", str(i), *args.passthru.split(), *args.overrides]
                    for s in STAGES[args.stage]]
            fh = open(logdir / f"task_{i}.log", "a")
            script = " && ".join(" ".join(c) for c in cmds)
            p = subprocess.Popen(["bash", "-c", script], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE.parent)
            running[i] = (p, fh, time.time())
        time.sleep(5)
        for i, (p, fh, ts) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                del running[i]
                done += 1
                status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
                if p.returncode != 0:
                    if p.returncode == 75 and busy.get(i, 0) < BUSY_RETRIES:     # task held by another job
                        busy[i] = busy.get(i, 0) + 1
                        retry_at[i] = time.time() + KILL_RETRY_DELAY_S
                        status = f"busy elsewhere; re-check in {KILL_RETRY_DELAY_S // 60} min"
                    elif p.returncode in (137, -9) and kills.get(i, 0) < KILL_RETRIES:
                        kills[i] = kills.get(i, 0) + 1
                        retry_at[i] = time.time() + KILL_RETRY_DELAY_S
                        status += f" - killed; retry {kills[i]}/{KILL_RETRIES} in {KILL_RETRY_DELAY_S // 60} min"
                    elif wait_ids and retries.get(i, 0) < 2:      # e.g. inputs still being written: retry later
                        retries[i] = retries.get(i, 0) + 1
                        deferred.append(i)
                        status += " - will retry"
                    else:
                        failed.append(i)
                print(f"[{time.strftime('%H:%M:%S')}] task {i}: {status} after {(time.time() - ts) / 60:.1f} min "
                      f"({done} finished)", flush=True)
        if queue:
            continue
        if not deferred:
            if not running and not retry_at:
                break
            continue
        if len(running) >= args.workers:
            continue
        # idle worker slots and only deferred tasks left: re-check them every few minutes
        if final_pass:
            if not running and not retry_at:
                print(f"inputs never appeared for tasks {sorted(deferred)}", flush=True)
                failed += deferred
                break
            continue
        if not jobs_pending(wait_ids):
            final_pass = True                       # producers finished: one last look at the deferred tasks
        elif not running:
            time.sleep(240)
        queue, deferred = deferred, []
        if not final_pass and running:
            time.sleep(55)
    print(f"stage {args.stage} finished in {(time.time() - t0) / 60:.1f} min; failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
