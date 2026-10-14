"""Print the UUID of the least-loaded full (non-MIG) GPU that none of my other jobs uses.

For nodes where the scheduler counts GPUs but does not pin them (and some GPUs are MIG-partitioned):
each job then uses exactly one whole GPU, addressed by UUID, instead of every job landing on CUDA
device 0. A lock-protected reservation file prevents two of my jobs from picking the same GPU.
  export CUDA_VISIBLE_DEVICES=$(python scripts/pick_gpu.py)
"""
from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

RES = Path.home() / "cltranscpd" / "gpu_reservations"


def smi(args):
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=60).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def job_alive(jid: str) -> bool:
    if not jid or jid == "local":
        return False
    return subprocess.run(["qstat", jid], capture_output=True, timeout=30).returncode == 0


def main():
    RES.mkdir(parents=True, exist_ok=True)
    me = os.environ.get("PBS_JOBID", "local")
    with open(RES / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reserved = set()
        for f in RES.glob("*.res"):
            jid = f.read_text().strip()
            if jid != me and job_alive(jid):
                reserved.add(f.stem)
            else:
                f.unlink(missing_ok=True)
        mine = set()
        for pid, uuid in smi(["--query-compute-apps=pid,gpu_uuid"]):
            try:
                if os.stat(f"/proc/{pid}").st_uid == os.getuid():
                    mine.add(uuid)
            except OSError:
                pass
        cands = []
        for idx, uuid, mig, used, total, util in smi(
                ["--query-gpu=index,uuid,mig.mode.current,memory.used,memory.total,utilization.gpu"]):
            if mig.lower() == "enabled" or uuid in reserved or uuid in mine:
                continue
            u = float(util) if util.replace(".", "").isdigit() else 0.0
            cands.append((float(used) + 400.0 * u, uuid, idx))
        if not cands:
            raise SystemExit("no free full GPU")
        _, uuid, idx = min(cands)
        (RES / f"{uuid}.res").write_text(me)
    print(uuid)


if __name__ == "__main__":
    main()
