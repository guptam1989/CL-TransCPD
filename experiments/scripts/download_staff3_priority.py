"""Prioritised, resumable, parallel download of the STAFF III files the benchmark needs.

Useful when PhysioNet is slow: headers and inflation annotations first, then the balloon-
inflation recordings, then cath-lab baselines, then room baselines (change-free data).

  python scripts/download_staff3_priority.py --dest $DATA_ROOT/staffiii/1.0.0 --threads 12
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.request as u
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "https://physionet.org/files/staffiii/1.0.0/"


def fetch(rel: str, dest: Path, retries: int = 6):
    out = dest / rel
    if out.exists() and out.stat().st_size > 0:
        return rel, "exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    for k in range(retries):
        try:
            start = part.stat().st_size if part.exists() else 0
            req = u.Request(BASE + rel, headers={"Range": f"bytes={start}-"} if start else {})
            with u.urlopen(req, timeout=180) as r:
                mode = "ab" if (start and r.status == 206) else "wb"
                with open(part, mode) as fh:
                    while True:
                        b = r.read(1 << 16)
                        if not b:
                            break
                        fh.write(b)
            part.rename(out)
            return rel, "ok"
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "code", None) == 404:
                return rel, "404"
            time.sleep(10 * (k + 1))
    return rel, "failed"


def run(batch, dest, threads, label):
    t0 = time.time()
    with cf.ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda r: fetch(r, dest), batch))
    bad = [r for r, s in res if s == "failed"]
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {len(batch)} files in {(time.time() - t0) / 60:.1f} min, "
          f"failed {len(bad)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-null", type=int, default=40)
    ap.add_argument("--exclude", default="1,4,5,6,89")
    args = ap.parse_args()
    dest = Path(args.dest)
    run(["RECORDS", "STAFF-III-Database-Annotations.xlsx", "ANNOTATORS"], dest, 3, "metadata")
    from clcpd.data.staff3 import _rec_name, parse_sheet

    sheet = parse_sheet(dest)
    excl = {int(x) for x in args.exclude.split(",") if x}
    rows = [r for _, r in sheet.iterrows() if int(r["#"]) not in excl]
    bi, bc, br = [], [], []
    for r in rows:
        bi += [n for n in (_rec_name(r.get(f"BI{k}")) for k in range(1, 6)) if n]
        bc += [n for n in (_rec_name(r.get(c)) for c in ("BC2", "BC1")) if n]
        n = _rec_name(r.get("BR"))
        if n:
            br.append(n)
    bi = list(dict.fromkeys(bi))
    hea = [f"data/{n}.hea" for n in dict.fromkeys(bi + bc + br)]
    run(hea + [f"data/{n}.event" for n in bi], dest, args.threads, "headers + inflation annotations")
    run([f"data/{n}.dat" for n in bi], dest, args.threads, "inflation recordings")
    run([f"data/{n}.dat" for n in bc], dest, args.threads, "cath-lab baselines")
    run([f"data/{n}.dat" for n in br[:args.max_null]], dest, args.threads, "room baselines (null streams)")
    print("STAFF III priority download complete", flush=True)


if __name__ == "__main__":
    main()
