r"""下载 CheXphoto train natural/iphone（iPhone 1k）frontal 图。
端点：GET /rawFiles/{file_id}（SDK File.download 的实际路径），
并发 12，断点续传。token 在 /root/.redivis_token，不打印。
"""
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

tok = open("/root/.redivis_token").read().strip()
h = {"Authorization": f"Bearer {tok}"}
base = "https://redivis.com/api/v1"
DEST = pathlib.Path("/root/autodl-tmp/chexphoto/train/natural/iphone")
DEST.mkdir(parents=True, exist_ok=True)
W = 12
_lock = threading.Lock()
_done = {"n": 0, "skip": 0, "fail": 0}


def fetch(row):
    stem = row["file_name"][len("natural/iphone/"):]
    target = DEST / stem
    target.parent.mkdir(parents=True, exist_ok=True)
    size = int(row["size"])
    if target.exists() and target.stat().st_size == size:
        with _lock:
            _done["n"] += 1; _done["skip"] += 1
        return None
    rr = requests.get(f"{base}/rawFiles/{row['file_id']}", headers=h, timeout=120)
    if rr.status_code != 200:
        return f"{stem}: HTTP {rr.status_code}"
    open(target, "wb").write(rr.content)
    with _lock:
        _done["n"] += 1
        if _done["n"] % 50 == 0:
            print(f"  [{_done['n']}] skip {_done['skip']} fail {_done['fail']}", flush=True)
    return None


def main():
    r = requests.get(
        f"{base}/tables/aimi.chexphoto:2qwg:v1_0.train:pbtn/rows?maxResults=100000",
        headers=h, timeout=120)
    r.raise_for_status()
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()
            and l.lstrip().startswith("{")]
    rows = [x for x in rows if isinstance(x, dict) and "file_name" in x
            and x["file_name"].startswith("natural/iphone/")
            and x["file_name"].endswith("_frontal.jpg")]
    print(f"targets: {len(rows)}", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(fetch, x) for x in rows]
        for fu in as_completed(futs):
            e = fu.result()
            if e:
                errors.append(e)
    print(f"done: ok {_done['n'] - _done['skip'] - _done['fail']} skip {_done['skip']} fail {_done['fail']}")
    for e in errors[:10]:
        print("  ", e)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
