"""NIH ChestX-ray14 official 1024px images via HTTP range on HF-hosted zips.

Reads each zip's central directory remotely (Range requests), then extracts
only the target files by fetching their local header + compressed bytes.
Avoids downloading the 45GB archive set for 326 needed images.

Usage (gpu09):
    python nih1024_fetch.py list                  # map target images -> zips
    python nih1024_fetch.py fetch                 # extract targets to OUTDIR
"""
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main"
BBOX = Path("/root/autodl-tmp/experiments/rep_robust/BBox_List_2017.csv")
OUTDIR = Path("/root/autodl-tmp/nih_cxr14_1024")
MAPJSON = OUTDIR / "zip_map.json"
ZIPS = [f"data/images/images_{i:03d}.zip" for i in range(1, 13)]


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "python-rangezip"
    return s


def resolve_url(s: requests.Session, rel: str) -> str:
    def head_():
        r = s.head(f"{REPO}/{rel}", allow_redirects=True, timeout=30)
        r.raise_for_status()
        return r.url
    return _retry(head_)


def _retry(fn, tries: int = 8, base: float = 4.0):
    import time as _t
    for i in range(tries):
        try:
            return fn()
        except requests.RequestException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if i == tries - 1:
                raise
            wait = base * (2 ** i) if code in (429, 503) else base
            print(f"  retry {i+1}/{tries} after {code}: sleep {wait}s", flush=True)
            _t.sleep(wait)


def range_get(s: requests.Session, url: str, start: int, length: int) -> bytes:
    def one():
        r = s.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"},
                  timeout=60)
        r.raise_for_status()
        return r
    return _retry(one).content


def read_central_directory(s: requests.Session, url: str) -> dict[str, tuple[int, int, int]]:
    """Return {name: (local_header_offset, compressed_size, method)}; cached on disk."""
    import hashlib
    cache = OUTDIR / "cd_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfile = cache / (hashlib.md5(url.encode()).hexdigest() + ".pkl")
    if cfile.exists():
        import pickle
        return pickle.loads(cfile.read_bytes())
    def head_():
        r = s.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers["Content-Length"])
    size = _retry(head_)
    tail = range_get(s, url, size - 65536, 65536)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    _disk, _diskcd, n_this, n_total, cd_size, cd_offset, _clen = struct.unpack(
        "<HHHHIIH", tail[eocd + 4:eocd + 22])
    cd = range_get(s, url, cd_offset, cd_size)
    out, pos = {}, 0
    for _ in range(n_total):
        assert cd[pos:pos + 4] == b"PK\x01\x02", "bad central dir"
        method = struct.unpack("<H", cd[pos + 10:pos + 12])[0]
        csize = struct.unpack("<I", cd[pos + 20:pos + 24])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        lho = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        name = cd[pos + 46:pos + 46 + nlen].decode()
        out[name] = (lho, csize, method)
        pos += 46 + nlen + elen + clen
    import pickle
    cfile.write_bytes(pickle.dumps(out))
    return out


def target_files() -> list[str]:
    bbox = pd.read_csv(BBOX, skiprows=1, header=None, usecols=range(6),
                       names=["f", "lab", "x", "y", "w", "h"])
    files = set()
    for lab in ["Cardiomegaly", "Atelectasis"]:
        files |= set(bbox[bbox.lab == lab].f.unique())
    return sorted(files)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    targets = target_files()
    print(f"targets: {len(targets)} images")
    s = session()
    zmap = {}
    if MAPJSON.exists():
        zmap = json.loads(MAPJSON.read_text())
    if mode == "list" or not zmap:
        from concurrent.futures import ThreadPoolExecutor

        def scan(rel: str):
            s2 = session()
            url = resolve_url(s2, rel)
            cd = read_central_directory(s2, url)
            bybase = {n.rsplit("/", 1)[-1]: n for n in cd
                      if not n.startswith("__MACOSX")}
            hit = [(t, bybase[t]) for t in targets if t in bybase]
            return rel, url, hit

        with ThreadPoolExecutor(max_workers=8) as ex:
            for rel, url, hit in ex.map(scan, ZIPS):
                if hit:
                    zmap[rel] = {"url": url, "files": [h[1] for h in hit]}
                    print(f"{rel}: {len(hit)} targets", flush=True)
                else:
                    print(f"{rel}: 0", flush=True)
        MAPJSON.write_text(json.dumps(zmap, indent=1))
    total = sum(len(v["files"]) for v in zmap.values())
    print(f"mapped {total}/{len(targets)}")
    if mode == "list":
        return 0
    # fetch（并行）
    from concurrent.futures import ThreadPoolExecutor

    def grab(job):
        rel, name, cd = job
        s2 = session()
        url = zmap[rel]["url"]
        lho, csize, method = cd[name]
        nlen = len(name.encode())
        head = range_get(s2, url, lho, 30 + nlen)
        assert head[:4] == b"PK\x03\x04"
        elen_local = struct.unpack("<H", head[28:30])[0]
        data = range_get(s2, url, lho + 30 + nlen + elen_local, csize)
        if method == zipfile.ZIP_DEFLATED:
            import zlib
            raw = zlib.decompress(data, -15)
        else:
            raw = data
        outp = OUTDIR / "images" / Path(name).name
        outp.parent.mkdir(exist_ok=True)
        outp.write_bytes(raw)
        return name

    jobs = []
    cds = {}
    for rel, info in sorted(zmap.items()):
        cds[rel] = read_central_directory(session(), info["url"])
        for name in info["files"]:
            jobs.append((rel, name, cds[rel]))
    print(f"cd preloaded for {len(cds)} zips", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in ex.map(grab, jobs):
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    got = len(list((OUTDIR / "images").glob("*.png")))
    print(f"extracted {got} images -> {OUTDIR / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
