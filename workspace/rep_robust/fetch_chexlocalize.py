"""下载 CheXlocalize 标注文件（跳过 gradcam_maps）。"""
import json
import os

import requests

tok = open("/root/.redivis_token").read().strip()
base = "https://redivis.com/api/v1"
h = {"Authorization": f"Bearer {tok}"}
outdir = "/root/autodl-tmp/chexlocalize"
os.makedirs(outdir, exist_ok=True)

url = (f"{base}/tables/aimi.chexlocalize:efx9:v1_0.chexlocalize:jcje/"
       "rows?maxResults=3000")
r = requests.get(url, headers=h)
r.raise_for_status()
files = []
for line in r.text.splitlines():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    files.append((row["file_name"], row["file_id"]))
print("total files:", len(files))

targets = [(n, fid) for n, fid in files if not n.startswith("gradcam")]
print("to download:", len(targets))
for n, fid in targets:
    out = os.path.join(outdir, n)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue
    r = requests.get(f"{base}/files/{fid}", headers=h, allow_redirects=True)
    r.raise_for_status()
    open(out, "wb").write(r.content)
    print("  ", n, len(r.content))
print("done")
