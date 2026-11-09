"""Fetch daily gasprices.aaa.com snapshots from the Wayback Machine and parse the national regular average."""
import gzip, re, sys, time, csv, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
wb = Path(sys.argv[1]); out = wb / "html"; out.mkdir(exist_ok=True)
rows = [l.split() for l in open(wb / "cdx.txt") if l.strip()]

def fetch(r):
    ts, orig = r
    p = out / f"{ts}.html"
    if p.exists() and p.stat().st_size > 1000: return
    for k in range(6):
        try:
            req = urllib.request.Request(f"http://web.archive.org/web/{ts}id_/{orig}", headers={"User-Agent": "gas-research-script"})
            b = urllib.request.urlopen(req, timeout=60).read()
            p.write_bytes(b); return
        except Exception as e:
            time.sleep(15 * (k + 1))
    print("FAIL", ts, file=sys.stderr)

if "--fetch" in sys.argv:
    with ThreadPoolExecutor(3) as ex: list(ex.map(fetch, rows))

num = r"\$\s*([0-9]+\.[0-9]+)"
res = []
for p in sorted(out.glob("2*.html")):
    b = p.read_bytes()
    if b[:2] == b"\x1f\x8b": b = gzip.decompress(b)
    h = re.sub(r"\s+", " ", b.decode("utf8", "ignore"))
    cur = re.search(r"Current Avg\.?\s*</td>\s*<td[^>]*>\s*" + num, h)
    yes = re.search(r"Yesterday Avg\.?\s*</td>\s*<td[^>]*>\s*" + num, h)
    d = re.search(r"(?:Price|Prices updated) as of\s*(?:<br\s*/?>)?\s*(\d+)/(\d+)/(\d+)", h)
    if not cur: print("NOPARSE", p.name, file=sys.stderr); continue
    if d:
        m, dd, y = map(int, d.groups()); y += 2000 if y < 100 else 0
        date = datetime(y, m, dd).date(); src = "page_date"
    else:   # no date stamp: snapshot timestamp converted to US Eastern (approx UTC-5)
        date = (datetime.strptime(p.stem, "%Y%m%d%H%M%S") - timedelta(hours=5)).date(); src = "snapshot_ts"
    res.append((date, float(cur.group(1)), float(yes.group(1)) if yes else None, p.stem, src))
with open(wb / "aaa_wayback_raw.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date", "current", "yesterday", "snapshot", "date_src"]); w.writerows(res)
print(len(res), "parsed")
