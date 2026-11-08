"""For each missing date, fetch snapshots of any gasprices.aaa.com page and read the header 'National Average $x / Price as of'."""
import gzip, re, sys, csv, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
wb = Path(sys.argv[1]); missing = set(l.strip() for l in open(sys.argv[2]) if l.strip())
out = wb / "html2"; out.mkdir(exist_ok=True)
cands = {}
for line in open(wb / "cdx_all.txt"):
    ts, url = line.split()[:2]
    if url.rstrip("/").endswith(("gasprices.aaa.com",)): continue          # homepage already done
    t = datetime.strptime(ts, "%Y%m%d%H%M%S"); et = (t - timedelta(hours=5)).date()
    # AAA updates ~3-4 a.m. ET; snapshot before that still shows the previous day
    d = et if (t - timedelta(hours=5)).hour >= 5 else et - timedelta(days=1)
    k = d.isoformat()
    if k in missing: cands.setdefault(k, []).append((ts, url))
print("days with candidates", len(cands), file=sys.stderr)

def get(ts, url):
    p = out / (ts + "_" + re.sub(r"[^A-Za-z0-9]", "_", url)[-80:] + ".html")
    if not p.exists():
        for k in range(5):
            try:
                b = urllib.request.urlopen(urllib.request.Request(f"http://web.archive.org/web/{ts}id_/{url}", headers={"User-Agent": "gas-research-script"}), timeout=60).read()
                p.write_bytes(b); break
            except Exception: time.sleep(15 * (k + 1))
        else: return None
    b = p.read_bytes(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    h = re.sub(r"\s+", " ", b.decode("utf8", "ignore"))
    m = re.search(r"National Average\s*\$\s*([0-9]+\.[0-9]+).{0,200}?(?:Price|Prices updated) as of\s*(?:<br\s*/?>)?\s*(\d+)/(\d+)/(\d+)", h)
    if not m: return None
    y = int(m.group(4)); y += 2000 if y < 100 else 0
    return datetime(y, int(m.group(2)), int(m.group(3))).date().isoformat(), float(m.group(1)), ts, url

def day(k):
    found = []
    for ts, url in cands[k][:3]:
        r = get(ts, url)
        if r: found.append(r)
        if r and r[0] == k: break
    return found

rows = []
with ThreadPoolExecutor(3) as ex:
    for f in ex.map(day, sorted(cands)): rows += f
with open(wb / "aaa_wayback_fill.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date", "national", "snapshot", "url"]); w.writerows(rows)
print(len(rows), "rows;", len({r[0] for r in rows} & missing), "missing days filled")
