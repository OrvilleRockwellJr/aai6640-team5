import gzip, re, sys, csv
from pathlib import Path
from datetime import datetime
wb = Path(sys.argv[1]); rows = []
for p in sorted((wb / "html").glob("2*.html")):
    b = p.read_bytes(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    h = re.sub(r"\s+", " ", b.decode("utf8", "ignore"))
    d = re.search(r"(?:Price|Prices updated) as of\s*(?:<br\s*/?>)?\s*(\d+)/(\d+)/(\d+)", h)
    if not d: continue
    y = int(d.group(3)); y += 2000 if y < 100 else 0
    r = {"date": datetime(y, int(d.group(1)), int(d.group(2))).date().isoformat(), "snapshot": p.stem}
    for lab in ["Current", "Yesterday", "Week Ago", "Month Ago", "Year Ago"]:
        m = re.search(lab + r" Avg\.?\s*</td>\s*<td[^>]*>\s*\$\s*([0-9]+\.[0-9]+)", h)
        r[lab] = float(m.group(1)) if m else None
    rows.append(r)
with open(wb / "aaa_wayback_ago.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
print(len(rows))
