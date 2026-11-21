"""Back-fill Kalshi daily AAA (KXAAAGASD) settlements and the 9 a.m. book for the final holdout.

    python tools/fetch_kalshi_history.py                         # default range 2026-10-20 .. 2026-11-20
    python tools/fetch_kalshi_history.py --start 2026-10-20 --end 2026-11-20

Why: tools/update_daily.py only pulls contracts that closed in the last 10 days, so Oct 21-30 2026 were never saved
and Oct 20 was saved before it settled. It also never saves bid/ask. This script fills both gaps once, by hand.

Method
  1. Settlements. For each settlement day D in [start, end], GET /markets?event_ticker=KXAAAGASD-<YYMONDD> from
     Kalshi's public trade API (no key), with the same columns and mapping as update_daily.update_kalshi. In
     data/kalshi_markets.csv, tickers not yet in the file are appended, and existing rows that are not yet
     'finalized' (e.g. Oct 20, saved while still active) are replaced in place. Finalized rows and all other tickers
     are left byte for byte as they were (the file is read as text). A day is reported if its event is missing or
     not every strike is finalized with a yes/no result.
  2. Book at open + 1 hour. For each contract, GET /series/KXAAAGASD/markets/<ticker>/candlesticks with
     period_interval=60 (hourly bars, which end on the clock hour), start_ts = open_time, end_ts = open_time + 7 h.
     The FIRST bar ending after open_time is used; its yes_bid.close and yes_ask.close are the best bid and ask at
     that bar's end (snap_ts). Ladders open at 8:00 a.m. ET (12:00 UTC before Nov 1, 13:00 UTC after), so for
     these contracts the bar ends at open + 1 h = 9:00 a.m. ET. Strikes added later in the day (e.g. Nov 11-13,
     open 13:11 UTC) get their first full-hour quote, less than 1 h after their open. This is the same rule as
     data/kalshi_open_book.csv (checked: for KXAAAGASD-26OCT15-4.3350 the API gives bid 0.17 / ask 0.18, equal to
     the stored row; late-added strikes there also use the first bar). Unlike that file, hours_after_open here is
     the real offset (snap_ts - open_time) / 3600, not always 1.0. No bar in the 7 h -> bid/ask left empty.
  3. Output. Written to a NEW file, data/kalshi_holdout_book.csv, columns
     ticker, event, bid, ask, snap_ts, hours_after_open, price_close, volume. data/kalshi_open_book.csv is never
     modified. gaslab.data.daily_contracts() merges the holdout book for tickers the open book does not have.
  The public API rate-limits shared IPs, so requests back off on HTTP 429 and pause briefly between calls.
"""
import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXAAAGASD"
UA = {"User-Agent": "gaslab-history-backfill"}


def get(url, timeout=40):
    """GET JSON; retries with back-off on HTTP 429 and on network errors (timeouts, resets)."""
    err = None
    for wait in (0, 5, 15, 30, 60):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
    raise RuntimeError(f"gave up after retries ({type(err).__name__}: {err}): {url}")


def event_ticker(day):
    return f"{SERIES}-{day.strftime('%y%b%d').upper()}"


def market_row(m):
    """Same mapping as tools/update_daily.update_kalshi."""
    st = {"finalized": "finalized", "settled": "finalized"}
    return dict(series=SERIES, event=m["event_ticker"], ticker=m["ticker"], strike=m.get("floor_strike"),
                strike_type=m.get("strike_type"), cap=m.get("cap_strike"), result=m.get("result") or "",
                value=m.get("expiration_value") or "", open_time=m.get("open_time"), close_time=m.get("close_time"),
                exp=m.get("expected_expiration_time") or m.get("expiration_time"),
                status=st.get(m.get("status"), m.get("status")), last=m.get("last_price_dollars"), vol=m.get("volume_fp"))


def fetch_settlements(days):
    rows, problems = [], []
    for d in days:
        ev = event_ticker(d)
        ms = get(f"{API}/markets?event_ticker={ev}&limit=200").get("markets", [])
        time.sleep(0.3)
        if not ms:
            problems.append(f"{d.date()}: no markets for {ev}"); continue
        bad = [m["ticker"] for m in ms if m.get("status") not in ("finalized", "settled") or m.get("result") not in ("yes", "no")]
        if bad:
            problems.append(f"{d.date()}: {len(bad)} of {len(ms)} strikes not settled")
        rows += [market_row(m) for m in ms]
    return pd.DataFrame(rows), problems


def fetch_book(markets):
    out = []
    for m in markets.itertuples():
        o = int(pd.Timestamp(m.open_time).timestamp())
        c = get(f"{API}/series/{SERIES}/markets/{m.ticker}/candlesticks?start_ts={o}&end_ts={o + 7 * 3600}&period_interval=60")
        time.sleep(0.2)
        bars = sorted((b for b in c.get("candlesticks", []) if b.get("end_period_ts", 0) > o), key=lambda b: b["end_period_ts"])
        bar = bars[0] if bars else None
        num = lambda x: float(x) if x not in (None, "") else None
        end = bar["end_period_ts"] if bar else None
        out.append(dict(ticker=m.ticker, event=m.event,
                        bid=num(bar["yes_bid"].get("close_dollars")) if bar else None,
                        ask=num(bar["yes_ask"].get("close_dollars")) if bar else None,
                        snap_ts=float(end) if bar else None, hours_after_open=round((end - o) / 3600, 4) if bar else None,
                        price_close=num((bar.get("price") or {}).get("close_dollars")) if bar else None,
                        volume=num(bar.get("volume_fp")) if bar else None))
    return pd.DataFrame(out)


def main(start, end):
    days = pd.date_range(start, end, freq="D")
    print(f"Settlements {start} .. {end} ({len(days)} days)")
    new, problems = fetch_settlements(days)
    path = DATA / "kalshi_markets.csv"
    cur = pd.read_csv(path, dtype=str, keep_default_na=False)          # as text, so untouched rows are written back byte for byte
    before = len(cur)
    new = new[cur.columns].fillna("").astype(str)
    pos = pd.Series(range(len(cur)), index=cur.ticker)
    old_status = cur.set_index("ticker").status
    # Existing rows are replaced in place only if not yet finalized (a settled contract never changes); new tickers
    # are appended. This keeps the file order, so the git diff shows only real changes.
    upd = new[new.ticker.isin(pos.index) & (old_status.reindex(new.ticker).values != "finalized")]
    cur.iloc[pos[upd.ticker].values] = upd.values
    cur = pd.concat([cur, new[~new.ticker.isin(pos.index)]], ignore_index=True)
    cur.to_csv(path, index=False)
    print(f"  {len(new)} contracts on {new.event.nunique()} events: {len(upd)} unsettled rows updated in place, "
          f"{len(cur) - before} added; kalshi_markets.csv {before} -> {len(cur)} rows")
    for p in problems: print("  WARNING", p)

    print("Book at open + 1 hour (hourly candlesticks)")
    book = fetch_book(new)
    book.to_csv(DATA / "kalshi_holdout_book.csv", index=False)
    miss = book[book.bid.isna() | book.ask.isna()]
    late = book[book.hours_after_open.notna() & (book.hours_after_open != 1.0)]
    print(f"  {len(book)} contracts, {len(miss)} without a quote, {len(late)} quoted less/more than 1 h after a "
          f"mid-day listing -> data/kalshi_holdout_book.csv")
    for t in miss.ticker: print("  no quote:", t)
    return 0 if not problems else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-20"); ap.add_argument("--end", default="2026-11-20")
    a = ap.parse_args()
    sys.exit(main(a.start, a.end))
