"""
Cluster Scanner — Botak & Engulfing + Record Breakers
Scans KNOWN_STOCKS for today's bar matching the "botak" or "bullish engulfing"
pattern, groups hits by industry, and sends a Telegram alert if any industry
has enough same-day hits to count as a cluster (mirrors the thresholds used
in the Streamlit dashboard: engulfing > 1 ticker per industry, botak > 2).

Also scans every ticker's own daily close-to-close % change history over
RECORD_LOOKBACK_PERIOD and flags any ticker where TODAY's % move is a new
all-time high (within that window) single-day % up-move or % down-move.

Intended to be run on a schedule (every 30 min during market hours) by
GitHub Actions. Safe to also run manually / locally for testing.
"""

import os
import sys
import json
import datetime
import requests
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import INDUSTRIES, KNOWN_STOCKS

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

STATE_FILE = "cluster_state.json"

# Thresholds — mirrors the original Streamlit logic:
#   engulf_industry_count > 1   -> at least 2 tickers
#   botak_industry_count  > 2   -> at least 3 tickers
ENGULF_MIN_PER_INDUSTRY = 2
BOTAK_MIN_PER_INDUSTRY = 3

# How far back to look when establishing each ticker's own record for the
# single-day highest % up-move and highest % down-move. Adjust as needed
# (e.g. "6mo", "2y", "5y") — longer = harder record to break, more meaningful.
RECORD_LOOKBACK_PERIOD = "1y"

MARKET_TZ = ZoneInfo("America/New_York")


# ------------------------------------------------------------------------
# MARKET HOURS GUARD
# ------------------------------------------------------------------------
def is_market_open_now() -> bool:
    """True on NYSE trading hours (weekday 9:30am-4:00pm ET).
    Does NOT account for market holidays — GitHub Actions will still fire
    on holidays, but this script will just find no fresh clusters and
    (harmlessly) do nothing since price data won't have moved."""
    now_et = datetime.datetime.now(MARKET_TZ)
    if now_et.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t


# ------------------------------------------------------------------------
# DATA + PATTERN DETECTION
# ------------------------------------------------------------------------
def download_data(tickers, period="5d"):
    """Download OHLC data for `tickers` over `period` (daily interval).
    Returns {ticker: DataFrame} for tickers with at least 2 valid rows."""
    raw = yf.download(
        tickers, period=period, interval="1d", progress=False, auto_adjust=True
    )
    dfs = {}
    for t in tickers:
        try:
            df = pd.DataFrame(
                {
                    "Open": raw["Open"][t],
                    "High": raw["High"][t],
                    "Low": raw["Low"][t],
                    "Close": raw["Close"][t],
                }
            ).dropna()
            if len(df) >= 2:
                dfs[t] = df
        except Exception:
            continue
    return dfs


def detect_botak_today(df: pd.DataFrame) -> bool:
    o = df["Open"].iloc[-1]
    h = df["High"].iloc[-1]
    c = df["Close"].iloc[-1]
    is_botak = abs(c - h) < 0.05 and c > o
    is_pct = c > o and ((c - o) / max(h - o, 0.001)) > 0.9
    return (is_botak or is_pct) and c > 20


def detect_engulfing_today(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    o0, c0 = df["Open"].iloc[-1], df["Close"].iloc[-1]
    h1, l1 = df["High"].iloc[-2], df["Low"].iloc[-2]
    return bool((o0 < l1) and (c0 > h1) and (c0 > 20))


def build_industry_clusters(hits_set, min_count):
    clusters = {}
    for industry, tickers in INDUSTRIES.items():
        matched = sorted(t for t in tickers if t in hits_set and t in KNOWN_STOCKS)
        if len(matched) >= min_count:
            clusters[industry] = matched
    return clusters


def find_record_breakers(record_dfs):
    """
    For each ticker, compute its daily close-to-close % change history over
    RECORD_LOOKBACK_PERIOD. Split off TODAY's % change from the rest of the
    history, and check whether today's move is a NEW record (higher than
    every prior % up-move, or lower than every prior % down-move).

    Returns (new_up_records, new_down_records), each a dict:
        {ticker: {"today_pct": float, "prior_record_pct": float}}
    """
    new_up_records = {}
    new_down_records = {}

    for t, df in record_dfs.items():
        pct_changes = df["Close"].pct_change().dropna() * 100  # vectorized
        if len(pct_changes) < 2:
            continue  # need at least 1 prior day + today to compare

        today_pct = pct_changes.iloc[-1]
        history = pct_changes.iloc[:-1]

        prior_max_up = history.max()
        prior_max_down = history.min()

        if today_pct > prior_max_up:
            new_up_records[t] = {
                "today_pct": today_pct,
                "prior_record_pct": prior_max_up,
            }

        if today_pct < prior_max_down:
            new_down_records[t] = {
                "today_pct": today_pct,
                "prior_record_pct": prior_max_down,
            }

    return new_up_records, new_down_records


# ------------------------------------------------------------------------
# STATE (avoid re-emailing the exact same result set every 30 min)
# ------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ------------------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------------------
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        timeout=15,
    )
    resp.raise_for_status()


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    if not is_market_open_now():
        print("Market closed right now (ET) — skipping.")
        return

    if not INDUSTRIES or not KNOWN_STOCKS:
        print("ERROR: config.py is still empty. Paste your INDUSTRIES dict "
              "and KNOWN_STOCKS list before running.", file=sys.stderr)
        sys.exit(1)

    all_tickers = sorted(set(KNOWN_STOCKS))

    print(f"Downloading data for {len(all_tickers)} tickers...")
    dfs = download_data(all_tickers, period="5d")
    print(f"Got data for {len(dfs)} tickers.")

    print(f"Downloading {RECORD_LOOKBACK_PERIOD} history for record check...")
    record_dfs = download_data(all_tickers, period=RECORD_LOOKBACK_PERIOD)
    print(f"Got record-lookback data for {len(record_dfs)} tickers.")

    botak_hits = {t for t, df in dfs.items() if detect_botak_today(df)}
    engulf_hits = {t for t, df in dfs.items() if detect_engulfing_today(df)}

    botak_clusters = build_industry_clusters(botak_hits, BOTAK_MIN_PER_INDUSTRY)
    engulf_clusters = build_industry_clusters(engulf_hits, ENGULF_MIN_PER_INDUSTRY)

    new_up_records, new_down_records = find_record_breakers(record_dfs)

    if not botak_clusters and not engulf_clusters and not new_up_records and not new_down_records:
        print("No clusters or record breakers found this run.")
        return

    now_et = datetime.datetime.now(MARKET_TZ)
    today_str = now_et.strftime("%Y-%m-%d")
    sig = json.dumps(
        {
            "botak": botak_clusters,
            "engulf": engulf_clusters,
            "up_records": sorted(new_up_records.keys()),
            "down_records": sorted(new_down_records.keys()),
        },
        sort_keys=True,
    )

    state = load_state()
    if state.get("date") == today_str and state.get("sig") == sig:
        print("Identical results to the last alert sent today — skipping duplicate email.")
        return

    # build a short count header ("is" style) and include the detailed blocks below
    summary = [
        f"{len(botak_clusters)} Botak",
        f"{len(engulf_clusters)} Engulfing",
        f"{len(new_up_records)} New Up Records",
        f"{len(new_down_records)} New Down Records",
        "",
    ]

    details = []
    if botak_clusters:
        details.append(f"BOTAK ({len(botak_clusters)} industries):")
        for ind, tickers in sorted(botak_clusters.items()):
            details.append(f"  {ind} = {', '.join(tickers)}")
        details.append("")

    if engulf_clusters:
        details.append(f"ENGULFING ({len(engulf_clusters)} industries):")
        for ind, tickers in sorted(engulf_clusters.items()):
            details.append(f"  {ind}: {', '.join(tickers)}")
        details.append("")

    if new_up_records:
        details.append(
            f"NEW {RECORD_LOOKBACK_PERIOD} HIGH DAY % UP RECORDS ({len(new_up_records)} tickers):"
        )
        for t, info in sorted(new_up_records.items(), key=lambda x: -x[1]["today_pct"]):
            details.append(
                f"  {t}: today +{info['today_pct']:.2f}% "
                f"(prior record +{info['prior_record_pct']:.2f}%)"
            )
        details.append("")

    if new_down_records:
        details.append(
            f"NEW {RECORD_LOOKBACK_PERIOD} HIGH DAY % DROP RECORDS ({len(new_down_records)} tickers):"
        )
        for t, info in sorted(new_down_records.items(), key=lambda x: x[1]["today_pct"]):
            details.append(
                f"  {t}: today {info['today_pct']:.2f}% "
                f"(prior record {info['prior_record_pct']:.2f}%)"
            )

    subject = (
        f"Cluster Alert: {len(botak_clusters)} Botak / "
        f"{len(engulf_clusters)} Engulfing / "
        f"{len(new_up_records)} Up-Records / "
        f"{len(new_down_records)} Down-Records"
    )

    # compose the Telegram message
    body = "\n".join(summary + details)

    send_telegram(body)
    print("Telegram message sent:", subject)

    save_state({"date": today_str, "sig": sig})


if __name__ == "__main__":
    main()
