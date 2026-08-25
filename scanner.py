"""
Cluster Scanner — Botak, Engulfing, Wick, Volume, Volatility + Record Breakers
Scans KNOWN_STOCKS for today's bar matching several intraday patterns,
groups hits by industry (where applicable), and sends a Telegram alert if
any industry has enough same-day hits to count as a cluster (mirrors the
thresholds used in the Streamlit dashboard):
    - Engulfing:        > 1 ticker per industry  (>= 2)
    - Botak:            > 2 tickers per industry (>= 3)
    - Long Upper Wick:  > 2 tickers per industry  (>= 3)
    - Long Bottom Wick: > 2 tickers per industry  (>= 3)
    - Volume Cluster:   >= 3 tickers per industry (volume above 50D avg + up day)

Also scans every ticker's own daily close-to-close % change history over
RECORD_LOOKBACK_PERIOD and flags any ticker where TODAY's % move is a new
all-time high (within that window) single-day % up-move or % down-move.

Also flags individual tickers (not clustered by industry) whose daily-range
Z-score (vs its own 20-day mean/stdev) is >= 2 today — a Volatility pickup
signal, same formula as the Streamlit dashboard's Volatility screen.

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
import numpy as np
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
#   upper/lower wick count > 2  -> at least 3 tickers
#   volume cluster: qualifying_tickers >= 3
ENGULF_MIN_PER_INDUSTRY = 2
BOTAK_MIN_PER_INDUSTRY = 3
UPPER_WICK_MIN_PER_INDUSTRY = 3
LOWER_WICK_MIN_PER_INDUSTRY = 3
VOLUME_MIN_PER_INDUSTRY = 3

# Volatility Z-score threshold (daily range vs its own 20-day mean/stdev)
VOLATILITY_Z_THRESHOLD = 2.0

# How far back to look when establishing each ticker's own record for the
# single-day highest % up-move and highest % down-move. Adjust as needed
# (e.g. "6mo", "2y", "5y") — longer = harder record to break, more meaningful.
RECORD_LOOKBACK_PERIOD = "1y"

# How far back to look for Volume (50D avg) and Volatility (20D range Z-score)
# calculations — needs to comfortably cover a 50-day rolling window.
EXTENDED_LOOKBACK_PERIOD = "4mo"

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
    """Download OHLCV data for `tickers` over `period` (daily interval).
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
                    "Volume": raw["Volume"][t],
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


def detect_long_upper_wick_today(df: pd.DataFrame) -> bool:
    o = df["Open"].iloc[-1]
    h = df["High"].iloc[-1]
    l = df["Low"].iloc[-1]
    c = df["Close"].iloc[-1]
    rng = h - l
    if rng <= 0 or c <= 20:
        return False
    upper_wick = h - max(o, c)
    return (upper_wick / rng) > 0.5


def detect_long_lower_wick_today(df: pd.DataFrame) -> bool:
    o = df["Open"].iloc[-1]
    h = df["High"].iloc[-1]
    l = df["Low"].iloc[-1]
    c = df["Close"].iloc[-1]
    rng = h - l
    if rng <= 0 or c <= 20:
        return False
    lower_wick = min(o, c) - l
    return (lower_wick / rng) > 0.5


def detect_volume_cluster_today(df: pd.DataFrame) -> bool:
    """Today's volume above its own 50-day average AND price closed up
    vs prior day — same condition used by the Streamlit Volume Cluster."""
    if len(df) < 51:
        return False
    vol = df["Volume"]
    close = df["Close"]

    avg_vol50 = vol.rolling(50).mean().iloc[-1]
    if pd.isna(avg_vol50) or avg_vol50 <= 0:
        return False

    c_today = close.iloc[-1]
    c_prev = close.iloc[-2]
    if pd.isna(c_today) or pd.isna(c_prev) or c_prev == 0:
        return False

    is_vol_above = vol.iloc[-1] > avg_vol50
    is_price_up = c_today > c_prev
    return bool(is_vol_above and is_price_up)


def compute_volatility_hit(df: pd.DataFrame):
    """Daily-range Z-score (vs its own 20-day mean/stdev) — same formula as
    the Streamlit Volatility screen. Returns (z_score, pct_chg) if today's
    Z-score >= VOLATILITY_Z_THRESHOLD and price > 20, else None."""
    if len(df) < 22:
        return None

    high, low, close = df["High"], df["Low"], df["Close"]
    if close.iloc[-1] < 20:
        return None

    daily_range = 100 * (high / low - 1)
    avg_range = daily_range.rolling(20).mean()
    std_range = daily_range.rolling(20).std(ddof=1)

    z_series = (daily_range - avg_range) / std_range.replace(0, np.nan)
    z_today = z_series.iloc[-1]

    if pd.isna(z_today) or z_today < VOLATILITY_Z_THRESHOLD:
        return None

    if len(close) >= 2 and close.iloc[-2] != 0 and pd.notna(close.iloc[-2]):
        pct_chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    else:
        pct_chg = 0.0

    return round(float(z_today), 2), round(float(pct_chg), 2)


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


def find_volatility_hits(ext_dfs):
    """Returns dict {ticker: {"z": float, "pct": float}} for every ticker
    whose today's daily-range Z-score >= VOLATILITY_Z_THRESHOLD."""
    hits = {}
    for t, df in ext_dfs.items():
        result = compute_volatility_hit(df)
        if result is not None:
            z, pct = result
            hits[t] = {"z": z, "pct": pct}
    return hits


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

    print(f"Downloading {EXTENDED_LOOKBACK_PERIOD} history for volume/volatility checks...")
    ext_dfs = download_data(all_tickers, period=EXTENDED_LOOKBACK_PERIOD)
    print(f"Got extended-lookback data for {len(ext_dfs)} tickers.")

    print(f"Downloading {RECORD_LOOKBACK_PERIOD} history for record check...")
    record_dfs = download_data(all_tickers, period=RECORD_LOOKBACK_PERIOD)
    print(f"Got record-lookback data for {len(record_dfs)} tickers.")

    botak_hits = {t for t, df in dfs.items() if detect_botak_today(df)}
    engulf_hits = {t for t, df in dfs.items() if detect_engulfing_today(df)}
    upper_wick_hits = {t for t, df in dfs.items() if detect_long_upper_wick_today(df)}
    lower_wick_hits = {t for t, df in dfs.items() if detect_long_lower_wick_today(df)}
    volume_hits = {t for t, df in ext_dfs.items() if detect_volume_cluster_today(df)}

    botak_clusters = build_industry_clusters(botak_hits, BOTAK_MIN_PER_INDUSTRY)
    engulf_clusters = build_industry_clusters(engulf_hits, ENGULF_MIN_PER_INDUSTRY)
    upper_wick_clusters = build_industry_clusters(upper_wick_hits, UPPER_WICK_MIN_PER_INDUSTRY)
    lower_wick_clusters = build_industry_clusters(lower_wick_hits, LOWER_WICK_MIN_PER_INDUSTRY)
    volume_clusters = build_industry_clusters(volume_hits, VOLUME_MIN_PER_INDUSTRY)

    new_up_records, new_down_records = find_record_breakers(record_dfs)
    volatility_hits = find_volatility_hits(ext_dfs)

    if (
        not botak_clusters
        and not engulf_clusters
        and not upper_wick_clusters
        and not lower_wick_clusters
        and not volume_clusters
        and not new_up_records
        and not new_down_records
        and not volatility_hits
    ):
        print("No clusters or record breakers found this run.")
        return

    now_et = datetime.datetime.now(MARKET_TZ)
    today_str = now_et.strftime("%Y-%m-%d")
    sig = json.dumps(
        {
            "botak": botak_clusters,
            "engulf": engulf_clusters,
            "upper_wick": upper_wick_clusters,
            "lower_wick": lower_wick_clusters,
            "volume": volume_clusters,
            "up_records": sorted(new_up_records.keys()),
            "down_records": sorted(new_down_records.keys()),
            "volatility": sorted(volatility_hits.keys()),
        },
        sort_keys=True,
    )

    state = load_state()
    if state.get("date") == today_str and state.get("sig") == sig:
        print("Identical results to the last alert sent today — skipping duplicate email.")
        return

    # build a short count header ("is" style) and include the detailed blocks below
    summary = [
        f"🔄 {len(engulf_clusters)} Engulfing",
        f"🧑‍🦲 {len(botak_clusters)} Botak",
        f"✅ {len(lower_wick_clusters)} Long Bottom Wick",
        f"❌ {len(upper_wick_clusters)} Long Upper Wick",
        f"📊 {len(volume_clusters)} Volume",
        f"🚀 {len(new_up_records)} New Up Records",
        f"📉 {len(new_down_records)} New Down Records",
        f"⚡ {len(volatility_hits)} Volatility Z-Score",
        "",
    ]

    details = []
    if engulf_clusters:
        details.append(f"🔄 ENGULFING ({len(engulf_clusters)} industries):")
        for ind, tickers in sorted(engulf_clusters.items()):
            details.append(f"  {ind}: {', '.join(tickers)}")
        details.append("")

    if botak_clusters:
        details.append(f"🧑‍🦲 BOTAK ({len(botak_clusters)} industries):")
        for ind, tickers in sorted(botak_clusters.items()):
            details.append(f"  {ind} = {', '.join(tickers)}")
        details.append("")

    if lower_wick_clusters:
        details.append(f"✅ LONG BOTTOM WICK ({len(lower_wick_clusters)} industries):")
        for ind, tickers in sorted(lower_wick_clusters.items()):
            details.append(f"  {ind}: {', '.join(tickers)}")
        details.append("")

    if upper_wick_clusters:
        details.append(f"❌ LONG UPPER WICK ({len(upper_wick_clusters)} industries):")
        for ind, tickers in sorted(upper_wick_clusters.items()):
            details.append(f"  {ind}: {', '.join(tickers)}")
        details.append("")

    if volume_clusters:
        details.append(f"📊 VOLUME ({len(volume_clusters)} industries):")
        for ind, tickers in sorted(volume_clusters.items()):
            details.append(f"  {ind}: {', '.join(tickers)}")
        details.append("")

    if new_up_records:
        details.append(
            f"🚀 NEW {RECORD_LOOKBACK_PERIOD} HIGH DAY % UP RECORDS ({len(new_up_records)} tickers):"
        )
        for t, info in sorted(new_up_records.items(), key=lambda x: -x[1]["today_pct"]):
            details.append(
                f"  {t}: today +{info['today_pct']:.2f}% "
                f"(prior record +{info['prior_record_pct']:.2f}%)"
            )
        details.append("")

    if new_down_records:
        details.append(
            f"📉 NEW {RECORD_LOOKBACK_PERIOD} HIGH DAY % DROP RECORDS ({len(new_down_records)} tickers):"
        )
        for t, info in sorted(new_down_records.items(), key=lambda x: x[1]["today_pct"]):
            details.append(
                f"  {t}: today {info['today_pct']:.2f}% "
                f"(prior record {info['prior_record_pct']:.2f}%)"
            )
        details.append("")

    if volatility_hits:
        details.append(
            f"⚡ VOLATILITY Z-SCORE >= {VOLATILITY_Z_THRESHOLD} ({len(volatility_hits)} tickers):"
        )
        for t, info in sorted(volatility_hits.items(), key=lambda x: -x[1]["z"]):
            sign = "+" if info["pct"] >= 0 else ""
            details.append(
                f"  {t}: z={info['z']:.2f} ({sign}{info['pct']:.2f}%)"
            )

    subject = (
        f"Cluster Alert: {len(engulf_clusters)} Engulfing / "
        f"{len(botak_clusters)} Botak / "
        f"{len(lower_wick_clusters)} LowerWick / "
        f"{len(upper_wick_clusters)} UpperWick / "
        f"{len(volume_clusters)} Volume / "
        f"{len(new_up_records)} Up-Records / "
        f"{len(new_down_records)} Down-Records / "
        f"{len(volatility_hits)} Volatility"
    )

    # compose the Telegram message
    body = "\n".join(summary + details)

    send_telegram(body)
    print("Telegram message sent:", subject)

    save_state({"date": today_str, "sig": sig})


if __name__ == "__main__":
    main()
