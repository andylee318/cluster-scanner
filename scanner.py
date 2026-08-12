"""
Cluster Scanner — Botak & Engulfing
Scans KNOWN_STOCKS for today's bar matching the "botak" or "bullish engulfing"
pattern, groups hits by industry, and emails an alert if any industry has
enough same-day hits to count as a cluster (mirrors the thresholds used in
the Streamlit dashboard: engulfing > 1 ticker per industry, botak > 2).

Intended to be run on a schedule (every 30 min during market hours) by
GitHub Actions. Safe to also run manually / locally for testing.
"""

import os
import sys
import json
import smtplib
import datetime
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import INDUSTRIES, KNOWN_STOCKS

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
EMAIL_FROM = os.environ["EMAIL_FROM"]
EMAIL_TO = [e.strip() for e in os.environ["EMAIL_TO"].split(",") if e.strip()]
EMAIL_APP_PASSWORD = os.environ["EMAIL_APP_PASSWORD"]

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

STATE_FILE = "cluster_state.json"

# Thresholds — mirrors the original Streamlit logic:
#   engulf_industry_count > 1   -> at least 2 tickers
#   botak_industry_count  > 2   -> at least 3 tickers
ENGULF_MIN_PER_INDUSTRY = 2
BOTAK_MIN_PER_INDUSTRY = 3

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
def download_data(tickers):
    raw = yf.download(
        tickers, period="5d", interval="1d", progress=False, auto_adjust=True
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


# ------------------------------------------------------------------------
# STATE (avoid re-emailing the exact same cluster set every 30 min)
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
# EMAIL
# ------------------------------------------------------------------------
def send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    try:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    finally:
        server.quit()


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
    dfs = download_data(all_tickers)
    print(f"Got data for {len(dfs)} tickers.")

    botak_hits = {t for t, df in dfs.items() if detect_botak_today(df)}
    engulf_hits = {t for t, df in dfs.items() if detect_engulfing_today(df)}

    botak_clusters = build_industry_clusters(botak_hits, BOTAK_MIN_PER_INDUSTRY)
    engulf_clusters = build_industry_clusters(engulf_hits, ENGULF_MIN_PER_INDUSTRY)

    if not botak_clusters and not engulf_clusters:
        print("No clusters found this run.")
        return

    now_et = datetime.datetime.now(MARKET_TZ)
    today_str = now_et.strftime("%Y-%m-%d")
    sig = json.dumps(
        {"botak": botak_clusters, "engulf": engulf_clusters}, sort_keys=True
    )

    state = load_state()
    if state.get("date") == today_str and state.get("sig") == sig:
        print("Identical clusters to the last alert sent today — skipping duplicate email.")
        return

    lines = [f"Cluster Scan — {now_et.strftime('%Y-%m-%d %H:%M %Z')}", ""]

    if botak_clusters:
        lines.append(f"BOTAK CLUSTER ({len(botak_clusters)} industries):")
        for ind, tickers in sorted(botak_clusters.items()):
            lines.append(f"  {ind}: {', '.join(tickers)}")
        lines.append("")

    if engulf_clusters:
        lines.append(f"ENGULFING CLUSTER ({len(engulf_clusters)} industries):")
        for ind, tickers in sorted(engulf_clusters.items()):
            lines.append(f"  {ind}: {', '.join(tickers)}")

    body = "\n".join(lines)
    subject = (
        f"Cluster Alert: {len(botak_clusters)} Botak / "
        f"{len(engulf_clusters)} Engulfing industries"
    )

    send_email(subject, body)
    print("Email sent:", subject)

    save_state({"date": today_str, "sig": sig})


if __name__ == "__main__":
    main()
