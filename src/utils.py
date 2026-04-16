"""
Utility functions: config loading/saving, .env bootstrapping, helpers.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

CONFIG_PATH = Path("config.json")


def _parse_watchlist(raw: str) -> list[dict]:
    """Parse WATCHLIST env value into minimal watchlist entries.

    Supports both plain 'AAPL' and typed 'AAPL:growth' formats.
    Valid types: stable / growth / cyclical / etf
    Any unrecognised or missing type defaults to 'unknown' (auto-analysed).

    Examples
    --------
    WATCHLIST=AAPL:growth,GOOGL,MSFT:stable,XOM:cyclical
    """
    VALID_TYPES = {"stable", "growth", "cyclical", "etf"}
    entries = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":", 1)
        ticker = parts[0].strip().upper()
        if not ticker:
            continue

        if len(parts) == 2 and parts[1].strip().lower() in VALID_TYPES:
            predefined_type = parts[1].strip().lower()
            type_source = "env"
            recommended_metric = "PB" if predefined_type == "cyclical" else "PE"
            reason = f"類型由 .env 預設為 {predefined_type}"
        else:
            predefined_type = "unknown"
            type_source = "auto"
            recommended_metric = "PE"
            reason = "尚未分析"

        entries.append(
            {
                "ticker": ticker,
                "name": "",
                "type": predefined_type,
                "type_source": type_source,
                "recommended_metric": recommended_metric,
                "suitability_score": 0,
                "reason": reason,
                "added_date": date.today().isoformat(),
            }
        )
    return entries


def _parse_holdings(raw: str) -> list[dict]:
    """Parse HOLDINGS env value into holding entries."""
    entries = []
    if not raw or not raw.strip():
        return entries
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            continue
        ticker, cost, shares = parts
        try:
            entries.append(
                {
                    "ticker": ticker.strip().upper(),
                    "cost": float(cost.strip()),
                    "shares": float(shares.strip()),
                    "buy_date": date.today().isoformat(),
                }
            )
        except ValueError:
            continue
    return entries


def init_config_from_env() -> dict:
    """Bootstrap config.json from .env (first-run only)."""
    load_dotenv(override=False)

    watchlist_raw = os.getenv("WATCHLIST", "GOOGL,AAPL,MSFT")
    holdings_raw = os.getenv("HOLDINGS", "")
    pe_history_years = int(os.getenv("PE_HISTORY_YEARS", "5"))
    entry_percentile = int(os.getenv("ENTRY_PERCENTILE", "25"))
    exit_percentile = int(os.getenv("EXIT_PERCENTILE", "75"))
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")

    config = {
        "watchlist": _parse_watchlist(watchlist_raw),
        "holdings": _parse_holdings(holdings_raw),
        "settings": {
            "pe_history_years": pe_history_years,
            "entry_percentile": entry_percentile,
            "exit_percentile": exit_percentile,
            "data_dir": "data",
            "report_dir": "reports",
            "finnhub_api_key": finnhub_api_key,
            "news_days_lookback": 14,
            "news_weight": 0.3,
        },
    }
    return config


_SETTINGS_DEFAULTS = {
    "strategy_d": {
        "enabled": False,
        "kd_window": 10,
        "n_bars": 3,
        "recovery_pct": 0.7,
        "kd_k_threshold": 20,
    },
    "smtp_enabled": False,
    "notification_email": "",
}


def load_config() -> dict:
    """Load config.json; create from .env if it doesn't exist.

    Merges any missing settings keys with defaults so new features are
    available even when the user's config.json pre-dates them.
    """
    if not CONFIG_PATH.exists():
        config = init_config_from_env()
        save_config(config)
        return config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Merge in any missing default settings (non-destructive)
    settings = config.setdefault("settings", {})
    for key, default in _SETTINGS_DEFAULTS.items():
        if key not in settings:
            settings[key] = default
    # Backfill type_source for watchlist entries that predate this field
    for entry in config.get("watchlist", []):
        if "type_source" not in entry:
            entry["type_source"] = "auto"
    return config


def save_config(config: dict) -> None:
    """Persist config.json to disk."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def ensure_dirs(config: dict) -> None:
    """Create data/ and reports/ directories if they don't exist."""
    Path(config["settings"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["settings"]["report_dir"]).mkdir(parents=True, exist_ok=True)


def get_watchlist_tickers(config: dict) -> list[str]:
    return [entry["ticker"] for entry in config.get("watchlist", [])]


def get_holding(config: dict, ticker: str) -> dict | None:
    for h in config.get("holdings", []):
        if h["ticker"] == ticker:
            return h
    return None


def add_to_watchlist(config: dict, entry: dict) -> None:
    tickers = get_watchlist_tickers(config)
    if entry["ticker"] not in tickers:
        config["watchlist"].append(entry)
        save_config(config)


def remove_from_watchlist(config: dict, ticker: str) -> None:
    config["watchlist"] = [
        e for e in config["watchlist"] if e["ticker"] != ticker
    ]
    # also remove from holdings
    config["holdings"] = [
        h for h in config["holdings"] if h["ticker"] != ticker
    ]
    save_config(config)


def upsert_holding(config: dict, ticker: str, cost: float, shares: float, buy_date: str | None = None) -> None:
    for h in config["holdings"]:
        if h["ticker"] == ticker:
            h["cost"] = cost
            h["shares"] = shares
            if buy_date:
                h["buy_date"] = buy_date
            save_config(config)
            return
    config["holdings"].append(
        {
            "ticker": ticker,
            "cost": cost,
            "shares": shares,
            "buy_date": buy_date or date.today().isoformat(),
        }
    )
    save_config(config)


def remove_holding(config: dict, ticker: str) -> None:
    config["holdings"] = [h for h in config["holdings"] if h["ticker"] != ticker]
    save_config(config)


def days_since(date_str: str) -> int:
    """Return number of days between a date string (YYYY-MM-DD) and today."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999
