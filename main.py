"""
main.py — Daily P/E monitor runner.

Usage:
    python main.py

Scans all watchlist tickers, prints a summary table to stdout,
and saves the results to reports/daily_YYYYMMDD.csv.
"""

from datetime import datetime

from src.notifier import send_signal_change_email
from src.report_generator import (
    _yesterday_date_str,
    compare_signals,
    load_report,
    save_daily_report,
    scan_all,
)
from src.utils import ensure_dirs, load_config


def print_banner() -> None:
    print("=" * 65)
    print("  PE Monitor — Daily Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


def print_results(results: list[dict]) -> None:
    col_w = [8, 20, 8, 8, 8, 8, 22]
    header = (
        f"{'Ticker':<{col_w[0]}}"
        f"{'Name':<{col_w[1]}}"
        f"{'Price':>{col_w[2]}}"
        f"{'TTM EPS':>{col_w[3]}}"
        f"{'Metric':>{col_w[4]}}"
        f"{'%ile':>{col_w[5]}}"
        f"{'Signal':<{col_w[6]}}"
    )
    print(header)
    print("-" * sum(col_w))

    for r in results:
        if r.get("error"):
            print(f"{r['ticker']:<{col_w[0]}}  ⚠️  {r['error']}")
            continue

        eps_str = f"{r['ttm_eps']:.2f}" if r.get("ttm_eps") is not None else "N/A"
        metric_str = f"{r['metric_value']:.1f}" if r.get("metric_value") is not None else "N/A"
        pct_str = f"{r['percentile_rank']:.0f}%" if r.get("percentile_rank") is not None else "N/A"
        stale_flag = " ⚠️ " if r.get("eps_stale") else ""

        line = (
            f"{r['ticker']:<{col_w[0]}}"
            f"{(r['name'] or '')[:19]:<{col_w[1]}}"
            f"{r['price']:>{col_w[2]}.2f}"
            f"{eps_str:>{col_w[3]}}"
            f"{metric_str:>{col_w[4]}}"
            f"{pct_str:>{col_w[5]}}"
            f"  {r.get('signal_display', 'N/A')}{stale_flag}"
        )
        print(line)

        if r.get("holding_cost") is not None:
            pnl = r.get("holding_pnl_pct")
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "N/A"
            print(
                f"  {'':>{col_w[0]}}持倉 {r['holding_shares']} 股 @ {r['holding_cost']}"
                f"  損益: {pnl_str}"
            )


def main() -> None:
    print_banner()
    config = load_config()
    ensure_dirs(config)

    if not config.get("watchlist"):
        print("watchlist 為空，請先透過 Streamlit 介面新增股票。")
        print("  streamlit run app.py")
        return

    print(f"\n掃描 {len(config['watchlist'])} 檔股票...\n")
    results = scan_all(config)
    print_results(results)

    report_path = save_daily_report(results, config["settings"]["report_dir"])
    print(f"\n報告已儲存：{report_path}")

    # Signal change detection & notification
    report_dir = config["settings"]["report_dir"]
    yesterday_df = load_report(_yesterday_date_str(), report_dir)
    if not yesterday_df.empty:
        changes = compare_signals(results, yesterday_df)
        if changes:
            print(f"\n訊號變更：{len(changes)} 檔")
            for c in changes:
                print(f"  {c['ticker']:8s} {c['from_signal']} → {c['to_signal']}")
            sent = send_signal_change_email(changes, config)
            if not sent and config["settings"].get("smtp_enabled"):
                print("  ⚠️  郵件發送失敗，請確認 .env 中的 SMTP_USER / SMTP_PASSWORD")
        else:
            print("\n訊號無變化")
    else:
        print(f"\n（無昨日報告可比較）")

    # Summary counts
    signals = [r.get("signal", "N/A") for r in results if not r.get("error")]
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    if buy_count:
        print(f"\n🟢 BUY ZONE：{buy_count} 檔")
    if sell_count:
        print(f"🔴 SELL ZONE：{sell_count} 檔")


if __name__ == "__main__":
    main()
