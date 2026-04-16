"""
notifier.py — Send email notifications when composite signals change.

Credentials are read from environment variables (via .env):
  SMTP_USER     : Gmail address used to send the email
  SMTP_PASSWORD : Gmail App Password (NOT your account password)
                  Generate at: myaccount.google.com/apppasswords

Config keys used (from config["settings"]):
  smtp_enabled        : bool — master on/off switch (default: false)
  notification_email  : str  — recipient address
"""

import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # SSL


def _build_subject(changes: list[dict]) -> str:
    if len(changes) == 1:
        c = changes[0]
        return f"[PE Monitor] {c['ticker']} 訊號變更: {c['from_signal']} → {c['to_signal']}"
    return f"[PE Monitor] {len(changes)} 檔股票訊號變更"


def _build_html_body(changes: list[dict]) -> str:
    rows = ""
    for c in changes:
        price_str = f"${c['current_price']:.2f}" if c.get("current_price") else "N/A"
        metric_str = f"{c['metric_value']:.2f}" if c.get("metric_value") else "N/A"
        rank_str = f"{c['percentile_rank']:.1f}%" if c.get("percentile_rank") else "N/A"
        rows += (
            f"<tr>"
            f"<td><b>{c['ticker']}</b></td>"
            f"<td>{c.get('name', '')}</td>"
            f"<td>{c['from_signal']}</td>"
            f"<td><b>{c['to_signal']}</b></td>"
            f"<td>{price_str}</td>"
            f"<td>{c.get('metric_label', '')}: {metric_str}</td>"
            f"<td>{rank_str}</td>"
            f"</tr>"
        )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <html><body style="font-family:sans-serif;color:#333">
    <h2 style="color:#1a73e8">PE Monitor — 訊號變更通知</h2>
    <p>掃描時間：{timestamp}</p>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;font-size:14px">
      <tr style="background:#f1f3f4">
        <th>代號</th><th>名稱</th><th>前訊號</th><th>新訊號</th>
        <th>收盤價</th><th>估值</th><th>百分位</th>
      </tr>
      {rows}
    </table>
    <hr style="margin-top:24px">
    <small style="color:#888">由 PE Monitor 自動發送</small>
    </body></html>
    """


def send_signal_change_email(changes: list[dict], config: dict) -> bool:
    """
    Send an HTML email summarising composite signal changes.

    Returns True if the email was sent successfully, False otherwise.
    Never raises — all failures are logged to stdout so a misconfigured
    SMTP setup never crashes the daily scan.
    """
    if not changes:
        return False

    settings = config.get("settings", {})
    if not settings.get("smtp_enabled", False):
        return False

    recipient = settings.get("notification_email", "").strip()
    if not recipient:
        print("[notifier] smtp_enabled=True 但 notification_email 未設定")
        return False

    load_dotenv()
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    if not smtp_user or not smtp_password:
        print("[notifier] .env 中未設定 SMTP_USER 或 SMTP_PASSWORD")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = _build_subject(changes)
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(_build_html_body(changes), "html", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"[notifier] Email 已發送至 {recipient}（{len(changes)} 項變更）")
        return True
    except Exception as e:
        print(f"[notifier] Email 發送失敗：{e}")
        return False
