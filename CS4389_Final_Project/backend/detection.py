# detection.py
import sqlite3
from typing import List, Dict, Any

from db_utils import create_alert_tables
from config import DB_PATH


def detect_dos_like_activity(
    db_path: str = DB_PATH,
    tx_threshold: int = 10,
    window_minutes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detect possible DoS / spam style behavior:
    accounts that send too many transactions inside a short time window.

    - tx_threshold: minimum number of tx in the window to raise an alert
    - window_minutes: length of time bucket in minutes
    """
    conn = sqlite3.connect(db_path)
    create_alert_tables(conn)
    cur = conn.cursor()

    window_seconds = window_minutes * 60

    # 1) Bucket transactions by (from_hash, time bucket)
    cur.execute(
        """
        WITH bucketed AS (
            SELECT
                hash,
                from_hash,
                timestamp,
                (timestamp / ?) AS bucket
            FROM transactions
            WHERE from_hash IS NOT NULL
              AND from_hash != ''
        )
        SELECT
            from_hash,
            bucket,
            COUNT(*) AS tx_count
        FROM bucketed
        GROUP BY from_hash, bucket
        HAVING tx_count >= ?
        ORDER BY tx_count DESC;
        """,
        (window_seconds, tx_threshold),
    )

    suspicious_rows = cur.fetchall()

    # 2) Figure out next alert_id (based on existing alerts)
    cur.execute("SELECT COALESCE(MAX(alert_id), 0) FROM alerts;")
    row = cur.fetchone()
    next_alert_id = (row[0] or 0) + 1

    alerts: List[Dict[str, Any]] = []

    for from_hash, bucket, tx_count in suspicious_rows:
        alert_id = next_alert_id
        next_alert_id += 1

        window_start_ts = int(bucket * window_seconds)
        window_end_ts = int(window_start_ts + window_seconds)

        # 2a) Insert metadata for this alert so alert_id is clearly tied to this rule
        cur.execute(
            """
            INSERT OR REPLACE INTO alerts
            (alert_id, alert_type, rule_name, severity, window_start, window_end)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                alert_id,
                "dos",                             # alert_type
                "high_volume_transactions_window", # rule_name
                "medium",                          # severity (tune as needed)
                window_start_ts,
                window_end_ts,
            ),
        )

        # 3) Link account to alert
        cur.execute(
            """
            INSERT OR IGNORE INTO account_alerts (alert_id, account_hash)
            VALUES (?, ?);
            """,
            (alert_id, from_hash),
        )

        # 4) Get all tx for that account in this time window
        cur.execute(
            """
            SELECT hash
            FROM transactions
            WHERE from_hash = ?
              AND timestamp >= ?
              AND timestamp < ?
            """,
            (from_hash, window_start_ts, window_end_ts),
        )
        tx_hashes = [r[0] for r in cur.fetchall()]

        # 5) Link tx to alert
        cur.executemany(
            """
            INSERT OR IGNORE INTO transaction_alerts (alert_id, transaction_hash)
            VALUES (?, ?);
            """,
            [(alert_id, h) for h in tx_hashes],
        )

        alerts.append(
            {
                "alert_id": alert_id,
                "alert_type": "dos",
                "rule_name": "high_volume_transactions_window",
                "severity": "medium",
                "account": from_hash,
                "tx_count": tx_count,
                "window_start": window_start_ts,
                "window_end": window_end_ts,
                "transactions": tx_hashes,
            }
        )

    conn.commit()
    conn.close()
    return alerts


def find_transaction_chain(
    db_path: str = DB_PATH,
    min_length: int = 4,
    time_window_seconds: int = 3600,
) -> List[Dict[str, Any]]:
    """
    Simplified same-value transaction chain detector.
    Looks for chains A -> B -> C -> ... where:
      - successive tx have the same value
      - each tx happens within time_window_seconds of the previous
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT hash, from_hash, to_hash, value, timestamp, blocknumber
        FROM transactions
        WHERE from_hash IS NOT NULL
          AND to_hash   IS NOT NULL
        ORDER BY timestamp ASC;
        """
    )
    rows = cur.fetchall()
    conn.close()

    txs = [
        {
            "hash": h,
            "from": f,
            "to": t,
            "value": v,
            "timestamp": ts,
            "blocknumber": b,
        }
        for (h, f, t, v, ts, b) in rows
    ]

    chains: List[Dict[str, Any]] = []

    # Naive O(n^2) search is fine for small testnet datasets
    for i, tx in enumerate(txs):
        chain = [tx]
        current = tx

        for j in range(i + 1, len(txs)):
            nxt = txs[j]
            if (
                nxt["from"] == current["to"]
                and nxt["value"] == current["value"]
                and 0 <= nxt["timestamp"] - current["timestamp"] <= time_window_seconds
            ):
                chain.append(nxt)
                current = nxt

                if len(chain) >= min_length:
                    chains.append(
                        {
                            "value": tx["value"],
                            "length": len(chain),
                            "transactions": chain,
                        }
                    )
                    break  # only one chain per starting tx

    return chains
