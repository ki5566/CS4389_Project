# db_utils.py
import sqlite3
from typing import Optional

from config import DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DB_PATH)


def create_core_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # blocks table used by query.get_block_count()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS blocks (
            blocknumber INTEGER PRIMARY KEY,
            timestamp   INTEGER
        );
        """
    )

    # transactions table used by query.get_transactions()
    cur.execute(
    """
    CREATE TABLE IF NOT EXISTS transactions (
        hash         TEXT PRIMARY KEY,
        timestamp    INTEGER,
        blocknumber  INTEGER,
        from_hash    TEXT,
        to_hash      TEXT,
        value        TEXT,
        block_index  INTEGER
    );
    """
)



    # simple wallets table (you can extend later)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wallets (
            hash    TEXT PRIMARY KEY,
            balance INTEGER DEFAULT 0
        );
        """
    )

    conn.commit()


def create_alert_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Master alerts table: one row per alert_id
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id     INTEGER PRIMARY KEY,
            alert_type   TEXT NOT NULL,   -- e.g. 'dos', 'same_value_chain'
            rule_name    TEXT NOT NULL,   -- e.g. 'high_volume_transactions_per_window'
            severity     TEXT,            -- e.g. 'low', 'medium', 'high'
            window_start INTEGER,         -- optional: time window context
            window_end   INTEGER          -- optional: time window context
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transaction_alerts (
            alert_id         INTEGER,
            transaction_hash TEXT NOT NULL,
            PRIMARY KEY (alert_id, transaction_hash)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS account_alerts (
            alert_id     INTEGER,
            account_hash TEXT NOT NULL,
            PRIMARY KEY (alert_id, account_hash),
            FOREIGN KEY (alert_id) REFERENCES transaction_alerts(alert_id)
        );
        """
    )

    conn.commit()



def get_max_block_in_db(conn: sqlite3.Connection) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT MAX(blocknumber) FROM blocks;")
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])
