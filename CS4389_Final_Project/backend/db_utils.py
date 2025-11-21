# db_utils.py
import sqlite3
from typing import Optional

from config import DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DB_PATH)


def create_core_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # blocks table – matches existing DB: blocks(block_number)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS blocks (
            block_number INTEGER PRIMARY KEY
        );
        """
    )

    # wallets table – matches existing DB: wallets(hash, balance)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wallets (
            hash    STRING PRIMARY KEY,
            balance INTEGER DEFAULT 0
        );
        """
    )

    # transactions table – matches existing DB
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            hash        STRING PRIMARY KEY,
            timestamp   INTEGER,
            blocknumber INTEGER,
            from_hash   STRING,
            to_hash     STRING,
            value       INTEGER,
            block_index INTEGER
        );
        """
    )

    # transaction-level alerts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transaction_alerts (
            alert_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_hash TEXT NOT NULL
        );
        """
    )

    # wallet-level alerts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wallet_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet   TEXT NOT NULL
        );
        """
    )

    # mapping alert → in/out tx pair
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wallet_alert_transaction_pairs (
            alert_id       INTEGER NOT NULL,
            in_transaction TEXT NOT NULL,
            out_transaction TEXT NOT NULL
        );
        """
    )

    # DoS / rapid-tx alerts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dos_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet   TEXT NOT NULL
        );
        """
    )

    conn.commit()




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
    # match existing DB schema: blocks.block_number
    cur.execute("SELECT MAX(block_number) FROM blocks;")
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])

