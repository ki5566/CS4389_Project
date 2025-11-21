"""
backend/api_server.py

FastAPI backend exposing:
- health check
- basic stats
- transactions listing
- fraud alerts for:
    - same-value transaction chains
    - DoS / rapid-like activity
"""

from typing import List, Dict, Any

import sqlite3
from fastapi import FastAPI, Query
from pydantic import BaseModel

from config import DB_PATH
from db_utils import get_connection, create_core_tables
from detection import detect_dos_like_activity, find_transaction_chain

app = FastAPI(
    title="Mitigation of Blockchain Fraud – API",
    version="1.0.0",
    description=(
        "Rule-based fraud detection prototype for Ethereum Sepolia.\n\n"
        "Backed by a local SQLite database populated using the CLI in `cli.py`."
    ),
)


class Transaction(BaseModel):
    hash: str
    timestamp: int
    blocknumber: int
    from_hash: str
    to_hash: str
    value: int


@app.on_event("startup")
def _startup() -> None:
    # Ensure core tables exist so the API doesn't crash on first run
    conn = get_connection(DB_PATH)
    try:
        create_core_tables(conn)
    finally:
        conn.close()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stats/summary")
def stats_summary() -> Dict[str, Any]:
    conn = get_connection(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM blocks;")
        block_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM transactions;")
        tx_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT from_hash) + COUNT(DISTINCT to_hash) FROM transactions;")
        wallet_count = cur.fetchone()[0]

        return {
            "blocks": block_count,
            "transactions": tx_count,
            "wallets": wallet_count,
        }
    finally:
        conn.close()


@app.get("/transactions", response_model=List[Transaction])
def list_transactions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[Transaction]:
    conn = get_connection(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT hash, timestamp, blocknumber, from_hash, to_hash, value
            FROM transactions
            ORDER BY blocknumber DESC, timestamp DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    result: List[Transaction] = []
    for h, ts, bn, fh, th, v in rows:
        # value is stored as TEXT in DB, but we cast to int for the API schema
        try:
            v_int = int(v)
        except Exception:
            try:
                v_int = int(float(v))
            except Exception:
                v_int = 0
        result.append(
            Transaction(
                hash=h,
                timestamp=ts,
                blocknumber=bn,
                from_hash=fh,
                to_hash=th,
                value=v_int,
            )
        )
    return result


@app.get("/alerts/chains")
def alerts_chains(
    min_length: int = Query(4, ge=2, description="Minimum length of same-value chain"),
    time_window_seconds: int = Query(
        3600,
        ge=60,
        description="Max time gap in seconds between consecutive tx in a chain",
    ),
) -> List[Dict[str, Any]]:
    """
    Run the same-value transaction chain detector and return chains as JSON.
    """
    chains = find_transaction_chain(
        db_path=DB_PATH,
        min_length=min_length,
        time_window_seconds=time_window_seconds,
    )
    return chains


@app.get("/alerts/dos")
def alerts_dos(
    tx_threshold: int = Query(10, ge=1, description="Min tx count per (address, window)"),
    window_minutes: int = Query(10, ge=1, description="Window size in minutes"),
) -> List[Dict[str, Any]]:
    """
    Run the DoS / rapid-transaction detector and return alerts as JSON.
    """
    alerts = detect_dos_like_activity(
        db_path=DB_PATH,
        tx_threshold=tx_threshold,
        window_minutes=window_minutes,
    )
    return alerts
