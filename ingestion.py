# ingestion.py
import time
import sqlite3
from typing import Iterable

from api_client import ApiClient
from config import QUERY_DELAY
from db_utils import create_core_tables


def insert_block_and_transactions(conn: sqlite3.Connection, block_dict) -> None:
    """
    Insert one block and all its transactions into the database.
    block_dict is the normalized dict from ApiClient.get_block_by_number.
    """
    cur = conn.cursor()

    block_number = block_dict["number"]
    timestamp = block_dict["timestamp"]

    # Insert block
    cur.execute(
        """
        INSERT OR IGNORE INTO blocks (blocknumber, timestamp)
        VALUES (?, ?);
        """,
        (block_number, timestamp),
    )

    # Insert transactions
    tx_rows = []
    wallet_rows = set()

    for tx in block_dict["transactions"]:
        tx_rows.append(
                    (
                        tx["hash"],
                        tx["timestamp"],
                        tx["blockNumber"],
                        tx["from"],
                        tx["to"],
                        str(tx["value"]),     
                        tx["transactionIndex"],
                    )
                )

        wallet_rows.add(tx["from"])
        if tx["to"]:
            wallet_rows.add(tx["to"])

    if tx_rows:
        cur.executemany(
            """
            INSERT OR IGNORE INTO transactions
            (hash, timestamp, blocknumber, from_hash, to_hash, value, block_index)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            tx_rows,
        )

    if wallet_rows:
        cur.executemany(
            """
            INSERT OR IGNORE INTO wallets (hash, balance)
            VALUES (?, 0);
            """,
            [(addr,) for addr in wallet_rows],
        )

    conn.commit()


def fetch_and_store_blocks(
    conn: sqlite3.Connection,
    client: ApiClient,
    block_numbers: Iterable[int],
    delay: float = QUERY_DELAY,
) -> None:
    for bn in block_numbers:
        block = client.get_block_by_number(bn)
        if block is None:
            print(f"[WARN] Failed to fetch block {bn}")
            continue
        insert_block_and_transactions(conn, block)
        print(f"[INFO] Stored block {bn} with {len(block['transactions'])} tx")
        time.sleep(delay)
