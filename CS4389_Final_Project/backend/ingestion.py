# ingestion.py
import time
import sqlite3
from typing import Iterable

from api_client import ApiClient
from config import QUERY_DELAY
from db_utils import create_core_tables


import sqlite3

def insert_block_and_transactions(conn: sqlite3.Connection, block_dict) -> None:
    """
    Insert one block and all its transactions into the database.

    Assumes the DB schema:
      blocks(block_number)
      transactions(hash, timestamp, blocknumber, from_hash, to_hash, value, block_index)
    """
    cur = conn.cursor()

    block_number = int(block_dict["number"])
    txs = block_dict.get("transactions", [])

    # 1) Insert the block row – match existing schema: block_number
    cur.execute(
        """
        INSERT OR IGNORE INTO blocks (block_number)
        VALUES (?);
        """,
        (block_number,),
    )

    # 2) Insert all transactions for this block
    tx_rows = []
    for tx in txs:
        tx_hash = tx["hash"]
        timestamp = int(tx["timestamp"])
        from_hash = tx["from"]
        to_hash = tx["to"]

        # value is already an int from ApiClient, but it may be > 2^63-1.
        # Store it as TEXT in SQLite to avoid 64-bit overflow in the driver.
        value_wei = int(tx["value"])
        value = str(value_wei)  # <-- KEY CHANGE: store as string, not int

        # transactionIndex is already small enough to fit in 64-bit
        block_index = int(tx["transactionIndex"])

        tx_rows.append(
            (
                tx_hash,
                timestamp,
                block_number,   # goes into 'blocknumber' column in your DB
                from_hash,
                to_hash,
                value,
                block_index,
            )
        )

    if tx_rows:
        cur.executemany(
            """
            INSERT OR IGNORE INTO transactions
                (hash, timestamp, blocknumber, from_hash, to_hash, value, block_index)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            tx_rows,
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
