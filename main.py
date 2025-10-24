import requests
import json
from dotenv import load_dotenv
import os
import asyncio
import time
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Wait between queries in seconds
QUERY_DELAY = .25

class ApiClient:
    def __init__(self):
        self.base_url = "https://api.etherscan.io/v2/api"
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables")
        
        self.chainid = 11155111

    def get_latest_block_number(self):
        # curl "https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_blockNumber&apikey=YourApiKeyToken"
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None

    def get_block_by_number(self, block_number):
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": block_number,
            "boolean": "true",
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None
    
    def get_transactions_by_block(self, block_number: str):
        block = self.get_block_by_number(block_number)
        if block:
            raw_transactions = block["transactions"]
            transactions = [{
                "timestamp": block["timestamp"],
                "blockNumber": transaction["blockNumber"],
                "hash": transaction["hash"],
                "from": transaction["from"],
                "to": transaction["to"],
                "value": transaction["value"],
                "index": i
            } for i, transaction in enumerate(raw_transactions) if int(transaction["value"], 16) > 0]
            return transactions
        return []

async def query_latest_block(ApiClient : ApiClient):
    block_number = ApiClient.get_latest_block_number()
    return ApiClient.get_block_by_number(block_number)

def ensure_db_schema(conn: sqlite3.Connection):
    """Create required tables if they don't exist.

    - wallets(hash INTEGER PRIMARY KEY, balance INTEGER)
    - transactions(hash INTEGER PRIMARY KEY, timestamp INTEGER, blocknumber INTEGER, from_hash INTEGER, to_hash INTEGER, value INTEGER)
    - blocks(block_number TEXT PRIMARY KEY) -- to track processed blocks
    """
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS wallets (
            hash STRING PRIMARY KEY,
            balance INTEGER DEFAULT 0
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS blocks (
            block_number INTEGER PRIMARY KEY
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS transactions (
            hash STRING PRIMARY KEY,
            timestamp INTEGER,
            blocknumber INTEGER REFERENCES blocks(block_number), 
            from_hash STRING REFERENCES wallets(hash),
            to_hash STRING REFERENCES wallets(hash),
            value STRING,
            block_index INTEGER
        )"""
    )
    conn.commit()

def insert_wallet_if_missing(conn: sqlite3.Connection, addr_str: str):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO wallets (hash, balance) VALUES (?, 0)", (addr_str,))

def insert_transactions_batch(conn: sqlite3.Connection, rows: list):
    """Insert a batch of transactions. Each row is a tuple matching the transactions table.
    (hash, timestamp, blocknumber, from_hash, to_hash, value, block_index)
    """
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO transactions (hash, timestamp, blocknumber, from_hash, to_hash, value, block_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

def block_already_processed(conn: sqlite3.Connection, block_hex: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM blocks WHERE block_number = ?", (block_hex,))
    return cur.fetchone() is not None

def mark_block_processed(conn: sqlite3.Connection, block_hex: int):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO blocks (block_number) VALUES (?)", (block_hex,))
    conn.commit()

async def query_n_blocks(ApiClient : ApiClient, n: int, start_block: int = None):
    block_number_hex = hex(start_block) if start_block else ApiClient.get_latest_block_number()
    if not block_number_hex:
        return None

    db_path = os.path.join("./blockchain.db")
    conn = sqlite3.connect(db_path)
    ensure_db_schema(conn)

    processed_count = 0
    skipped_count = 0
    skipcount = 0
    for i in range(n):
        block_num_hex = hex(int(block_number_hex, 16) - i - skipcount)
        # If this block already processed in DB, skip it
        while block_already_processed(conn, int(block_num_hex, 16)):
            print(f"Block {block_num_hex} already fetched, skipping...")
            skipcount += 1
            block_num_hex = hex(int(block_number_hex, 16) - i - skipcount)
        if i > 0:  
            time.sleep(QUERY_DELAY)

        block_transactions = ApiClient.get_transactions_by_block(block_num_hex)
        # print(f"Fetched {len(block_transactions)} transactions from block {block_num_hex}")
        if i * 100 // n != (i - 1) * 100 // n:
            print(f"Progress: {i/n * 100:.2f}%")

        # Prepare rows and insert into sqlite
        rows = []
        for tx in block_transactions:
            if not tx:
                skipped_count += 1
                continue
            try:
                timestamp_int = int(tx.get("timestamp", "0"), 16) if isinstance(tx.get("timestamp"), str) else int(tx.get("timestamp", 0))
            except Exception:
                timestamp_int = 0
            try:
                blocknumber_int = int(tx.get("blockNumber", "0"), 16) if isinstance(tx.get("blockNumber"), str) else int(tx.get("blockNumber", 0))
            except Exception:
                blocknumber_int = 0
            # try:
            #     value_int = int(tx.get("value", "0"), 16) if isinstance(tx.get("value"), str) else int(tx.get("value", 0))
            # except Exception:
            #     value_int = 0
            tx_hash = tx.get("hash")
            from_val = tx.get("from")
            to_val = tx.get("to")
            block_index = tx.get("index")
            value = tx.get("value")
            
            if from_val is not None:
                insert_wallet_if_missing(conn, from_val)
            if to_val is not None:
                insert_wallet_if_missing(conn, to_val)

            rows.append((tx_hash, timestamp_int, blocknumber_int, from_val, to_val, value, block_index))

        # Insert batch and mark block processed
        insert_transactions_batch(conn, rows)
        mark_block_processed(conn, int(block_num_hex, 16))
        processed_count += 1

    conn.close()
    return { "processed_blocks": processed_count, "skipped_blocks": skipped_count }

def main():
    client = ApiClient()
    data = asyncio.run(query_n_blocks(client, 1000, 9472018))
    open("./data/transactions.json", "w").write(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()
