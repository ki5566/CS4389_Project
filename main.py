import requests
import random
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

    def get_native_balance(self, address: str, block_number: str = "latest"):
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": block_number,
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None

    def get_native_balances_batch(self, addresses: list[str], block_number: str = "latest"):
        address_str = ",".join(addresses)
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "account",
            "action": "balancemulti",
            "address": address_str,
            "tag": block_number,
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None


async def query_latest_block(ApiClient: ApiClient):
    def get_native_balance(self, address: str, block_number: str = "latest"):
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": block_number,
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None

    def get_native_balances_batch(self, addresses: list[str], block_number: str = "latest"):
        address_str = ",".join(addresses)
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "account",
            "action": "balancemulti",
            "address": address_str,
            "tag": block_number,
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None


async def query_latest_block(ApiClient: ApiClient):
    block_number = ApiClient.get_latest_block_number()
    return ApiClient.get_block_by_number(block_number)


def ensure_db_schema(conn: sqlite3.Connection):
    """Create required tables if they don't exist.

    - wallets(hash INTEGER PRIMARY KEY, balance INTEGER)
    - transactions(hash INTEGER PRIMARY KEY, timestamp INTEGER, blocknumber INTEGER,
                   from_hash INTEGER, to_hash INTEGER, value INTEGER)
    - transactions(hash INTEGER PRIMARY KEY, timestamp INTEGER, blocknumber INTEGER,
                   from_hash INTEGER, to_hash INTEGER, value INTEGER)
    - blocks(block_number TEXT PRIMARY KEY) -- to track processed blocks
    """
    cur = conn.cursor()

    # Enable foreign keys
    # cur.execute( "PRAGMA foreign_keys = ON" )
    cur.execute("PRAGMA foreign_keys = OFF")
    
    cur.execute(
        """CREATE TABLE IF NOT EXISTS blocks (
            block_number INTEGER PRIMARY KEY
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS wallets (
            hash TEXT PRIMARY KEY,
            balance INTEGER DEFAULT 0,
            last_updated_block_number INTEGER REFERENCES blocks(block_number) DEFAULT 0
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS transactions (
            hash TEXT PRIMARY KEY,
            timestamp INTEGER,
            blocknumber INTEGER REFERENCES blocks(block_number),
            from_hash TEXT REFERENCES wallets(hash),
            to_hash TEXT REFERENCES wallets(hash),
            value TEXT,
            block_index INTEGER
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS chain_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_length INTEGER
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS chain_alert_transactions (
            alert_id INTEGER REFERENCES chain_alerts(alert_id) ON DELETE CASCADE,
            transaction_hash TEXT REFERENCES transactions(hash) ON DELETE CASCADE,
            PRIMARY KEY (alert_id, transaction_hash)
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS chain_alert_wallets (
            alert_id INTEGER REFERENCES chain_alerts(alert_id) ON DELETE CASCADE,
            wallet TEXT REFERENCES wallets(hash) ON DELETE CASCADE,
            PRIMARY KEY (alert_id, wallet)
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS wallet_alerts (
            alert_id INTEGER ,
            wallet TEXT REFERENCES wallets(hash) ON DELETE CASCADE,
            PRIMARY KEY (alert_id, wallet)
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS wallet_alert_transaction_pairs(
            alert_id INTEGER REFERENCES wallet_alerts(alert_id) ON DELETE CASCADE,
            in_transaction TEXT REFERENCES transactions(hash) ON DELETE CASCADE,
            out_transaction TEXT REFERENCES transactions(hash) ON DELETE CASCADE,
            PRIMARY KEY (alert_id, in_transaction, out_transaction)
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS dos_alerts(
            alert_id INTEGER,
            wallet TEXT REFERENCES wallets(hash) ON DELETE CASCADE,
            PRIMARY KEY (alert_id, wallet)
        )"""
    )
    conn.commit()


def insert_wallet_if_missing(conn: sqlite3.Connection, addr_str: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO wallets (hash, balance) VALUES (?, 0)", (addr_str,))

    cur.execute(
        "INSERT OR IGNORE INTO wallets (hash, balance) VALUES (?, 0)", (addr_str,))


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


def insert_alerts_batch(conn: sqlite3.Connection, rows: list):
    """Insert a batch of alerts. Each row is a tuple matching alerts table.
    (alert_id, account_hash)
    """
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO (alert_id, account_hash) VALUES (?, ?)",
        rows,
    )
    conn.commit()


def block_already_processed(conn: sqlite3.Connection, block_hex: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM blocks WHERE block_number = ?", (block_hex,))
    return cur.fetchone() is not None


def mark_block_processed(conn: sqlite3.Connection, block_hex: int):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO blocks (block_number) VALUES (?)", (block_hex,))
    cur.execute(
        "INSERT OR IGNORE INTO blocks (block_number) VALUES (?)", (block_hex,))
    conn.commit()


async def query_n_blocks(ApiClient: ApiClient, n: int, start_block: int, reverse=False):
    if reverse:
        mult = -1
    else:
        mult = 1
    block_number_hex = hex(
        start_block) if start_block else ApiClient.get_latest_block_number()
    if not block_number_hex:
        return None

    db_path = os.path.join("./blockchain.db")
    conn = sqlite3.connect(db_path)
    ensure_db_schema(conn)

    processed_count = 0
    skipped_count = 0
    skipcount = 0
    for i in range(n):
        block_num_hex = hex(int(block_number_hex, 16) -
                            (mult*i) - (mult*skipcount))
        # If this block already processed in DB, skip it
        while block_already_processed(conn, int(block_num_hex, 16)):
            print(f"Block {block_num_hex} already fetched, skipping...")
            skipcount += 1
            block_num_hex = hex(int(block_number_hex, 16) -
                                (mult*i) - (mult*skipcount))
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
                timestamp_int = int(tx.get("timestamp", "0"), 16) if isinstance(
                    tx.get("timestamp"), str) else int(tx.get("timestamp", 0))
                timestamp_int = int(tx.get("timestamp", "0"), 16) if isinstance(
                    tx.get("timestamp"), str) else int(tx.get("timestamp", 0))
            except Exception:
                timestamp_int = 0
            try:
                blocknumber_int = int(tx.get("blockNumber", "0"), 16) if isinstance(
                    tx.get("blockNumber"), str) else int(tx.get("blockNumber", 0))
                blocknumber_int = int(tx.get("blockNumber", "0"), 16) if isinstance(
                    tx.get("blockNumber"), str) else int(tx.get("blockNumber", 0))
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

            rows.append((tx_hash, timestamp_int, blocknumber_int,
                        from_val, to_val, value, block_index))
            rows.append((tx_hash, timestamp_int, blocknumber_int,
                        from_val, to_val, value, block_index))

        # Insert batch and mark block processed
        insert_transactions_batch(conn, rows)
        mark_block_processed(conn, int(block_num_hex, 16))
        processed_count += 1

    conn.close()
    return {"processed_blocks": processed_count, "skipped_blocks": skipped_count}

def fetch_transacting_wallets(db_conn):
    in_wallets = set(db_conn.execute(
        "SELECT DISTINCT from_hash FROM transactions"
    ).fetchall())
    out_wallets = set(db_conn.execute(
        "SELECT DISTINCT to_hash FROM transactions"
    ).fetchall())

    return [wallet[0] for wallet in list(in_wallets.intersection(out_wallets))]

def fetch_sus_chains_and_wallets(db_path: str):
    conn = sqlite3.connect(db_path)
    sus_wallets = set()
    chains = []
    interactions = 0
    print("Fetching suspicious wallets and potential chains:")
    search_wallets = fetch_transacting_wallets(conn)
    progress = 0
    total = len(search_wallets)
    print(f"Total wallets to search: {total}")
    for start_wallet in search_wallets:
        print(f"Progress: {progress/total*100:.2f}%")
        in_transactions = conn.execute(
            "SELECT from_hash, value, timestamp, hash FROM transactions WHERE to_hash = ?", (start_wallet,)).fetchall()
        out_transactions = conn.execute(
            "SELECT to_hash, value, timestamp, hash FROM transactions WHERE from_hash = ?", (start_wallet,)).fetchall()
        if (len(in_transactions) > 0 and len(out_transactions) > 0):
            # print(f"Wallet {start_wallet} has incoming:\n {in_transactions} \n outgoing: \n {out_transactions}")
            for tx in in_transactions:
                from_hash, in_value, in_timestamp, in_tx_hash = tx
                for out_tx in out_transactions:
                    to_hash, out_value, out_timestamp, out_tx_hash = out_tx
                    if (int(in_timestamp) < int(out_timestamp)):
                        if abs(int(in_value, 16) - int(out_value, 16)) < max(int(in_value, 16) // (10 ** 5), 100):
                            interactions += 1
                            # print(f"Found interaction chain for wallet {start_wallet}:\n value: {int(in_value, 16)}\n tolerance: {max(int(in_value, 16) // (10 ** 5), 100)/int(in_value, 16)}\n timediff: {out_timestamp - in_timestamp}\n")
                            chains.append({
                                "value": int(in_value, 16),
                                "chain": [from_hash, start_wallet, to_hash],
                                "transactions": [in_tx_hash, out_tx_hash],
                                "timestamps": [in_timestamp, out_timestamp]
                            })
                            sus_wallets.add(start_wallet)
        progress += 1
    print(f"Found {len(chains)} chains to chase.")
    return chains, list(sus_wallets)

def find_transaction_chain(db_path: str, potential_chains, length: int = 5):
    conn = sqlite3.connect(db_path)
    progress = 0
    total = len(potential_chains)
    print("Chasing potential chains:")
    for chain in potential_chains:
        value_len = len(hex(chain["value"]))
        print(f"Progress: {progress/total*100:.2f}%")
        # print("Chasing chain:", chain)
        # chase chain backwards
        current_wallet = chain["chain"][0]
        current_timestamp = chain["timestamps"][0]
        while True:
            found = False
            prev_tx = conn.execute(
                "SELECT from_hash, value, timestamp, hash FROM transactions WHERE to_hash = ? AND timestamp < ? AND LENGTH(value) > ? AND LENGTH(value) < ? ORDER BY timestamp DESC",
                (current_wallet, current_timestamp,value_len-2, value_len+2,)
            ).fetchall()
            for tx in prev_tx:
                from_hash, value, timestamp, tx_hash = tx
                if abs(int(value, 16) - chain["value"]) < max(int(value, 16) // (10 ** 5), 100) and from_hash not in chain["chain"]:
                    chain["chain"].insert(0, from_hash)
                    chain["transactions"].insert(0, tx_hash)
                    chain["timestamps"].insert(0, timestamp)
                    current_wallet = from_hash
                    current_timestamp = timestamp
                    found = True
                    break
            if not found:
                break
        # chase chain forwards
        current_wallet = chain["chain"][-1]
        current_timestamp = chain["timestamps"][-1]
        while True:
            found = False
            next_tx = conn.execute(
                "SELECT to_hash, value, timestamp, hash FROM transactions WHERE from_hash = ? AND timestamp > ? AND LENGTH(value) > ? AND LENGTH(value) < ? ORDER BY timestamp ASC",
                (current_wallet, current_timestamp,value_len-2, value_len+2,)
            ).fetchall()
            for tx in next_tx:
                to_hash, value, timestamp, tx_hash = tx
                if abs(int(value, 16) - chain["value"]) < max(int(value, 16) // (10 ** 5), 100) and to_hash not in chain["chain"]:
                    chain["chain"].append(to_hash)
                    chain["transactions"].append(tx_hash)
                    chain["timestamps"].append(timestamp)
                    current_wallet = to_hash
                    current_timestamp = timestamp
                    found = True
                    break
            if not found:
                break
        progress += 1
    chains = list(filter(lambda c: len(c["timestamps"]) >= length, potential_chains))
    chains.sort(key=lambda x: len(x["timestamps"]), reverse=True)
    print("Chains found:", len(chains))
    return chains

# Set threshold at 15
def fetch_wallet_repeats(db_path: str, sus_wallets: list[str], threshold: int = 15):
    conn = sqlite3.connect(db_path)
    wallets = []
    for start_wallet in sus_wallets:
        sus_tx_pairs = []
        in_txs = conn.execute(
            "SELECT from_hash, value, timestamp, hash FROM transactions WHERE to_hash = ? AND from_hash != ?",
            (start_wallet, start_wallet,)
        ).fetchall()

        out_txs = conn.execute(
            "SELECT to_hash, value, timestamp, hash FROM transactions WHERE from_hash = ? AND to_hash != ?",
            (start_wallet, start_wallet,)
        ).fetchall()
        all_txs = [{
            "type": "in",
            "hash": tx[3],
            "to_hash": tx[0],
            "value": tx[1],
            "timestamp": tx[2]
        } for tx in in_txs] + [{
            "type": "out",
            "hash": tx[3],
            "from_hash": tx[0],
            "value": tx[1],
            "timestamp": tx[2]
        } for tx in out_txs]
        all_txs.sort(key=lambda x: x["timestamp"])
        counter = 0
        j = 0
        for i in range(len(all_txs)-1):
            tx1 = all_txs[i]
            if tx1["type"] != "in":
                continue
            j = max(j, i+1)
            for tx2 in all_txs[j:]:
                j += 1
                if tx1["type"] == tx2["type"]:
                    continue
                if tx2["timestamp"] - tx1["timestamp"] > 3600:
                    break
                if abs(int(tx1["value"], 16) - int(tx2["value"], 16)) < max(int(tx1["value"], 16) // (10 ** 5), 100):
                    sus_tx_pairs.append((tx1, tx2))
                    counter += 1
                    break
        wallets.append({"wallet": start_wallet, "count": counter, "sus_tx_pairs": sus_tx_pairs})
    wallets = list(filter(lambda x: x["count"] >= threshold, wallets))
    wallets.sort(key=lambda x: x["count"], reverse=True)
    return wallets

def create_alerts(conn: sqlite3.Connection, wallets, chains):
    cur = conn.cursor()
    alert_id = 1
    for chain in chains:
        cur.execute(
            "INSERT INTO chain_alerts (alert_id, chain_length) VALUES (?, ?)", (alert_id, len(chain["chain"]))
        )
        for tx_hash in chain["transactions"]:
            cur.execute(
                "INSERT INTO chain_alert_transactions (alert_id, transaction_hash) VALUES (?, ?)",
                (alert_id, tx_hash)
            )
        for wallet in chain["chain"]:
            cur.execute(
                "INSERT OR IGNORE INTO chain_alert_wallets (alert_id, wallet) VALUES (?, ?)",
                (alert_id, wallet)
            )
        alert_id += 1
    alert_id = 1
    for wallet in wallets:
        cur.execute(
            "INSERT INTO wallet_alerts (alert_id, wallet) VALUES (?, ?)", (alert_id, wallet["wallet"])
        )
        for tx_pair in wallet["sus_tx_pairs"]:
            in_tx = tx_pair[0]
            out_tx = tx_pair[1]
            cur.execute(
                "INSERT INTO wallet_alert_transaction_pairs (alert_id, in_transaction, out_transaction) VALUES (?, ?, ?)",
                (alert_id, in_tx["hash"], out_tx["hash"])
            )
        alert_id += 1
    conn.commit()

def clean_alerts(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM chain_alert_transactions")
    cur.execute("DELETE FROM chain_alert_wallets")
    cur.execute("DELETE FROM chain_alerts")
    cur.execute("DELETE FROM wallet_alert_transaction_pairs")
    cur.execute("DELETE FROM wallet_alerts")
    conn.commit()

def create_alert_dbs(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE transaction_alerts ("
        "   alert_id INTEGER,"
        "   transaction_hash TEXT NOT NULL,"
        "   PRIMARY KEY (alert_id, transaction_hash)"
        ");", ())
    cur.execute(
        "CREATE TABLE account_alerts ("
        "   alert_id INTEGER,"
        "   account_hash TEXT NOT NULL,"
        "   PRIMARY KEY (alert_id, account_hash)"
        "   FOREIGN KEY (alert_id) REFERENCES transaction_alerts(alert_id)"
        ");", ())
    conn.commit()

def clean_wallets_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("DELETE FROM wallets")
    wallets = conn.execute(
            "SELECT from_hash as wallet FROM transactions WHERE value > 0 UNION SELECT to_hash as wallet FROM transactions WHERE value > 0",
            ()
        ).fetchall()
    for wallet in wallets:
        conn.execute(
            "INSERT OR IGNORE INTO wallets (hash) VALUES (?)", (wallet[0],)
            )
    conn.commit()

def create_indexes_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_hash ON transactions(hash);", ()
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_value_len ON transactions(LENGTH(value));", ()
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_from_hash ON transactions(from_hash);", ()
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_to_hash ON transactions(to_hash);", ()
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);", ()
    )
    conn.commit()

def main():
    # create_indexes_db(sqlite3.connect("./blockchain.db"))
    # sus_chains, sus_wallets = fetch_sus_chains_and_wallets("./blockchain.db")
    # with open("./data/sus_chains.json", "w") as f:
    #     f.write(json.dumps(sus_chains, indent=2))
    # with open("./data/sus_wallets.json", "w") as f:
    #     f.write(json.dumps(sus_wallets, indent=2))

    clean_alerts("./blockchain.db")

    sus_chains = json.load(open("./data/chains.json", "r"))
    # chains = find_transaction_chain("./blockchain.db", sus_chains, length=6)
    # with open("./data/chains.json", "w") as f:
    #     f.write(json.dumps(chains, indent=2))

    sus_wallets = json.load(open("./data/wallets.json", "r"))
    # wallets = fetch_wallet_repeats("./blockchain.db", sus_wallets, threshold=15)
    # with open("./data/wallets.json", "w") as f:
    #     f.write(json.dumps(wallets, indent=2))

    create_alerts(sqlite3.connect("./blockchain.db"), sus_wallets, sus_chains)

    # client = ApiClient()
    # asyncio.run(query_n_blocks(client, 10000, 9477668, reverse=True))


if __name__ == "__main__":
    main()
