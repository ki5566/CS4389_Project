import sqlite3
import pandas as pd

def get_transactions():
    conn = sqlite3.connect('blockchain.db')
    cur = conn.cursor()

    cur.execute("""
                SELECT hash, value, from_hash, to_hash, blocknumber, timestamp
                FROM transactions
                """)
    
    transactions = cur.fetchall()
    column_names = ['Transaction Hash', 'Value (wei)', 'From', 'To', 'Block', 'Timestamp']
    df = pd.DataFrame(data=transactions, columns=column_names)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    conn.close()
    return df

def get_tx_count():
    conn = sqlite3.connect('blockchain.db')
    cur = conn.cursor()

    cur.execute("""
                SELECT COUNT(*)
                FROM transactions
                """)
    
    count = cur.fetchone()
    conn.close()

    if count:
        return count[0]
    return None

def get_block_count():
    conn = sqlite3.connect('blockchain.db')
    cur = conn.cursor()

    cur.execute("""
                SELECT COUNT(*)
                FROM blocks
                """)
    
    count = cur.fetchone()
    conn.close()

    if count:
        return count[0]
    return None

def get_avg_value():
    conn = sqlite3.connect('blockchain.db')
    cur = conn.cursor()

    cur.execute("""
                SELECT AVG(value)
                FROM transactions
                """)
    
    count = cur.fetchone()
    conn.close()

    if count:
        return count[0]
    return None