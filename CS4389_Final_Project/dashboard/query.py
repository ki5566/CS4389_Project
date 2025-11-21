import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "backend" / "blockchain.db"


# Connection Manage

@contextmanager
def get_connection() -> sqlite3.Connection:
    """Get a database connection with proper cleanup."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Helpers

def value_to_int(val: Any) -> int:
    """Convert hex or numeric value to integer (wei)."""
    if val is None:
        return 0

    if isinstance(val, str):
        val = val.strip()
        if not val:
            return 0
        if val.startswith("0x"):
            try:
                return int(val, 16)
            except (ValueError, TypeError):
                return 0

    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def wei_to_eth(wei: int | float) -> float:
    """Convert wei to ETH."""
    try:
        return float(wei) / 1e18
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def format_value_display(wei: int | float, mode: str = "auto") -> str:
    """Format value for display with smart unit selection."""
    if mode == "auto":
        eth = wei_to_eth(wei)
        if eth == 0:
            return "0 ETH"
        if eth < 0.0001:
            return f"{float(wei):.3E} wei"
        if eth < 1:
            return f"{eth:.6f} ETH"
        return f"{eth:.4f} ETH"

    if mode == "eth":
        return f"{wei_to_eth(wei):.6f} ETH"

    return f"{float(wei):.3E} wei"

# Optimization

def ensure_indexes() -> None:
    """Create database indexes for optimal performance."""
    with get_connection() as conn:
        cur = conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_hash)",
            "CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_hash)",
            "CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tx_value ON transactions(value)",
            "CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(blocknumber)",
            "CREATE INDEX IF NOT EXISTS idx_tx_from_to ON transactions(from_hash, to_hash)",
            "CREATE INDEX IF NOT EXISTS idx_wallet_hash ON wallets(hash)",
            "CREATE INDEX IF NOT EXISTS idx_alert_id ON transaction_alerts(alert_id)",
        ]

        for idx in indexes:
            try:
                cur.execute(idx)
            except sqlite3.OperationalError:
                pass  # Index may already exist

        conn.commit()

# Core Query Functions

def get_transactions(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
) -> pd.DataFrame:
    """Get transactions as DataFrame."""
    with get_connection() as conn:
        query = """
            SELECT 
                hash,
                value,
                from_hash,
                to_hash,
                blocknumber,
                timestamp
            FROM transactions
        """

        params = []
        conditions = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(int(start_date.timestamp()))

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(int(end_date.timestamp()))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame({
            "Transaction Hash": pd.Series(dtype="string"),
            "From": pd.Series(dtype="string"),
            "To": pd.Series(dtype="string"),
            "Block": pd.Series(dtype="int64"),
            "ValueWei": pd.Series(dtype="int64"),
            "Timestamp": pd.Series(dtype="datetime64[ns]")
        })

    # Convert values
    df["ValueWei"] = df["value"].apply(value_to_int)
    df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Rename columns
    df = df.rename(columns={
        "hash": "Transaction Hash",
        "from_hash": "From",
        "to_hash": "To",
        "blocknumber": "Block",
    })

    return df[["Transaction Hash", "From", "To", "Block", "ValueWei", "Timestamp"]]


def get_tx_count() -> int:
    """Get total transaction count."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transactions")
        result = cur.fetchone()
        return int(result[0]) if result else 0


def get_block_count() -> int:
    """Get total block count."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM blocks")
        result = cur.fetchone()
        return int(result[0]) if result else 0


def get_wallet_count() -> int:
    """Get total wallet count."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM wallets")
        result = cur.fetchone()
        return int(result[0]) if result else 0


def get_active_wallet_count(days: int = 1) -> int:
    """Get count of wallets active in last N days."""
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT address) FROM (
                SELECT from_hash as address FROM transactions WHERE timestamp > ?
                UNION
                SELECT to_hash as address FROM transactions WHERE timestamp > ?
            )
        """, (cutoff, cutoff))

        result = cur.fetchone()
        return int(result[0]) if result else 0


def get_avg_value() -> Optional[float]:
    """Get average transaction value in wei."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT AVG(CAST(value AS REAL)) 
            FROM transactions 
            WHERE value != '0x0' AND value != '0' AND value IS NOT NULL
        """)
        result = cur.fetchone()

        if result and result[0]:
            try:
                # Convert hex average to proper value
                return float(result[0])
            except (TypeError, ValueError):
                pass

    return None


def get_value_statistics() -> Dict[str, float]:
    """Get comprehensive value statistics."""
    with get_connection() as conn:
        # Get non-zero values
        df = pd.read_sql_query("""
            SELECT value 
            FROM transactions 
            WHERE value != '0x0' AND value != '0' AND value IS NOT NULL
            LIMIT 100000
        """, conn)

        if df.empty:
            return {
                'avg_wei': 0, 'median_wei': 0, 'min_wei': 0, 'max_wei': 0,
                'avg_eth': 0, 'median_eth': 0, 'min_eth': 0, 'max_eth': 0,
                'std_wei': 0, 'total_wei': 0, 'total_eth': 0
            }

        values = df['value'].apply(value_to_int)
        values = values[values > 0]  # Filter zero values

        if len(values) == 0:
            return {
                'avg_wei': 0, 'median_wei': 0, 'min_wei': 0, 'max_wei': 0,
                'avg_eth': 0, 'median_eth': 0, 'min_eth': 0, 'max_eth': 0,
                'std_wei': 0, 'total_wei': 0, 'total_eth': 0
            }

        avg_wei = float(values.mean())
        median_wei = float(values.median())
        min_wei = float(values.min())
        max_wei = float(values.max())
        std_wei = float(values.std())
        total_wei = float(values.sum())

        return {
            'avg_wei': avg_wei,
            'median_wei': median_wei,
            'min_wei': min_wei,
            'max_wei': max_wei,
            'std_wei': std_wei,
            'total_wei': total_wei,
            'avg_eth': wei_to_eth(avg_wei),
            'median_eth': wei_to_eth(median_wei),
            'min_eth': wei_to_eth(min_wei),
            'max_eth': wei_to_eth(max_wei),
            'total_eth': wei_to_eth(total_wei)
        }


def get_network_velocity(hours: int = 1) -> Dict[str, float]:
    """Calculate network velocity metrics."""
    cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())

    with get_connection() as conn:
        cur = conn.cursor()

        # Get transaction count, time span
        cur.execute("""
            SELECT 
                COUNT(*) as tx_count,
                MIN(timestamp) as first_tx,
                MAX(timestamp) as last_tx
            FROM transactions
            WHERE timestamp > ?
        """, (cutoff,))

        result = cur.fetchone()

        if result and result['tx_count'] > 0:
            tx_count = result['tx_count']
            time_span = max(result['last_tx'] - result['first_tx'], 1)

            tx_per_second = tx_count / time_span
            tx_per_minute = tx_per_second * 60
            tx_per_hour = tx_per_second * 3600

            return {
                'tx_count': tx_count,
                'tx_per_second': tx_per_second,
                'tx_per_minute': tx_per_minute,
                'tx_per_hour': tx_per_hour,
                'time_window_hours': hours,
                'time_span_seconds': time_span
            }

    # If no recent transactions, calculate from all data
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total_tx,
                MIN(timestamp) as first_tx,
                MAX(timestamp) as last_tx
            FROM transactions
        """)

        result = cur.fetchone()
        if result and result['total_tx'] > 0:
            total_tx = result['total_tx']
            total_span = max(result['last_tx'] - result['first_tx'], 3600)

            return {
                'tx_count': 0,  # No recent transactions
                'tx_per_second': 0,
                'tx_per_minute': 0,
                'tx_per_hour': (total_tx / total_span) * 3600,  # Historical average
                'time_window_hours': hours,
                'time_span_seconds': 0
            }

    return {
        'tx_count': 0,
        'tx_per_second': 0,
        'tx_per_minute': 0,
        'tx_per_hour': 0,
        'time_window_hours': hours,
        'time_span_seconds': 0
    }


def get_hourly_statistics(date: Optional[datetime] = None) -> pd.DataFrame:
    """Get hourly transaction statistics."""
    with get_connection() as conn:
        if date:
            # Specific date
            start = int(datetime(date.year, date.month, date.day).timestamp())
            end = start + 86400

            df = pd.read_sql_query("""
                SELECT 
                    CAST(strftime('%H', datetime(timestamp, 'unixepoch')) AS INTEGER) as hour,
                    COUNT(*) as tx_count,
                    AVG(CAST(value AS REAL)) as avg_value_hex
                FROM transactions
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY hour
                ORDER BY hour
            """, conn, params=[start, end])
        else:
            # All dates
            df = pd.read_sql_query("""
                SELECT 
                    date(datetime(timestamp, 'unixepoch')) as date,
                    CAST(strftime('%H', datetime(timestamp, 'unixepoch')) AS INTEGER) as hour,
                    COUNT(*) as tx_count,
                    AVG(CAST(value AS REAL)) as avg_value_hex
                FROM transactions
                GROUP BY date, hour
                ORDER BY date DESC, hour
                LIMIT 168
            """, conn)

    if not df.empty and 'avg_value_hex' in df.columns:
        # Convert hex averages to proper values
        df['avg_value_wei'] = df['avg_value_hex'].apply(
            lambda x: value_to_int(int(x)) if pd.notna(x) and x != 0 else 0
        )
        df['avg_value_eth'] = df['avg_value_wei'].apply(wei_to_eth)

    return df


def get_top_addresses(
    n: int = 10,
    by: str = "value",
    address_type: str = "both"
) -> pd.DataFrame:
    """
    Get top addresses by total transferred value or transaction count.

    Values are converted from hex to integer wei BEFORE aggregation so totals
    are numerically correct.
    """
    with get_connection() as conn:

        if address_type == "sender":
            base_query = """
                SELECT from_hash AS address, value
                FROM transactions
                WHERE value IS NOT NULL
            """
        elif address_type == "receiver":
            base_query = """
                SELECT to_hash AS address, value
                FROM transactions
                WHERE value IS NOT NULL
            """
        else:  # both sender and receiver
            base_query = """
                SELECT from_hash AS address, value
                FROM transactions
                WHERE value IS NOT NULL
                UNION ALL
                SELECT to_hash AS address, value
                FROM transactions
                WHERE value IS NOT NULL
            """

        df = pd.read_sql_query(base_query, conn)

    if df.empty:
        return df

    # Convert each value to integer wei, then aggregate
    df["value_wei"] = df["value"].apply(value_to_int)

    grouped = (
        df.groupby("address", dropna=True)["value_wei"]
        .agg(tx_count="count", total_value_wei="sum")
        .reset_index()
    )

    if grouped.empty:
        return grouped

    # Sort by total value/count
    sort_col = "tx_count" if by == "count" else "total_value_wei"
    grouped = grouped.sort_values(sort_col, ascending=False).head(n)

    # Add ETH and shortened address for display
    grouped["total_value_eth"] = grouped["total_value_wei"].apply(wei_to_eth)
    grouped["address_short"] = grouped["address"].apply(
        lambda x: f"{x[:6]}...{x[-4:]}" if isinstance(x, str) and len(x) > 10 else x
    )

    return grouped



def get_alert_summary() -> Dict[str, int]:
    """Get alert summary statistics."""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(alert_id) as chain_alerts
            FROM chain_alerts
        """)
        chain_alerts = cur.fetchone()

        cur.execute("""
            SELECT COUNT(alert_id) as wallet_alerts
            FROM wallet_alerts
        """)
        wallet_alerts = cur.fetchone()

        return {
            "chain_alerts": chain_alerts[0],
            "wallet_alerts": wallet_alerts[0],
            "total": chain_alerts[0] + wallet_alerts[0]
        }


def get_chain_priority(chain_len: int) -> str:
    if chain_len >= 20:
        return 'high'
    elif chain_len >= 10:
        return 'med'
    else:
        return 'low'

def get_account_priority(tx_count: int) -> str:
    """Algorithm 2: Priority based on number of transactions for an account."""
    if tx_count >= 50:
        return 'high'
    elif tx_count >= 20:
        return 'med'
    else:
        return 'low'

def get_timeframe_priority(tx_count: int) -> str:
    """Algorithm 3: Priority based on transactions within time frame."""
    if tx_count >= 30:
        return 'high'
    elif tx_count >= 10:
        return 'med'
    else:
        return 'low'
    
def get_full_alert_data():
    """Get all alerts from alert tables, sorted by priority. Uses real data from chain_alerts and wallet_alerts tables."""
    with get_connection() as conn:
        cur = conn.cursor()

        all_alerts = pd.DataFrame(columns=['alert_id', 'algorithm', 'priority', 'timestamp', 'details'])
        
        # Algorithm 1: Chain-based alerts (priority based on chain length)
        try:
            cur.execute("""
                SELECT chain_alerts.alert_id, chain_length, time
                FROM chain_alerts JOIN 
                    (SELECT alert_id, MAX(timestamp) AS time
                    FROM chain_alert_transactions AS chain_txs JOIN transactions AS txs
                        ON chain_txs.transaction_hash = txs.hash
                    GROUP BY alert_id) AS alert_times
                ON chain_alerts.alert_id = alert_times.alert_id
                ORDER BY chain_length DESC
            """)
            chain_alerts = cur.fetchall()

            for alert_id, chain_len, time in chain_alerts:
                priority = get_chain_priority(int(chain_len))
                timestamp = time if time is not None else 0
                # Count transactions in this chain
                cur.execute("SELECT COUNT(*) FROM chain_alert_transactions WHERE alert_id = ?", (alert_id,))
                tx_count = cur.fetchone()[0]
                all_alerts.loc[len(all_alerts)] = [
                    alert_id, 
                    'Chain Detection', 
                    priority, 
                    timestamp,
                    f"Chain length: {chain_len} | Transactions: {tx_count}"
                ]
        except Exception as e:
            logger.warning(f"Error fetching chain alerts: {e}")

        # Algorithm 2: Wallet/Account alerts (priority based on transaction count)
        try:
            cur.execute("""
                SELECT 
                    wa.alert_id,
                    wa.wallet,
                    COUNT(DISTINCT watp.in_transaction || watp.out_transaction) as tx_pair_count,
                    MAX(COALESCE(t1.timestamp, t2.timestamp)) as last_timestamp
                FROM wallet_alerts wa
                LEFT JOIN wallet_alert_transaction_pairs watp ON wa.alert_id = watp.alert_id
                LEFT JOIN transactions t1 ON watp.in_transaction = t1.hash
                LEFT JOIN transactions t2 ON watp.out_transaction = t2.hash
                GROUP BY wa.alert_id, wa.wallet
                ORDER BY tx_pair_count DESC
            """)
            wallet_alerts = cur.fetchall()

            for alert_id, wallet, tx_count, timestamp in wallet_alerts:
                priority = get_account_priority(int(tx_count) if tx_count else 0)
                ts = timestamp if timestamp is not None else 0
                wallet_str = str(wallet) if wallet else 'Unknown'  # Show full hash, not truncated
                all_alerts.loc[len(all_alerts)] = [
                    alert_id,
                    'Account Activity',
                    priority,
                    ts,
                    f"Account: {wallet_str} | Transaction pairs: {tx_count}"
                ]
        except Exception as e:
            logger.warning(f"Error fetching wallet alerts: {e}")

        # Algorithm 3: Time-based alerts (reads from time_based_alerts table)
        try:
            # Read from time_based_alerts table in database
            cur.execute("""
                SELECT 
                    alert_id,
                    account,
                    transaction_count,
                    timestamp
                FROM time_based_alerts
                ORDER BY transaction_count DESC, timestamp DESC
            """)
            timeframe_alerts = cur.fetchall()

            for alert_id, account, tx_count, timestamp in timeframe_alerts:
                priority = get_timeframe_priority(int(tx_count) if tx_count else 0)
                ts = timestamp if timestamp is not None else 0
                account_str = str(account) if account else 'Unknown'  # Show full hash, not truncated
                all_alerts.loc[len(all_alerts)] = [
                    alert_id,
                    'Time-based Activity',
                    priority,
                    ts,
                    f"Account: {account_str} | Transactions in last hour: {tx_count}"
                ]
        except Exception as e:
            # If table doesn't exist yet, just log warning and continue
            logger.warning(f"Error fetching time-based alerts (table may not exist yet): {e}")

        # Sort by priority: high -> med -> low, then by timestamp (newest first)
        if not all_alerts.empty:
            priority_order = {'high': 0, 'med': 1, 'low': 2}
            all_alerts['priority_order'] = all_alerts['priority'].map(priority_order)
            all_alerts = all_alerts.sort_values(['priority_order', 'timestamp'], ascending=[True, False])
            all_alerts = all_alerts.drop('priority_order', axis=1)

        return all_alerts

def get_alert_info(alert_id, type):
    with get_connection() as conn:
        cur = conn.cursor()
    
        if type == 'chain':
            cur.execute(f"SELECT chain_length FROM chain_alerts WHERE alert_id = '{alert_id}'")
            chain_len = cur.fetchone()[0]
            tx_info = pd.read_sql_query(f"""
                            SELECT * 
                            FROM chain_alert_transactions AS chain_txs JOIN transactions AS txs
                                ON chain_txs.transaction_hash = txs.hash
                            WHERE chain_txs.alert_id = '{alert_id}'
                        """, conn)
            
            return {"chain_len": chain_len, "tx_info": tx_info}
        
        elif type == 'wallet':
            cur.execute(f"SELECT wallet FROM wallet_alerts WHERE alert_id = '{alert_id}'")
            wallet = cur.fetchone()[0]
            tx_info = pd.read_sql_query(f"""
                    SELECT in_hash, out_hash, in_value, out_value    
                    FROM
                        (SELECT wallet_pairs.rowid AS id, hash AS in_hash, value AS in_value
                        FROM wallet_alert_transaction_pairs AS wallet_pairs JOIN transactions AS txs
                            ON wallet_pairs.in_transaction = txs.hash
                        WHERE alert_id = '{alert_id}') AS in_txs
                    JOIN
                        (SELECT wallet_pairs.rowid AS id, hash AS out_hash, value AS out_value
                        FROM wallet_alert_transaction_pairs AS wallet_pairs JOIN transactions AS txs
                            ON wallet_pairs.out_transaction = txs.hash
                        WHERE alert_id = '{alert_id}') AS out_txs
                    ON in_txs.id = out_txs.id
                    ORDER BY in_txs.id
            """, conn)
            return {"wallet": wallet, "tx_info": tx_info}
        
        elif type == 'timebased' or type == 'time-based':
            cur.execute(f"SELECT account, transaction_count, timestamp FROM time_based_alerts WHERE alert_id = '{alert_id}'")
            result = cur.fetchone()
            if result:
                account, tx_count, timestamp = result
                # Get transactions for this account in the timeframe (last hour from timestamp)
                tx_info = pd.read_sql_query(f"""
                    SELECT hash, from_hash, to_hash, value, timestamp
                    FROM transactions
                    WHERE (from_hash = ? OR to_hash = ?)
                    AND timestamp > ? - 3600
                    AND timestamp <= ?
                    ORDER BY timestamp DESC
                """, conn, params=(account, account, timestamp, timestamp))
                return {"account": account, "transaction_count": tx_count, "timestamp": timestamp, "tx_info": tx_info}
            return {"account": None, "transaction_count": 0, "timestamp": 0, "tx_info": pd.DataFrame()}

def get_total_alert_count() -> int:
    """Get total alert count."""
    summary = get_alert_summary()
    return summary.get("total", 0)

def get_accounts_with_alert_priority():
    """Get accounts with their alert counts, sorted by alert count (no priority)."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Get accounts from chain alerts
        cur.execute("""
            SELECT DISTINCT caw.wallet as account, COUNT(DISTINCT ca.alert_id) as alert_count
            FROM chain_alert_wallets caw
            JOIN chain_alerts ca ON caw.alert_id = ca.alert_id
            GROUP BY caw.wallet
        """)
        chain_accounts = cur.fetchall()
        
        # Get accounts from wallet alerts
        cur.execute("""
            SELECT wallet as account, COUNT(DISTINCT alert_id) as alert_count
            FROM wallet_alerts
            GROUP BY wallet
        """)
        wallet_accounts = cur.fetchall()
        
        # Combine and aggregate
        account_dict = {}
        for account, count in chain_accounts:
            account_str = str(account)
            account_dict[account_str] = account_dict.get(account_str, 0) + count
        
        for account, count in wallet_accounts:
            account_str = str(account)
            account_dict[account_str] = account_dict.get(account_str, 0) + count
        
        # Create DataFrame without priorities
        accounts_list = []
        for account, alert_count in account_dict.items():
            accounts_list.append({
                'account': account,
                'alert_count': alert_count
            })
        
        if accounts_list:
            df = pd.DataFrame(accounts_list)
            # Sort by alert count descending
            df = df.sort_values('alert_count', ascending=False)
            return df
        return pd.DataFrame(columns=['account', 'alert_count'])

def get_account_details(account_hash: str):
    """Get full details for a specific account."""
    with get_connection() as conn:
        cur = conn.cursor()
        details = {
            'account': account_hash,
            'chain_alerts': [],
            'wallet_alerts': [],
            'transactions': []
        }
        
        # Get chain alerts for this account
        cur.execute("""
            SELECT DISTINCT ca.alert_id, ca.chain_length
            FROM chain_alert_wallets caw
            JOIN chain_alerts ca ON caw.alert_id = ca.alert_id
            WHERE caw.wallet = ?
        """, (account_hash,))
        chain_alerts = cur.fetchall()
        for alert_id, chain_len in chain_alerts:
            details['chain_alerts'].append({
                'alert_id': alert_id,
                'chain_length': chain_len,
                'priority': get_chain_priority(int(chain_len))
            })
        
        # Get wallet alerts for this account
        cur.execute("""
            SELECT alert_id FROM wallet_alerts WHERE wallet = ?
        """, (account_hash,))
        wallet_alerts = cur.fetchall()
        for (alert_id,) in wallet_alerts:
            cur.execute("""
                SELECT COUNT(*) FROM wallet_alert_transaction_pairs WHERE alert_id = ?
            """, (alert_id,))
            tx_count = cur.fetchone()[0]
            details['wallet_alerts'].append({
                'alert_id': alert_id,
                'transaction_pairs': tx_count,
                'priority': get_account_priority(int(tx_count))
            })
        
        # Get transaction statistics
        cur.execute("""
            SELECT 
                COUNT(DISTINCT hash) as tx_count,
                COUNT(DISTINCT CASE WHEN from_hash = ? THEN hash END) as sent_count,
                COUNT(DISTINCT CASE WHEN to_hash = ? THEN hash END) as received_count,
                SUM(CASE WHEN from_hash = ? THEN CAST(value AS INTEGER) ELSE 0 END) as total_sent,
                SUM(CASE WHEN to_hash = ? THEN CAST(value AS INTEGER) ELSE 0 END) as total_received
            FROM transactions
            WHERE from_hash = ? OR to_hash = ?
        """, (account_hash, account_hash, account_hash, account_hash, account_hash, account_hash))
        tx_stats = cur.fetchone()
        
        details['transaction_stats'] = {
            'total_transactions': tx_stats[0] if tx_stats else 0,
            'sent_count': tx_stats[1] if tx_stats else 0,
            'received_count': tx_stats[2] if tx_stats else 0,
            'total_sent': tx_stats[3] if tx_stats else 0,
            'total_received': tx_stats[4] if tx_stats else 0
        }
        
        return details


def get_suspicious_patterns() -> Dict[str, List]:
    """Detect suspicious patterns in recent transactions."""
    patterns = {
        'rapid_transfers': [],
        'same_value_chains': [],
        'circular_flows': [],
        'high_frequency': []
    }

    one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp())

    with get_connection() as conn:
        # Rapid transfers- addresses with many transactions in a short time
        rapid_df = pd.read_sql_query("""
            SELECT 
                from_hash,
                COUNT(*) as tx_count,
                MIN(timestamp) as first_tx,
                MAX(timestamp) as last_tx
            FROM transactions
            WHERE timestamp > ?
            GROUP BY from_hash
            HAVING COUNT(*) > 5
            ORDER BY tx_count DESC
            LIMIT 10
        """, conn, params=[one_hour_ago])

        if not rapid_df.empty:
            patterns['rapid_transfers'] = rapid_df.to_dict('records')

        high_freq_df = pd.read_sql_query("""
            SELECT 
                address,
                SUM(tx_count) as total_tx
            FROM (
                SELECT from_hash as address, COUNT(*) as tx_count
                FROM transactions
                GROUP BY from_hash
                HAVING COUNT(*) > 20

                UNION ALL

                SELECT to_hash as address, COUNT(*) as tx_count
                FROM transactions
                GROUP BY to_hash
                HAVING COUNT(*) > 20
            )
            GROUP BY address
            ORDER BY total_tx DESC
            LIMIT 5
        """, conn)

        if not high_freq_df.empty:
            patterns['high_frequency'] = high_freq_df.to_dict('records')

    return patterns

# Initialization

try:
    ensure_indexes()
    logger.info("Database indexes verified")
except Exception as e:
    logger.warning(f"Could not ensure indexes: {e}")
