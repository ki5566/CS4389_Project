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
DB_PATH = Path(__file__).resolve().parent / "blockchain.db"

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

        # Check if alert tables exist
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('alerts', 'transaction_alerts', 'account_alerts')
        """)
        tables = [row[0] for row in cur.fetchall()]

        if 'alerts' in tables:
            # Get alert counts by type
            cur.execute("""
                SELECT 
                    COUNT(CASE WHEN alert_type = 'transaction' THEN 1 END) as tx_alerts,
                    COUNT(CASE WHEN alert_type = 'account' THEN 1 END) as acct_alerts,
                    COUNT(*) as total
                FROM alerts
            """)
            result = cur.fetchone()
            if result:
                return {
                    "transaction_alerts": result[0] or 0,
                    "account_alerts": result[1] or 0,
                    "total": result[2] or 0
                }

        # Fallback
        tx_alerts = 0
        acct_alerts = 0

        if 'transaction_alerts' in tables:
            cur.execute("SELECT COUNT(DISTINCT alert_id) FROM transaction_alerts")
            result = cur.fetchone()
            tx_alerts = result[0] if result else 0

        if 'account_alerts' in tables:
            cur.execute("SELECT COUNT(DISTINCT alert_id) FROM account_alerts")
            result = cur.fetchone()
            acct_alerts = result[0] if result else 0

        return {
            "transaction_alerts": tx_alerts,
            "account_alerts": acct_alerts,
            "total": tx_alerts + acct_alerts
        }


def get_total_alert_count() -> int:
    """Get total alert count."""
    summary = get_alert_summary()
    return summary.get("total", 0)


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
                HAVING COUNT(*) > 100

                UNION ALL

                SELECT to_hash as address, COUNT(*) as tx_count
                FROM transactions
                GROUP BY to_hash
                HAVING COUNT(*) > 100
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
