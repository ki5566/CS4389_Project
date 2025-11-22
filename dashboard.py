import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import numpy as np
import query

st.set_page_config(
    page_title="Mitigation of Blockchain Fraud - Blockchain Monitoring Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Real-time blockchain transaction monitoring and fraud detection system"
    }
)

st.markdown("""
<style>
    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Alert severity colors */
    .alert-high { color: #ff4b4b; font-weight: bold; }
    .alert-medium { color: #ffa500; font-weight: bold; }
    .alert-low { color: #00cc00; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    /* Main tab navigation styling - make radio buttons look like tabs */
    div[data-testid="stRadio"]:has([name*="main_tab_selector"]) > div {
        flex-direction: row;
        gap: 0px;
        background-color: rgba(28, 131, 225, 0.05);
        border-radius: 8px;
        padding: 4px;
    }
    
    div[data-testid="stRadio"]:has([name*="main_tab_selector"]) label {
        padding: 12px 24px;
        margin: 0px;
        border-radius: 6px;
        font-size: 1em;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"]:has([name*="main_tab_selector"]) label:hover {
        background-color: rgba(28, 131, 225, 0.1);
    }
    
    /* Priority filter styling - horizontal and compact */
    div[data-testid="stRadio"]:has([name*="priority_radio"]) > div {
        flex-direction: row;
        gap: 10px;
    }
    
    div[data-testid="stRadio"]:has([name*="priority_radio"]) label {
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.9em;
        transition: background-color 0.2s;
    }
    
    div[data-testid="stRadio"]:has([name*="priority_radio"]) label:hover {
        background-color: rgba(28, 131, 225, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading transactions...")
def load_transactions() -> pd.DataFrame:
    """Load transactions with caching."""
    return query.get_transactions()

@st.cache_data(show_spinner="Loading alert data...")
def load_full_alerts() -> pd.DataFrame:
    """Load alerts with caching."""
    return query.get_full_alert_data()

@st.cache_data(show_spinner="Loading account data...")
def load_accounts() -> pd.DataFrame:
    """Load accounts with caching."""
    return query.get_accounts_with_alert_priority()

@st.cache_data(show_spinner=False)
def load_summary() -> Dict:
    """Load aggregated summary metrics."""
    summary = {
        "tx_count": query.get_tx_count(),
        "block_count": query.get_block_count(),
        "wallet_count": query.get_wallet_count(),
        "active_wallets": query.get_active_wallet_count(days=1),
        "avg_value_wei": query.get_avg_value(),
        "value_stats": query.get_value_statistics(),
        "alert_summary": query.get_alert_summary(),
        "total_alerts": query.get_total_alert_count(),
        "network_velocity": query.get_network_velocity(hours=1),
    }


    if summary.get("network_velocity", {}).get("tx_per_hour", 0) == 0:
        summary["network_velocity"]["tx_per_hour"] = summary["tx_count"] / 24 if summary["tx_count"] > 0 else 0

    return summary


@st.cache_data(show_spinner=False)
def load_hourly_stats(date: Optional[datetime] = None) -> pd.DataFrame:
    """Load hourly transaction statistics."""
    return query.get_hourly_statistics(date)


@st.cache_data(show_spinner=False)
def load_top_addresses(n: int = 10, by: str = "value") -> pd.DataFrame:
    """Load top addresses by volume or count."""
    return query.get_top_addresses(n=n, by=by, address_type="both")


@st.cache_data(show_spinner=False)
def load_suspicious_patterns() -> Dict:
    """Load suspicious transaction patterns."""
    return query.get_suspicious_patterns()


def format_wei_to_eth(wei: int | float) -> float:
    """Convert wei to ETH."""
    try:
        return float(query.wei_to_eth(wei))
    except Exception:
        return 0.0


def format_large_number(num: float | int) -> str:
    """Format large numbers with K/M/B suffixes."""
    num = float(num)
    if abs(num) < 1000:
        return f"{num:.0f}"
    if abs(num) < 1_000_000:
        return f"{num / 1_000:.1f}K"
    if abs(num) < 1_000_000_000:
        return f"{num / 1_000_000:.2f}M"
    return f"{num / 1_000_000_000:.2f}B"


def format_address(addr: str) -> str:
    """Format address for display (shortened)."""
    if not addr or len(addr) < 10:
        return addr
    return f"{addr[:6]}...{addr[-4:]}"


def create_daily_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create daily volume bar chart with date-only labels."""

    daily = df.groupby("Date")["ValueEth"].sum().reset_index()
    daily.columns = ["Date", "Volume"]

    daily["DateLabel"] = pd.to_datetime(daily["Date"]).dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["DateLabel"],
            y=daily["Volume"],
            name="Total Volume (ETH)",
            marker_color="#1f77b4",
            hovertemplate="Date: %{x}<br>Volume: %{y:.6f} ETH<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="ETH",
        height=400,
        showlegend=False,
        hovermode="x unified",
    )

    return fig



def create_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create hourly activity heatmap."""

    df_copy = df.copy()
    df_copy["DayOfWeek"] = df_copy["Timestamp"].dt.day_name()
    df_copy["Hour"] = df_copy["Hour"]

    heatmap_data = df_copy.groupby(["DayOfWeek", "Hour"])["ValueEth"].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="Hour", values="ValueEth")

    # chronological day order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=list(range(24)),
        y=heatmap_pivot.index,
        colorscale="Viridis",
        colorbar=dict(title="Avg ETH"),
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Avg: %{z:.6f} ETH<extra></extra>"
    ))

    fig.update_layout(
        title="Average Transaction Value by Hour and Day",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Day of Week", autorange="reversed"),  # 👈 key change
        height=400
    )

    return fig



def create_value_distribution(df: pd.DataFrame) -> go.Figure:
    """Create value distribution histogram."""
    non_zero = df[df["ValueEth"] > 0]["ValueEth"]

    if len(non_zero) == 0:
        return go.Figure().add_annotation(
            text="No non-zero value transactions",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


    log_values = np.log10(non_zero + 1e-10)

    fig = go.Figure(data=[
        go.Histogram(
            x=log_values,
            nbinsx=50,
            marker_color="#2ca02c",
            opacity=0.75
        )
    ])

    fig.update_layout(
        xaxis_title="log10(Value ETH)",
        yaxis_title="Count",
        height=400,
        bargap=0.1
    )

    return fig


def display_metrics_row(summary: Dict):
    """Display main metrics row."""
    cols = st.columns(6)

    with cols[0]:
        st.metric(
            "Blocks",
            format_large_number(summary["block_count"])
        )

    with cols[1]:
        st.metric(
            "Transactions",
            format_large_number(summary["tx_count"])
        )

    with cols[2]:
        st.metric(
            "Wallets",
            format_large_number(summary["wallet_count"])
        )

    with cols[3]:
        active = summary.get("active_wallets", 0)
        total = max(summary["wallet_count"], 1)
        st.metric(
            "Active (24h)",
            format_large_number(active),
            delta=f"{(active / total * 100):.1f}%" if active > 0 else None
        )

    with cols[4]:
        velocity = summary.get("network_velocity", {})
        tx_per_hour = velocity.get("tx_per_hour", 0)
        st.metric(
            "TX / Hour",
            format_large_number(tx_per_hour) if tx_per_hour > 0 else "0"
        )

    with cols[5]:
        alerts = summary["total_alerts"]
        st.metric(
            "Total Alerts",
            format_large_number(alerts),
            delta="Active" if alerts > 0 else None
        )


def display_value_metrics(summary: Dict):
    """Display value metrics row."""
    stats = summary.get("value_stats", {})

    cols = st.columns(4)

    with cols[0]:
        avg_eth = stats.get("avg_eth", 0)
        st.metric(
            "Avg Value",
            f"{avg_eth:.6f} ETH" if avg_eth > 0 else "0 ETH"
        )

    with cols[1]:
        median_eth = stats.get("median_eth", 0)
        st.metric(
            "Median Value",
            f"{median_eth:.6f} ETH" if median_eth > 0 else "0 ETH"
        )

    with cols[2]:
        min_eth = stats.get("min_eth", 0)
        st.metric(
            "Min Value",
            f"{min_eth:.6f} ETH" if min_eth > 0 else "0 ETH"
        )

    with cols[3]:
        max_eth = stats.get("max_eth", 0)
        st.metric(
            "Max Value",
            f"{max_eth:.6f} ETH" if max_eth > 0 else "0 ETH"
        )


def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe."""
    filtered = df.copy()

    if "date_range" in filters:
        start, end = filters["date_range"]
        filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]

    if "value_range" in filters:
        min_val, max_val = filters["value_range"]
        filtered = filtered[(filtered["ValueEth"] >= min_val) & (filtered["ValueEth"] <= max_val)]

    if filters.get("address"):
        addr = filters["address"].strip().lower()
        mask = (
                filtered["From"].str.lower().str.contains(addr, na=False) |
                filtered["To"].str.lower().str.contains(addr, na=False)
        )
        filtered = filtered[mask]

    tx_type = filters.get("tx_type", "All")
    if tx_type == "With Value":
        filtered = filtered[filtered["ValueEth"] > 0]
    elif tx_type == "Zero Value":
        filtered = filtered[filtered["ValueEth"] == 0]

    return filtered

def show_alert_details(alert_row):
    st.header("Alert Details")
    st.markdown("---")
    
    st.subheader("Basic Information")
    st.write(f"**Alert ID:** {alert_row['alert_id']}")
    st.write(f"**Algorithm:** {alert_row['algorithm']}")
    st.write(f"**Priority:** {alert_row['priority'].upper()}")
    
    if alert_row['timestamp']:
        try:
            timestamp = pd.to_datetime(alert_row['timestamp'], unit='s', errors='coerce')
            st.write(f"**Timestamp:** {timestamp}")
        except:
            st.write(f"**Timestamp:** {alert_row['timestamp']}")
    
    st.write(f"**Details:** {alert_row['details']}")
    
    # Get additional details based on algorithm type
    if alert_row['algorithm'] == 'Chain Detection':
        try:
            alert_info = query.get_alert_info(alert_row['alert_id'], 'chain')
            st.markdown("---")
            st.subheader("Chain Information")
            st.write(f"**Chain Length:** {alert_info.get('chain_len', 'N/A')}")
            if not alert_info.get('tx_info', pd.DataFrame()).empty:
                st.write("**Transactions in Chain:**")
                tx_df = alert_info['tx_info']
                
                # Configure column display to show full hashes
                column_config = {}
                # Find hash columns and configure them to show full values without truncation
                for col in tx_df.columns:
                    col_lower = col.lower()
                    if 'hash' in col_lower or col_lower == 'hash' or 'from' in col_lower or 'to' in col_lower:
                        # Use TextColumn with no width limit to show full hash
                        column_config[col] = st.column_config.TextColumn(
                            col.replace('_', ' ').title(),
                            width=None  # No width limit - will use full available space
                        )

                st.dataframe(
                    tx_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config=column_config if column_config else None
                )
        except Exception as e:
            st.warning(f"Could not load chain details: {e}")
    
    elif alert_row['algorithm'] == 'Account Activity':
        try:
            alert_info = query.get_alert_info(alert_row['alert_id'], 'wallet')
            st.markdown("---")
            st.subheader("Account Information")
            wallet_hash = alert_info.get('wallet', 'N/A')
            # Display full wallet hash - use code block to show full hash without truncation
            st.write("**Account Hash:**")
            st.code(wallet_hash, language=None)
            if not alert_info.get('tx_info', pd.DataFrame()).empty:
                st.write("**Transaction Pairs:**")
                tx_df = alert_info['tx_info']
                
                # Configure column display to show full hashes
                column_config = {}
                for col in tx_df.columns:
                    col_lower = col.lower()
                    if 'hash' in col_lower:
                        # Use TextColumn with no width limit to show full hash
                        column_config[col] = st.column_config.TextColumn(
                            col.replace('_', ' ').title(),
                            width=None  # No width limit - will use full available space
                        )
                
                st.dataframe(
                    tx_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config=column_config if column_config else None
                )
        except Exception as e:
            st.warning(f"Could not load account details: {e}")
    
    elif alert_row['algorithm'] == 'Time-based Activity':
        try:
            alert_info = query.get_alert_info(alert_row['alert_id'], 'timebased')
            st.markdown("---")
            st.subheader("Account Information")
            account_hash = alert_info.get('account', 'N/A')
            # Display full account hash - use code block to show full hash without truncation
            st.write("**Account Hash:**")
            st.code(account_hash, language=None)
            st.write(f"**Transaction Count:** {alert_info.get('transaction_count', 0)}")
            if alert_info.get('timestamp'):
                try:
                    timestamp = pd.to_datetime(alert_info['timestamp'], unit='s', errors='coerce')
                    st.write(f"**Last Transaction Time:** {timestamp}")
                except:
                    st.write(f"**Last Transaction Time:** {alert_info['timestamp']}")
            
            if not alert_info.get('tx_info', pd.DataFrame()).empty:
                st.write("**Recent Transactions (last hour):**")
                tx_df = alert_info['tx_info']
                
                # Configure column display to show full hashes
                column_config = {}
                for col in tx_df.columns:
                    col_lower = col.lower()
                    if 'hash' in col_lower:
                        # Use TextColumn with no width limit to show full hash
                        column_config[col] = st.column_config.TextColumn(
                            col.replace('_', ' ').title(),
                            width=None  # No width limit - will use full available space
                        )
                
                st.dataframe(
                    tx_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config=column_config if column_config else None
                )
        except Exception as e:
            st.warning(f"Could not load time-based alert details: {e}")

def show_account_details(account_row):
    """Show account details."""

    st.header("Account Details")
    st.markdown("---")
    
    account_hash = account_row['account']
    st.subheader("Basic Information")
    st.write(f"**Account Hash:** {account_hash}")
    st.write(f"**Total Alert Count:** {account_row['alert_count']}")
    
    # Get full account details
    try:
        account_details = query.get_account_details(account_hash)
        
        st.markdown("---")
        st.subheader("Transaction Statistics")
        stats = account_details.get('transaction_stats', {})
        st.write(f"**Total Transactions:** {stats.get('total_transactions', 0)}")
        st.write(f"**Sent:** {stats.get('sent_count', 0)}")
        st.write(f"**Received:** {stats.get('received_count', 0)}")
        st.write(f"**Total Sent:** {query.format_value_display(stats.get('total_sent', 0))}")
        st.write(f"**Total Received:** {query.format_value_display(stats.get('total_received', 0))}")
        
        st.markdown("---")
        st.subheader("Chain Alerts")
        chain_alerts = account_details.get('chain_alerts', [])
        if chain_alerts:
            for alert in chain_alerts:
                st.write(f"- Alert ID: {alert['alert_id']} | Chain Length: {alert['chain_length']} | Priority: {alert['priority'].upper()}")
        else:
            st.write("No chain alerts")
        
        st.markdown("---")
        st.subheader("Wallet Alerts")
        wallet_alerts = account_details.get('wallet_alerts', [])
        if wallet_alerts:
            for alert in wallet_alerts:
                st.write(f"- Alert ID: {alert['alert_id']} | Transaction Pairs: {alert['transaction_pairs']} | Priority: {alert['priority'].upper()}")
        else:
            st.write("No wallet alerts")
    except Exception as e:
        st.warning(f"Could not load account details: {e}")

def display_suspicious_patterns(patterns: Dict):
    """Display detected suspicious patterns."""
    st.subheader("Suspicious Patterns")
    st.caption(
        "Highlights addresses or values that might warrant closer inspection.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rapid Transfers (last hour)")
        rapid = patterns.get("rapid_transfers", [])
        if rapid:
            st.warning(f"Found {len(rapid)} addresses with rapid transfer patterns")
            rapid_df = pd.DataFrame(rapid[:5])
            if not rapid_df.empty:
                rapid_df["address"] = rapid_df["from_hash"].apply(format_address)
                st.dataframe(
                    rapid_df[["address", "tx_count"]],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No rapid transfer patterns detected in the last hour.")

    with col2:
        st.markdown("#### Same-Value Chains (last hour)")
        chains = patterns.get("same_value_chains", [])
        if chains:
            st.warning(f"Found {len(chains)} potential chain patterns")
            for i, chain in enumerate(chains[:3], 1):
                st.caption(f"Chain {i}: {len(chain.get('addresses', []))} addresses")
        else:
            st.info("No same-value chains detected in the last hour.")


def main():
    """Main application entry point."""

    col1, col2 = st.columns([10, 1])
    with col1:
        st.title("Mitigation of Blockchain Fraud - Blockchain Monitoring Dashboard")
        st.caption("Exploring Ethereum testnet transactions and fraud alerts.")
    with col2:
        if st.button("Refresh", use_container_width=True, help="Refresh all data"):
            st.cache_data.clear()
            st.rerun()

    # Load data - cache to prevent unnecessary reruns
    with st.spinner("Loading blockchain data..."):
        tx_df = load_transactions()
        full_alerts = load_full_alerts()
        accounts = load_accounts()
        summary = load_summary()
        patterns = load_suspicious_patterns()
    

    # Check for data
    if tx_df.empty:
        st.warning(
            "No transactions found in the database.\n\n"
            "Please ensure the data ingestion pipeline has been run:\n"
            "```bash\n"
            "python main.py --fetch 1000\n"
            "```"
        )
        st.stop()

    # Prepare data
    tx_df = tx_df.copy()
    tx_df["ValueEth"] = tx_df["ValueWei"].apply(format_wei_to_eth)
    tx_df["Date"] = tx_df["Timestamp"].dt.date
    tx_df["Hour"] = tx_df["Timestamp"].dt.hour

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range
    min_date, max_date = tx_df["Date"].min(), tx_df["Date"].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range if not isinstance(date_range, tuple) else date_range[0]

    # Value range
    max_eth = float(tx_df["ValueEth"].max())
    max_eth = max(max_eth, 0.0000001)

    value_range = st.sidebar.slider(
        "Value Range (ETH)",
        min_value=0.0,
        max_value=max_eth,
        value=(0.0, max_eth),
        format="%.6f"
    )

    # Address filter
    address_filter = st.sidebar.text_input(
        "Address Contains",
        placeholder="Enter partial address..."
    )

    # Transaction type
    tx_type = st.sidebar.selectbox(
        "Transaction Type",
        ["All", "With Value", "Zero Value"]
    )

    # Hour range
    hour_range = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    # Apply filters
    filters = {
        "date_range": (start_date, end_date),
        "value_range": value_range,
        "address": address_filter,
        "tx_type": tx_type,
    }

    filtered_df = filter_dataframe(tx_df, filters)
    filtered_df = filtered_df[
        (filtered_df["Hour"] >= hour_range[0]) &
        (filtered_df["Hour"] <= hour_range[1])
        ]

    # Display metrics
    st.markdown("## Network Overview")
    display_metrics_row(summary)

    st.markdown("## Value Metrics")
    display_value_metrics(summary)

    # Filter info
    if len(filtered_df) < len(tx_df):
        st.info(f"Showing {len(filtered_df):,} of {len(tx_df):,} transactions based on filters")

    st.markdown("---")

     # Tabs
    tabs = st.tabs(["Overview", "Analysis", "Patterns", "Alerts", "Accounts", "Transactions"])
    
    # Overview Tab
    with tabs[0]:
        st.subheader("Transaction Activity Overview")

        if filtered_df.empty:
            st.warning("No transactions match the current filters")
        else:
            # Daily volume chart
            st.markdown("#### Daily Volume (ETH)")
            fig_daily = create_daily_volume_chart(filtered_df)
            st.plotly_chart(fig_daily, use_container_width=True)

            # Hourly patterns
            st.markdown("#### Average Transaction Value by Hour")

            # Get unique dates for tabs
            unique_dates = sorted(filtered_df["Date"].unique())[-3:]

            if unique_dates:
                date_tabs = st.tabs([str(date) for date in unique_dates])

                for tab, date in zip(date_tabs, unique_dates):
                    with tab:
                        day_df = filtered_df[filtered_df["Date"] == date]
                        hourly = day_df.groupby("Hour")["ValueEth"].agg(["mean", "count"]).reset_index()

                        if not hourly.empty:
                            hourly["HourLabel"] = hourly["Hour"].apply(
                                lambda h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"
                            )

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=hourly["HourLabel"],
                                y=hourly["mean"],
                                mode="lines+markers",
                                name="Avg Value",
                                line=dict(color="#1f77b4", width=2),
                                hovertemplate="Time: %{x}<br>Avg: %{y:.6f} ETH<extra></extra>",
                            ))

                            fig.update_layout(
                                xaxis_title="Time of Day",
                                yaxis_title="Avg Value (ETH)",
                                height=350,
                                hovermode="x",
                            )

                            st.plotly_chart(fig, use_container_width=True)

    # Analysis Tab
    with tabs[1]:
        st.subheader("Value Distribution & Top Addresses")

        if filtered_df.empty:
            st.warning("No data for analysis")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Transaction Value Distribution")
                fig_dist = create_value_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)

            with col2:
                st.markdown("#### Top Addresses by Volume")
                top_addrs = load_top_addresses(n=10, by="value")
                if not top_addrs.empty:
                    top_addrs["address_display"] = top_addrs["address_short"]
                    top_addrs["value_display"] = top_addrs["total_value_eth"].apply(
                        lambda x: f"{x:.4f} ETH"
                    )

                    st.dataframe(
                        top_addrs[["address_display", "tx_count", "value_display"]].head(10),
                        column_config={
                            "address_display": "Address",
                            "tx_count": "TX Count",
                            "value_display": "Total Value"
                        },
                        hide_index=True,
                        use_container_width=True
                    )

    # Patterns Tab
    with tabs[2]:
        display_suspicious_patterns(patterns)

        # Heatmap
        st.markdown("#### Activity Heatmap")
        if not filtered_df.empty:
            fig_heat = create_hourly_heatmap(filtered_df)
            st.plotly_chart(fig_heat, use_container_width=True)

    # Alerts Tab
    with tabs[3]:
        st.subheader("Alert Summary")
        
        # Get all alerts from alert tables - REAL DATA from database tables
        # Algorithm 1: Pulls from chain_alerts table (chain detection results)
        # Algorithm 2: Pulls from wallet_alerts table (account activity alerts)
        # Algorithm 3: Pulls from time_based_alerts table (time-based activity alerts)
        all_alerts = full_alerts.copy()
        
        # Alert metrics
        cols = st.columns(4)
        with cols[0]:
            chain_count = len(all_alerts[all_alerts['algorithm'] == 'Chain Detection']) if not all_alerts.empty else 0
            st.metric("Chain Alerts", chain_count)
        with cols[1]:
            account_count = len(all_alerts[all_alerts['algorithm'] == 'Account Activity']) if not all_alerts.empty else 0
            st.metric("Account Alerts", account_count)
        with cols[2]:
            time_count = len(all_alerts[all_alerts['algorithm'] == 'Time-based Activity']) if not all_alerts.empty else 0
            st.metric("Time-based Alerts", time_count)
        with cols[3]:
            total_count = len(all_alerts) if not all_alerts.empty else 0
            st.metric("Total Alerts", total_count)
        
        st.markdown("---")
        
        if not all_alerts.empty:
            # Filters - Priority and Algorithm Type
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.subheader("Alerts Table")
            with col2:
                st.markdown("**Algorithm Filter**")
                # Initialize session state
                if 'alert_algorithm_filter' not in st.session_state:
                    st.session_state.alert_algorithm_filter = ['Chain Detection', 'Account Activity', 'Time-based Activity']
                
                # Use selectbox for algorithm filter
                algorithm_options = st.selectbox(
                    "",
                    options=['All', 'Chain Detection', 'Account Activity', 'Time-based Activity'],
                    key="algorithm_select",
                    label_visibility="collapsed"
                )
                
                # Map selection to filter list
                if algorithm_options == 'All':
                    algorithm_filter = ['Chain Detection', 'Account Activity', 'Time-based Activity']
                else:
                    algorithm_filter = [algorithm_options]
                
                st.session_state.alert_algorithm_filter = algorithm_filter
            with col3:
                st.markdown("**Priority Filter**")
                # Initialize session state
                if 'alert_priority_filter' not in st.session_state:
                    st.session_state.alert_priority_filter = ['high', 'med', 'low']
                
                # Use horizontal radio buttons
                priority_options = st.radio(
                    "",
                    options=['All', 'High', 'Medium', 'Low'],
                    horizontal=True,
                    key="priority_radio",
                    label_visibility="collapsed"
                )
                
                # Map radio selection to filter list
                if priority_options == 'All':
                    priority_filter = ['high', 'med', 'low']
                elif priority_options == 'High':
                    priority_filter = ['high']
                elif priority_options == 'Medium':
                    priority_filter = ['med']
                else:
                    priority_filter = ['low']
                
                st.session_state.alert_priority_filter = priority_filter
            
            # Filter alerts by both algorithm type and priority
            filtered_alerts = all_alerts[
                (all_alerts['algorithm'].isin(algorithm_filter)) & 
                (all_alerts['priority'].isin(priority_filter))
            ].copy()
            
            if not filtered_alerts.empty:
                # Format alerts for display
                display_alerts = filtered_alerts.copy()
                display_alerts['Alert ID'] = display_alerts['alert_id']
                display_alerts['Algorithm'] = display_alerts['algorithm']
                display_alerts['Priority'] = display_alerts['priority'].str.upper()
                # Convert timestamp to datetime
                try:
                    if display_alerts['timestamp'].dtype in ['int64', 'int32', 'float64']:
                        display_alerts['Timestamp'] = pd.to_datetime(display_alerts['timestamp'], unit='s', errors='coerce')
                    else:
                        display_alerts['Timestamp'] = pd.to_datetime(display_alerts['timestamp'], errors='coerce')
                except:
                    display_alerts['Timestamp'] = display_alerts['timestamp']
                display_alerts['Details'] = display_alerts['details']
                
                # Display table with clickable rows
                display_df = display_alerts[['Alert ID', 'Algorithm', 'Priority', 'Timestamp', 'Details']]
                
                # Configure Details column to show full text (including full hashes)
                column_config = {
                    'Details': st.column_config.TextColumn(
                        'Details',
                        width="large"  # Use large width to show full hash
                    )
                }
                
                # Use dataframe with selection support
                # Tab state is preserved via radio button key, so rerun won't reset tab
                selected_df = st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="alerts_table",
                    column_config=column_config
                )
                
                # Show details if row is selected
                if hasattr(selected_df, 'selection') and hasattr(selected_df.selection, 'rows') and selected_df.selection.rows:
                    selected_idx = selected_df.selection.rows[0]
                    if selected_idx < len(filtered_alerts):
                        selected_alert = filtered_alerts.iloc[selected_idx]
                        show_alert_details(selected_alert)
                
                # Priority summary
                st.markdown("### Priority Breakdown")
                priority_counts = filtered_alerts['priority'].value_counts()
                cols = st.columns(3)
                with cols[0]:
                    st.metric("High Priority", priority_counts.get('high', 0))
                with cols[1]:
                    st.metric("Medium Priority", priority_counts.get('med', 0))
                with cols[2]:
                    st.metric("Low Priority", priority_counts.get('low', 0))
            else:
                st.info("No alerts match the selected priority filters.")
        else:
            st.info("No alerts found in the database.")

    # Accounts Tab
    with tabs[4]:
        st.subheader("Accounts with Alerts")
        
        # Get accounts with alert counts (no priority)
        accounts_df = accounts.copy()
        
        if not accounts_df.empty:
            # Add sorting options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("")  # Spacer
            with col2:
                sort_order = st.selectbox(
                    "Sort by Alert Count",
                    options=["Most to Least", "Least to Most"],
                    key="accounts_sort_order"
                )
            
            # Sort the dataframe based on selection
            if sort_order == "Most to Least":
                accounts_df = accounts_df.sort_values('alert_count', ascending=False)
            else:
                accounts_df = accounts_df.sort_values('alert_count', ascending=True)
            
            # Format for display
            display_accounts = accounts_df.copy()
            display_accounts['Account Hash'] = display_accounts['account'].apply(lambda x: x[:16] + '...' if len(str(x)) > 16 else str(x))
            display_accounts['Alert Count'] = display_accounts['alert_count']
            
            # Display table with clickable rows
            display_df = display_accounts[['Account Hash', 'Alert Count']]
            
            # Use dataframe with selection support
            # Tab state is preserved via radio button key, so rerun won't reset tab
            selected_account_df = st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=400,
                on_select="rerun",
                selection_mode="single-row",
                key="accounts_table"
            )
            
            # Show details sidebar if row is selected
            if hasattr(selected_account_df, 'selection') and hasattr(selected_account_df.selection, 'rows') and selected_account_df.selection.rows:
                selected_idx = selected_account_df.selection.rows[0]
                if selected_idx < len(accounts_df):
                    selected_account = accounts_df.iloc[selected_idx]
                    show_account_details(selected_account)
            
            # Summary
            st.markdown("### Account Summary")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Total Accounts", len(accounts_df))
            with cols[1]:
                st.metric("Total Alerts", accounts_df['alert_count'].sum())
        else:
            st.info("No accounts with alerts found.")

    # Transactions Tab
    with tabs[5]:
        st.subheader("Transaction Details")

        if filtered_df.empty:
            st.info("No transactions to display")
        else:
            # Options
            cols = st.columns(4)
            with cols[0]:
                show_zero = st.checkbox("Show zero-value transactions", value=False)
            with cols[1]:
                rows_to_show = st.number_input("Rows to display", 100, 10000, 100, 100)
            with cols[2]:
                sort_by = st.selectbox("Sort by", ["Block", "Timestamp", "ValueEth"])
            with cols[3]:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])

            # Filter and sort
            display_df = filtered_df.copy()
            if not show_zero:
                display_df = display_df[display_df["ValueEth"] > 0]

            ascending = sort_order == "Ascending"
            display_df = display_df.sort_values(sort_by, ascending=ascending).head(rows_to_show)

            # Format for display
            display_df["From_Short"] = display_df["From"].apply(format_address)
            display_df["To_Short"] = display_df["To"].apply(format_address)
            display_df["Value_Display"] = display_df.apply(
                lambda r: query.format_value_display(r["ValueWei"]),
                axis=1
            )

            st.caption(f"Showing {len(display_df):,} transactions")

            # Display
            st.dataframe(
                display_df[["Transaction Hash", "From_Short", "To_Short", "Value_Display", "Block", "Timestamp"]],
                column_config={
                    "Transaction Hash": st.column_config.TextColumn("TX Hash", width="small"),
                    "From_Short": "From",
                    "To_Short": "To",
                    "Value_Display": "Value",
                    "Block": st.column_config.NumberColumn("Block", format="%d"),
                    "Timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss")
                },
                hide_index=True,
                use_container_width=True
            )


if __name__ == "__main__":
    main()
