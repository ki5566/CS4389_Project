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
    page_title="☄️ Mitigation of Blockchain Fraud - Blockchain Monitoring Dashboard",
    page_icon="☄️",
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
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading blockchain data...", ttl=30)
def load_transactions() -> pd.DataFrame:
    """Load transactions with caching."""
    return query.get_transactions()


@st.cache_data(show_spinner=False, ttl=30)
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


@st.cache_data(show_spinner=False, ttl=30)
def load_hourly_stats(date: Optional[datetime] = None) -> pd.DataFrame:
    """Load hourly transaction statistics."""
    return query.get_hourly_statistics(date)


@st.cache_data(show_spinner=False, ttl=60)
def load_top_addresses(n: int = 10, by: str = "value") -> pd.DataFrame:
    """Load top addresses by volume or count."""
    return query.get_top_addresses(n=n, by=by, address_type="both")


@st.cache_data(show_spinner=False, ttl=120)
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
        title_text="Daily Volume (ETH)",
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
        title="Transaction Value Distribution",
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
            "🧱 Blocks",
            format_large_number(summary["block_count"])
        )

    with cols[1]:
        st.metric(
            "💸 Transactions",
            format_large_number(summary["tx_count"])
        )

    with cols[2]:
        st.metric(
            "💼 Wallets",
            format_large_number(summary["wallet_count"])
        )

    with cols[3]:
        active = summary.get("active_wallets", 0)
        total = max(summary["wallet_count"], 1)
        st.metric(
            "🔥 Active (24h)",
            format_large_number(active),
            delta=f"{(active / total * 100):.1f}%" if active > 0 else None
        )

    with cols[4]:
        velocity = summary.get("network_velocity", {})
        tx_per_hour = velocity.get("tx_per_hour", 0)
        st.metric(
            "⚡ TX / Hour",
            format_large_number(tx_per_hour) if tx_per_hour > 0 else "0"
        )

    with cols[5]:
        alerts = summary["total_alerts"]
        st.metric(
            "🚨 Total Alerts",
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
            "💰 Avg Value",
            f"{avg_eth:.6f} ETH" if avg_eth > 0 else "0 ETH"
        )

    with cols[1]:
        median_eth = stats.get("median_eth", 0)
        st.metric(
            "📊 Median Value",
            f"{median_eth:.6f} ETH" if median_eth > 0 else "0 ETH"
        )

    with cols[2]:
        min_eth = stats.get("min_eth", 0)
        st.metric(
            "⬇️ Min Value",
            f"{min_eth:.6f} ETH" if min_eth > 0 else "0 ETH"
        )

    with cols[3]:
        max_eth = stats.get("max_eth", 0)
        st.metric(
            "⬆️ Max Value",
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


def display_suspicious_patterns(patterns: Dict):
    """Display detected suspicious patterns."""
    st.subheader("Suspicious Patterns")
    st.caption(
        "Highlights addresses or values that might warrant closer inspection.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Rapid Transfers (last hour) 🔄")
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
        st.title("☄️ Mitigation of Blockchain Fraud - Blockchain Monitoring Dashboard")
        st.caption("Exploring Ethereum testnet transactions and fraud alerts.")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Refresh all data"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    with st.spinner("Loading blockchain data..."):
        tx_df = load_transactions()
        summary = load_summary()
        patterns = load_suspicious_patterns()

    # Check for data
    if tx_df.empty:
        st.warning(
            "⚠️ No transactions found in the database.\n\n"
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
    st.sidebar.header("🔍 Filters")

    # Date range
    min_date, max_date = tx_df["Date"].min(), tx_df["Date"].max()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
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
        "💰 Value Range (ETH)",
        min_value=0.0,
        max_value=max_eth,
        value=(0.0, max_eth),
        format="%.6f"
    )

    # Address filter
    address_filter = st.sidebar.text_input(
        "🔎 Address Contains",
        placeholder="Enter partial address..."
    )

    # Transaction type
    tx_type = st.sidebar.selectbox(
        "📂 Transaction Type",
        ["All", "With Value", "Zero Value"]
    )

    # Hour range
    hour_range = st.sidebar.slider(
        "⏱ Hour of Day",
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
    st.markdown("## 📊 Network Overview")
    display_metrics_row(summary)

    st.markdown("## 💎 Value Metrics")
    display_value_metrics(summary)

    # Filter info
    if len(filtered_df) < len(tx_df):
        st.info(f"🔍 Showing {len(filtered_df):,} of {len(tx_df):,} transactions based on filters")

    st.markdown("---")

    # Tabs
    tabs = st.tabs(["📈 Overview", "📊 Analysis", "🔬 Patterns", "🚨 Alerts", "📋 Transactions"])

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
                                title=f"Average Transaction Value by Hour - {date}",
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

        alert_summary = summary["alert_summary"]
        if summary["total_alerts"] == 0:
            st.info("No alerts have been generated yet.")
        else:
            # Alert metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric("Transaction Alerts", alert_summary["transaction_alerts"])
            with cols[1]:
                st.metric("Account Alerts", alert_summary["account_alerts"])
            with cols[2]:
                st.metric("Total Alerts", summary["total_alerts"])

    # Transactions Tab
    with tabs[4]:
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
