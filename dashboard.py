import streamlit as st
import query
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

transactions = query.get_transactions()
transactions['Value (wei)'] = transactions['Value'].apply(lambda x: f'{x:.3E}') # Value formatted in scientific notation
transactions['Date'] = transactions['Timestamp'].dt.date
transactions['Hour'] = transactions['Timestamp'].dt.hour

tx_count = query.get_tx_count()
block_count = query.get_block_count()
avg_value = query.get_avg_value()

with st.container(horizontal=True):

    # For metrics
    with st.container(horizontal=True, horizontal_alignment='distribute', border=False, width=460):
        if  tx_count:
            st.metric(label='TOTAL TRANSACTIONS', value=tx_count, border=True, width='content')
        if  block_count:
            st.metric(label='TOTAL BLOCKS', value=block_count, border=True, width='content')
        st.metric(label='TOTAL ALERTS', value='10', border=True, width='content')
        if  avg_value:
            st.metric(label='AVERAGE TRANSACTION VALUE', value=f'{avg_value:.3E} wei', border=True, width='content')
        st.metric(label='SUSPICIOUS TRANSACTIONS', value='10%', border=True, width='content')
        
    # Add container for alerts table here


if not transactions.empty:
    with st.container(horizontal=True, horizontal_alignment='distribute'):
        line, pie = st.columns([0.65, 0.35], gap='large')
        
        with line:
            grouped = (
                transactions.groupby(['Date', 'Hour'])['Value']
                .mean()
                .reset_index()
                .sort_values(['Date', 'Hour'])
            )
            
            unique_dates = grouped['Date'].unique()
            chart_tabs = st.tabs([str(day) for day in unique_dates])

            for tab, day in zip(chart_tabs, unique_dates):
                with tab:
                    day_df = grouped[grouped['Date'] == day]
                    st.subheader(f"Average Transaction Value on {day}", anchor=False)
                    chart_df = day_df.set_index('Hour')['Value']
                    
                    st.line_chart(chart_df, x_label='Hour of Day', y_label='Average Value')
    
        with pie:
            st.subheader("Breakdown of Alert Types", anchor=False, divider='blue')
            # Sample data
            alert_df = pd.DataFrame({
                "Alert": ["Chain", "Account", "Spam"],
                "Count": [10, 4, 6]
            })
            fig = px.pie(alert_df, names="Alert", values="Count")
            st.plotly_chart(fig)


    columns = ['Transaction Hash', 'Value (wei)', 'From', 'To', 'Timestamp']
    st.dataframe(data=transactions[columns], hide_index=True)