import streamlit as st
import query

st.set_page_config(layout="wide")

transactions = query.get_transactions()
tx_count = query.get_tx_count()
block_count = query.get_block_count()
avg_value = query.get_avg_value()

if  tx_count:
    st.metric(label='TOTAL TRANSACTIONS', value=tx_count, border=True, width='content')

if  block_count:
    st.metric(label='TOTAL BLOCKS', value=block_count, border=True, width='content')

if  avg_value:
    st.metric(label='AVERAGE TRANSACTION VALUE', value=f'{avg_value} wei', border=True, width='content')

if not transactions.empty:
    st.line_chart(transactions.head(1000), x='Timestamp', y='Value (wei)', color='#32d7ed')
    
    st.dataframe(data=transactions, hide_index=True)