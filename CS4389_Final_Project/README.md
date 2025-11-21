# Mitigation of Blockchain Fraud – CS4389 Final Project

This repository is a **merged, cleaned-up version** of three development branches:

- `CS4389_Project-main`
- `CS4389_Project-add-wallet-chaining`
- `CS4389_Project-add-dos-protection`

It implements a **lightweight blockchain fraud monitoring prototype** for the Ethereum Sepolia testnet:

- Ingests blocks and transactions from Etherscan
- Stores them in a local SQLite DB (`blockchain.db`)
- Runs rule-based fraud detection:
  - Same-value transaction chains
  - DoS / rapid-transaction–style behavior
- Visualizes results in a Streamlit dashboard

The project is split into two tiers:

- `backend/` – ingestion, database, detection rules, and optional FastAPI API
- `dashboard/` – Streamlit dashboard that reads from `blockchain.db`

You will need a `.env` file with at least:

```env
API_KEY=YOUR_ETHERSCAN_API_KEY
DB_PATH=blockchain.db
CHAIN_ID=11155111
QUERY_DELAY=0.25
DEFAULT_BLOCK_WINDOW=200
```

See `docs/` for Deliverable 2 write-up skeleton, API notes, and rule descriptions.
