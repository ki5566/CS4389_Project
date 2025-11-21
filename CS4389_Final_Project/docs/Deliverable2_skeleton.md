# CS4389 Data and Applications Security  
## Final Project – Deliverable 2  

**Title:** Mitigation of Blockchain Fraud  
**Team:** Aris Papavassiliou, Luke Liu, Azlaan Shafi, Linh Tran, Megha Nettem, Jose Rodas, Vikas Thoutam  

---

## 1. Group Information and Task Delegation

> Fill this table in your final document – this is just a skeleton.

| Tasks | Member(s) |
|-------|-----------|
| Proposal and team delegation | All Members |
| Deliverable report formatting and writing | All Members |
| Fetched data from Sepolia Testnet using Etherscan API | Luke, Vikas |
| Created SQLite database with tables for transactions, blocks, and wallets and inserted data | Luke |
| Created algorithm for first working rule: same-value transaction chain detection | Vikas |
| Dashboard design and early component implementation (Figma + Streamlit) | Linh |
| Created laundering-like wallet detection logic | Vikas |
| Final frontend dashboard (alerts + accounts views) | Jose, Megha |
| Alerts & Account ID tables, priority ordering/filtering | Megha |
| Routing and UX troubleshooting | Megha |

You can expand this section with more detailed bullet points per person.

---

## 2. Introduction

We are building a lightweight blockchain transaction-monitoring prototype that listens to the Ethereum Sepolia testnet, stores transactions in a SQLite3 database, and applies rule-based fraud detection. The primary detection rules in this prototype are:

1. **Same-value transaction chain detection** – finds chains of transactions of the same value that occur across addresses within a configurable time window. This is a fast heuristic for detecting possible money-laundering (layering/peeling) activity.
2. **Suspicious wallet / laundering-like behavior detection** – identifies wallets where there are many incoming transactions with the same (or very similar) value as outgoing transactions within a short time window, suggesting the wallet is acting as a relay in a laundering chain.
3. **DoS / rapid-transaction–style behavior detection** – flags addresses that send a very high volume of transactions in short time buckets, which can correspond to spam, bot-driven attacks, or abuse-like activity.

The system ingests blocks and transactions from Sepolia via the Etherscan API, persists them in SQLite, runs these rule-based detectors over the stored data, and surfaces the findings through a dashboard and optional REST API. We position this project as a compact, transparent, and explainable complement to industrial-grade blockchain analytics platforms.

---

## 3. Background and Related Work

(Write this in your own words – this is just a guide.)

- Brief history of blockchain analytics and fraud detection (e.g., early manual tracing, emergence of Chainalysis, Elliptic, CipherTrace, etc.).
- High-level description of common laundering patterns: peeling chains, tumblers, mixers, layering, and rapid relay through throwaway wallets.
- Describe how commercial tools often:
  - Build address/cluster graphs.
  - Track flows across time and services.
  - Use ML models to identify suspicious patterns.

Then relate your implementation:

- You implement **transparent, rule-based detection**:
  - Same-value chains as a proxy for layering.
  - Wallet relay behavior as laundering-like behavior.
  - DoS-like activity as high-velocity spam/attack signal.
- Explain how your system differs:
  - Works on a small testnet dataset.
  - Uses a simple relational DB instead of full graph DB.
  - Emphasizes explainability and educational value over black-box ML.

Remember to include numbered IEEE-style citations in this section.

---

## 4. Implementation

### 4.1 Platforms, Tools, and Technologies

- **OS / Platforms:** Any OS that supports Python 3 (Windows, macOS, Linux).
- **Programming Languages:** Python 3, SQL (SQLite).
- **IDE / Tools:** VS Code (or any preferred editor).
- **Security Libraries / Frameworks:**
  - `fastapi` for the backend API (optional, but included in this merged repo).
  - `streamlit` for secure, quick dashboard prototyping.
  - `python-dotenv` for keeping API keys out of source code.
- **DBMS:** SQLite (file-based, `blockchain.db`).
- **Blockchain:** Ethereum Sepolia testnet, accessed via the Etherscan v2 API.
- **Others:** `requests`, `plotly`, `pandas`, etc.

### 4.2 System Architecture

Explain your two-tier design (you can paste or redraw the figure):

- **Backend / Ingestion & Detection Tier:**
  - `api_client.py` – wraps Etherscan v2 API and normalizes block / transaction data.
  - `ingestion.py` – iterates over block ranges and inserts blocks + transactions into SQLite.
  - `db_utils.py` – creates tables (`blocks`, `transactions`, `alerts`, etc.) and manages connections.
  - `detection.py` – implements:
    - `find_transaction_chain(...)` – same-value chain detection.
    - `detect_dos_like_activity(...)` – DoS-style high-volume detection.
  - `cli.py` – command-line entry point for ingestion and offline rule runs.
  - `api_server.py` – FastAPI app exposing REST endpoints for health, stats, transactions, and alerts.

- **Dashboard Tier:**
  - `dashboard/dashboard.py` – Streamlit UI.
  - `dashboard/query.py` – helper functions for querying SQLite for charts, tables, and metrics.

You can describe how the data flows:

> Sepolia (Etherscan) → `ApiClient` → `ingestion.fetch_and_store_blocks` → `blockchain.db` → `detection.*` → alerts → Streamlit dashboard / FastAPI.

### 4.3 Security Principles Addressed

For each of the main security principles, explain briefly how the project addresses it conceptually:

- **Confidentiality:** While blockchain data is public, API keys and DB paths are kept in `.env`. If authentication is added to the dashboard/API, you can discuss how it would restrict who can see alerts and analytics.
- **Integrity:** The system reads from an append-only ledger (blockchain), and SQLite constraints (primary keys, foreign keys) help prevent accidental data corruption. You can mention checks on duplicate blocks and hashes.
- **Authentication (Origin Integrity):** Blocks and transactions are assumed valid because they come from Sepolia’s consensus; discuss how, in production, node signatures and RPC authentication would matter.
- **Availability:** Local SQLite storage and a lightweight dashboard mean the system is easy to spin up, with minimal dependencies. You can also mention that detection is offline and does not impact blockchain availability.
- **Authorization:** Simple role separation can be argued: only users with the `.env` and ingestion scripts can modify the dataset; dashboard viewers only read.
- **Non-repudiation:** Once alerts are logged with transaction hashes, it is difficult for a party to deny that a given transaction occurred, since hashes correspond to immutable on-chain records.

Fill this section out with more detail in your own words.

---

## 4.4 Results

Here you will:

- Show screenshots of:
  - The Streamlit dashboard (overview, alerts table, wallet/transaction drill-down).
  - Example same-value chains caught by the rule.
  - Example DoS-style patterns (very high tx count from one address in a short window).
- Provide simple metrics:
  - Number of transactions processed.
  - Number of alerts generated per rule.
  - Observed false positives / limitations.
- Compare qualitatively with tools like Etherscan’s basic displays:
  - Etherscan can show tx history and raw data.
  - Your tool layers targeted alerting and chaining logic on top.

---

## 5. Conclusion and Future Work

Summarize:

- What you implemented:
  - End-to-end ingestion, storage, and rule-based detection.
  - Dashboard and API.
- What you learned about:
  - Blockchain data structure and constraints (timestamp granularity, gas, etc.).
  - How non-ML, rule-based systems can still yield useful insights.

Future work ideas:

- Add clustering of addresses and graph-based path search.
- Add more fraud scenarios (mixers, Ponzi-like schemes, abnormal gas usage, etc.).
- Incorporate ML / anomaly detection on top of rule outputs.
- Harden the system (authentication, role-based access, Dockerization, etc.).

---

## 6. References

Remember to format your references in **IEEE style** and cite them with numbers like [1], [2] in the text.

Example (do NOT copy verbatim, just a pattern):

1. S. Nakamoto, “Bitcoin: A Peer-to-Peer Electronic Cash System,” 2008.  
2. Chainalysis, “2023 Crypto Crime Report,” 2023.  
3. G. Wood, “Ethereum: A Secure Decentralised Generalised Transaction Ledger,” 2014.

Replace with real papers, articles, or documentation you actually used.
