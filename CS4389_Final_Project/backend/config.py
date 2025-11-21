# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Database
DB_PATH = os.getenv("DB_PATH", "blockchain.db")

# Ethereum / Etherscan
ETHERSCAN_API_KEY = os.getenv("API_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID", "11155111"))  # Sepolia by default

# Ingestion
QUERY_DELAY = float(os.getenv("QUERY_DELAY", "0.25"))  # seconds between requests

# Default ingestion range (for CLI convenience)
DEFAULT_BLOCK_WINDOW = int(os.getenv("DEFAULT_BLOCK_WINDOW", "200"))
