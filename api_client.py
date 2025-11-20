# api_client.py
import requests
from typing import Dict, Any, List, Optional

from config import ETHERSCAN_API_KEY, CHAIN_ID

BASE_URL = "https://api.etherscan.io/v2/api"


class ApiClient:
    """
    Lightweight wrapper around the Etherscan v2 API for Sepolia.
    """

    def __init__(self):
        if not ETHERSCAN_API_KEY:
            raise ValueError("API_KEY not found in environment variables")
        self.api_key = ETHERSCAN_API_KEY
        self.chainid = CHAIN_ID

    def _request(self, params: Dict[str, Any]) -> Any:
        params = {
            "chainid": self.chainid,
            "apikey": self.api_key,
            **params,
        }
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result")

    def get_latest_block_number(self) -> int:
        """
        Returns the latest block number (as an int) on the chain.
        """
        result = self._request(
            {
                "module": "proxy",
                "action": "eth_blockNumber",
            }
        )
        # result is hex string like "0x1234"
        return int(result, 16)

    def get_block_by_number(self, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Fetch block with full transactions by block number (int).
        Returns a normalized dict or None on failure.
        """
        tag = hex(block_number)  # "0x..." as required by Etherscan
        result = self._request(
            {
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": tag,
                "boolean": "true",
            }
        )

        if result is None:
            return None

        try:
            timestamp = int(result["timestamp"], 16)
            txs: List[Dict[str, Any]] = []
            for tx in result.get("transactions", []):
                txs.append(
                    {
                        "hash": tx["hash"],
                        "from": tx["from"],
                        "to": tx.get("to"),
                        "value": int(tx["value"], 16),
                        "blockNumber": int(tx["blockNumber"], 16),
                        "transactionIndex": int(tx["transactionIndex"], 16),
                        "timestamp": timestamp,
                    }
                )

            return {
                "number": int(result["number"], 16),
                "timestamp": timestamp,
                "transactions": txs,
            }
        except (KeyError, ValueError, TypeError):
            return None
