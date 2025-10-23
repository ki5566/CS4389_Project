import requests
import json
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()

class ApiClient:
    def __init__(self):
        self.base_url = "https://api.etherscan.io/v2/api"
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables")
        
        self.chainid = 11155111

    def get_latest_block_number(self):
        # curl "https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_blockNumber&apikey=YourApiKeyToken"
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None

    def get_block_by_number(self, block_number):
        response = requests.get(self.base_url, params={
            "chainid": self.chainid,
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": block_number,
            "boolean": "true",
            "apikey": self.api_key
        })
        if response.status_code == 200:
            data = response.json()
            return data.get("result")
        return None
    
    def get_transactions_by_block(self, block_number: int):
        block = self.get_block_by_number(block_number)
        if block:
            raw_transactions = block["transactions"]
            transactions = [{
                "blockNumber": transaction["blockNumber"],
                "hash": transaction["hash"],
                "from": transaction["from"],
                "to": transaction["to"],
                "value": transaction["value"],
                "index": i
            } for i, transaction in enumerate(raw_transactions) if int(transaction["value"], 16) > 0]
            return transactions
        return []

async def query_latest_block(ApiClient : ApiClient):
    block_number = ApiClient.get_latest_block_number()
    return ApiClient.get_block_by_number(block_number)

async def query_latest_n_blocks(ApiClient : ApiClient, n: int):
    block_number_hex = ApiClient.get_latest_block_number()
    if os.path.exists("./data/transactions.json"):
        with open("./data/transactions.json", "r") as json_file:
            data = json.load(json_file)
    else:
        data = {}
    transactions = data["transactions"] if "transactions" in data else []
    blocks = data["blocks"] if "blocks" in data else []
    skipcount = 0
    for i in range(n):
        block_num_hex = hex(int(block_number_hex, 16) - i - skipcount)
        while(block_num_hex in blocks):
            print(f"Block {block_num_hex} already fetched, skipping...")
            skipcount += 1
            block_num_hex = hex(int(block_number_hex, 16) - i - skipcount)
        block_transactions = ApiClient.get_transactions_by_block(block_num_hex)
        print(f"Fetched {len(block_transactions)} transactions from block {block_num_hex}")
        transactions.extend(block_transactions)
        blocks.append(block_num_hex)
    return { "transactions": transactions, "blocks": blocks }

def main():
    client = ApiClient()
    data = asyncio.run(query_latest_n_blocks(client, 2))
    open("./data/transactions.json", "w").write(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()
