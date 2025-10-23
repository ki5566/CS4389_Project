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

async def query_latest_block(ApiClient : ApiClient):
    block_number = ApiClient.get_latest_block_number()
    return ApiClient.get_block_by_number(block_number)
    
def main():
    client = ApiClient()
    loop = asyncio.get_event_loop()
    block = loop.run_until_complete(query_latest_block(client))
    print(json.dumps(block, indent=4))
    
if __name__ == "__main__":
    main()
