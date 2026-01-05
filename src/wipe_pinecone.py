
from pinecone import Pinecone
import os
import sys
from dotenv import load_dotenv

loaded = load_dotenv() 

# 2. Check if the file was found
print(f"Did Python find the .env file? {loaded}")
pinecone_api = os.getenv("PINECONE_API")
pc = Pinecone(api_key = pinecone_api)
index_name = "legal-intelligence"
index = pc.Index(index_name)
def wipe():
    index.delete(delete_all=True)

if __name__ == "__main__":
    wipe()