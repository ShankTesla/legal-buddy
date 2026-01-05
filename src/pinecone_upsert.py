import uuid
from pinecone import Pinecone
from openai import OpenAI
from preprocess import preprocess
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec # ADD THIS IMPORT
import time


#Initialize Clients
# 1. Force load the .env file
loaded = load_dotenv() 

# 2. Check if the file was found
print(f"Did Python find the .env file? {loaded}")
pinecone_api = os.getenv("PINECONE_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key = pinecone_api)
client = OpenAI(api_key=openai_api)

index_name = "legal-intelligence"

# 1. check if index exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Index '{index_name}' not found. Creating it now...")
    
    # 2. Create the index
    pc.create_index(
        name=index_name,
        dimension=1536, # Must match text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1" # This is the default for free tier
        )
    )
    
    # 3. Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists. Connecting...")

# 4. Now connect to the index
index = pc.Index(index_name)

def upsert(md_splits):
    for filename, documents in md_splits.items():
        print(f"Processing {filename}...")


        #batching upsets
        vectors_to_upsert = []

        for i, doc in enumerate(documents):
            #generating embedding
            response = client.embeddings.create(
                input = doc.page_content,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            #deterministic id
            chunk_id = f"{filename}_{i}" 

            # Metadata prep
            metadata = {
                "text": doc.page_content,
                "source": filename,
                "lang": doc.metadata.get("lang", "en"),
                "article": doc.metadata.get("article", "Unknown") # <-- Use the key from preprocess
            }

            # Creating the vector obj
            vectors_to_upsert.append({
                "id": chunk_id, #this gives unique ID for each chunk
                "values": embedding,
                "metadata": metadata
            })
        

        # Now put the batch into the Pinecone
        index.upsert(vectors=vectors_to_upsert)


if __name__ == "__main__":
    md_splits = preprocess("./data/markdown")
    upsert(md_splits)