from fastapi import FastAPI, HTTPException
from schema import QueryRequest, QueryResponse, SourceReference
from retrieve import ask_question, index, client, reranker_name # Import your existing logic
import uvicorn
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os 
import sys

#Initialize openai and pinecone models
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
index = pc.Index(index_name)
reranker_name = "bge-reranker-v2-m3"



app = FastAPI(
    title = "LegalBuddy API",
    description= "Bilingual RAG API for Polish and EU Law",
    version = "1.0.0"
)

@app.post("/ask", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # 1. Get the raw answer and retrieve the same data structure you built earlier
        # We slightly modify your original ask_question to return a dict for the API
        
        # Pass 1: Semantic Search
        query_vec = client.embeddings.create(
            input=request.query, 
            model='text-embedding-3-small'
        ).data[0].embedding
        
        initial_results = index.query(vector=query_vec, top_k=request.top_k, include_metadata=True)

        # Pass 1.5: Format for Reranker
        documents_to_rerank = [
            {
                "id": match["id"], 
                "text": match["metadata"]["text"],
                "source": match["metadata"].get("source", "Unknown"),
                "article": match["metadata"].get("article", "N/A")
            } 
            for match in initial_results["matches"]
        ]

        # Pass 2: Rerank
        reranked_results = pc.inference.rerank(
            model=reranker_name,
            query=request.query,
            documents=documents_to_rerank,
            top_n=5,
            return_documents=True
        )

        # Pass 3: Construct Context & Source Objects for Pydantic
        context_list = []
        source_objects = []
        
        for i, item in enumerate(reranked_results.data):
            source_name = item.document.get('source', 'Unknown')
            article_name = item.document.get('article', 'N/A')
            text_content = item.document.get('text', '')
            
            context_list.append(f"SOURCE {i+1} [{source_name} - {article_name}]:\n{text_content}")
            
            # Build the Pydantic sub-model for the response
            source_objects.append(
                SourceReference(
                    source=source_name,
                    article=article_name,
                    text=text_content,
                    score=item.score
                )
            )

        context = "\n---\n".join(context_list)
        
        # Pass 4: Generate Answer
        prompt = f"Answer strictly using context: {context}\nQuestion: {request.query}"
        
        ai_response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer_text = ai_response.choices[0].message.content

        # Return the final validated response
        return QueryResponse(
            answer=answer_text,
            sources=source_objects
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)