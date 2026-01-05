from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os 
import sys
from fastapi import FastAPI

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

def ask_question(query):

    #create query vector to do semantic search using cosine similarity
    query_vec = client.embeddings.create(input=query, model = 'text-embedding-3-small').data[0].embedding

    #pass 1 = semantic search with cosine similarity
    initial_results = index.query(vector=query_vec, top_k=50, include_metadata=True)

    #pass 1.5 = format documents for the Reranker
    # The Reranker needs to know the text it's actually reranking
    documents_to_rerank = [
        {
            "id": match["id"],
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "Unknown Source"),
            "article": match["metadata"].get("article", "N/A"),
            "lang": match["metadata"].get("lang", "en")
        }
        for match in initial_results["matches"]
    ]


    #pass 2 = reranking using opensource bge
    reranked_results = pc.inference.rerank(
                        model=reranker_name,
                        query=query,
                        documents=documents_to_rerank,
                        top_n=5,
                        return_documents=True
                        )
    

    #pass 3 = construct a "numbered" context for including metadata and sources
    context_list = []
    references = []
    
    for i, item in enumerate(reranked_results.data):
        # use document field to get metadata from pass 2
        source = item.document.get('source', 'Unknown Source')
        article = item.document.get('article', 'N/A')
        text = item.document.get('text', '')

        #append to context list
        context_list.append(f"SOURCE {i+1} [{source} - {article}]:\n{text}")

        # Store for our own display
        references.append(f"[{i+1}] {source} - {article}")

    #join contents
    context = "\n---\n".join(context_list)

    #prompt
    prompt = f"""
                SYSTEM: You are a strict Legal Assistant. Answer the question using ONLY the provided sources. 
                If the answer isn't there, say you don't know. 
                CRITICAL: For every claim, you MUST cite the source number in square brackets, e.g., [1] or [2].
                
                CONTEXT:
                {context}

                QUESTION: {query}
                """

    # answer generator using gpt-5 nano
    answer = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]

    )

    print("\n--- ANSWER ---")
    print(answer.choices[0].message.content)
    print("\n--- SOURCES USED ---")
    for ref in references:
        print(ref)
    return answer.choices[0].message.content

if __name__ == "__main__":
    #ask_question("Kto ponosi ciężar udowodnienia faktu według Kodeksu cywilnego?")
    #ask_question("What does Article 5 of the AI Act say about emotion recognition in schools?")
    #ask_question("Jakie są zasady przejrzystości dla systemów AI ogólnego przeznaczenia?")
    #ask_question("At what age does a person gain 'ograniczona zdolność do czynności prawnych' in Poland?")
    #ask_question("I am 14 years old and I bought a very expensive phone in Warsaw. Is this contract valid if my parents didn't sign it?")
    ask_question("Can an employer in Poland use AI to check if employees are happy or sad at their desks?")