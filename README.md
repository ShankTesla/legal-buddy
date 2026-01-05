# Legal Document Intelligence System

Production-ready RAG system for multilingual legal document Q&A

## Problem
Legal professionals waste hours manually searching through documents in 
multiple languages to find relevant information.

## Solution
RAG-based system that answers questions with cited sources from Polish 
and English legal documents.

## Architecture
Documents → Chunking → Pinecone → Query → 
2-Pass Retrieval → LLM → Answer with Citations

## Key Features
- **2-Pass Retrieval**: Cosine similarity search + reranking for precision
- **Multilingual**: Handles Polish and English documents
- **Source Citations**: Every answer includes document sources
- **Production Ready**: FastAPI endpoint, Docker deployment

## Tech Stack
- LangChain for RAG orchestration
- OpenAI API for embeddings & generation
- Pinecone for vector storage
- FastAPI for REST API
- Docker for containerization

## Quick Start
```bash
docker build -t legal-rag .
docker run -p 8000:8000 legal-rag
```

## API Usage
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the GDPR requirements?"}'
```

## Results
- Query latency: <2s
- Retrieval accuracy: [if you measured]
- Supports: [number] documents, [number] languages

## Production Considerations
- Pinecone for scalable vector storage
- 2-pass retrieval for quality/speed tradeoff
- Docker deployment for easy scaling

## Future Improvements
- Add query caching (Redis)
- Implement response streaming
- Add more reranking strategies
- Expand language support

## Why This Project?
Built to demonstrate ability to translate traditional ML engineering 
experience (fraud detection, MLOps) to modern LLM architectures while 
maintaining production-quality standards.
