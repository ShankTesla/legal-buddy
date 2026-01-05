from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The legal question to ask")
    
    # FIX: Change the first '=' to a ':'
    top_k: Optional[int] = Field(default=5, description="Number of initial chunks to retrieve")

class SourceReference(BaseModel):
    source: str
    article: str
    text: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    model_used: str = "gpt-5-nano"