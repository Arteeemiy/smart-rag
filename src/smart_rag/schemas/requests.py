from pydantic import BaseModel, Field


class RAGAskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    top_k: int | None = Field(default=5, ge=1, le=20)
    debug: bool = False


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=5, ge=1, le=20)
    debug: bool = True
