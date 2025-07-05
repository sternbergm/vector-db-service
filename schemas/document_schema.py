from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    library_id: str
    chunk_ids: List[str] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

class DocumentCreate(BaseModel):
    library_id: str
    metadata: Optional[DocumentMetadata] = None

class DocumentCreateFromChunks(BaseModel):
    chunk_texts: List[str] = Field(..., min_items=1, description="List of text content for chunks")
    document_metadata: Optional[DocumentMetadata] = None
    chunk_source: Optional[str] = None

class DocumentResponse(BaseModel):
    document_id: str
    library_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = 0

class DocumentSummary(BaseModel):
    document_id: str
    library_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0

# Note: No DocumentUpdate models
# Documents are auto-created and managed through chunk operations 