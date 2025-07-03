from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

class ChunkMetadata(BaseModel):
    source: Optional[str] = None
    sentence_number: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: Optional[List[float]] = None  # Optional since embedding is generated
    document_id: Optional[str] = None  # Reference to parent document
    library_id: str  # Direct reference for efficient searching
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

class ChunkCreate(BaseModel):
    text: str
    document_id: Optional[str] = None
    library_id: str
    metadata: Optional[ChunkMetadata] = None

class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[ChunkMetadata] = None

class ChunkResponse(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None
    document_id: Optional[str] = None
    library_id: str
    metadata: ChunkMetadata
    similarity_score: Optional[float] = None  # For search results 