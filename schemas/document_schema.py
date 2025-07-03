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

# Note: No DocumentCreate, DocumentUpdate, DocumentResponse models
# Documents are auto-created and managed through chunk operations 