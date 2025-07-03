from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

class LibraryMetadata(BaseModel):
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = Field(default_factory=dict)

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    indexed: bool = False  # Whether library has been indexed for search
    metadata: LibraryMetadata = Field(default_factory=LibraryMetadata)

class LibraryCreate(BaseModel):
    name: str
    metadata: Optional[LibraryMetadata] = None

class LibraryUpdate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[LibraryMetadata] = None

class LibraryResponse(BaseModel):
    id: str
    name: str
    indexed: bool
    metadata: LibraryMetadata 