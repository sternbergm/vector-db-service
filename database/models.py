from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase
import uuid

class Base(DeclarativeBase):
    pass

class Library(Base):
    __tablename__ = "libraries"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic fields
    name = Column(String(255), nullable=False, index=True)
    indexed = Column(Boolean, default=False, nullable=False)
    
    # Metadata fields
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata = Column(JSON, default=dict)
    
    # Relationships
    documents = relationship("Document", back_populates="library", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="library", cascade="all, delete-orphan")
    

class Document(Base):
    __tablename__ = "documents"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    library_id = Column(UUID(as_uuid=True), ForeignKey("libraries.id"), nullable=False, index=True)
    
    # Metadata fields
    title = Column(String(500))
    author = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata = Column(JSON, default=dict)
    
    # Relationships
    library = relationship("Library", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    

class Chunk(Base):
    __tablename__ = "chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    library_id = Column(UUID(as_uuid=True), ForeignKey("libraries.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    
    # Content
    text = Column(Text, nullable=False)
    
    # Embedding status (vectors stored in memory, not DB)
    has_embedding = Column(Boolean, default=False, nullable=False, index=True)
    
    # Metadata fields
    source = Column(String(500))
    sentence_number = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata = Column(JSON, default=dict)
    
    # Relationships
    library = relationship("Library", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")
    