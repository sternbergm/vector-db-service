from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase
import uuid
from typing import Optional

class Base(DeclarativeBase):
    pass

class Library(Base):
    __tablename__ = "libraries"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic fields
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata fields
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    
    # Relationships
    documents = relationship("Document", back_populates="library", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="library", cascade="all, delete-orphan")
    

class Document(Base):
    __tablename__ = "documents"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    library_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("libraries.id"), nullable=False, index=True)
    
    # Metadata fields
    title: Mapped[Optional[str]] = mapped_column(String(500))
    author: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    
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
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Embedding status (vectors stored in memory, not DB)
    has_embedding: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    
    # Metadata fields
    source: Mapped[Optional[str]] = mapped_column(String(500))
    sentence_number: Mapped[Optional[int]] = mapped_column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    
    # Relationships
    library = relationship("Library", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")
    