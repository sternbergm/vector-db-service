from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload
from database.models import Chunk
from schemas.chunk_schema import ChunkCreate, ChunkUpdate

class ChunkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, chunk_data: ChunkCreate) -> Chunk:
        """Create a new chunk"""
        chunk = Chunk(
            text=chunk_data.text,
            library_id=chunk_data.library_id,
            document_id=chunk_data.document_id,
            source=chunk_data.metadata.source if chunk_data.metadata else None,
            sentence_number=chunk_data.metadata.sentence_number if chunk_data.metadata else None,
            extra_metadata=chunk_data.metadata.extra if chunk_data.metadata else {}
        )
        self.db.add(chunk)
        await self.db.commit()
        await self.db.refresh(chunk)
        return chunk
    
    async def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        result = await self.db.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_library(self, library_id: str) -> List[Chunk]:
        """Get all chunks in a library"""
        result = await self.db.execute(
            select(Chunk).where(Chunk.library_id == library_id)
        )
        return result.scalars().all()
    
    async def get_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks in a document"""
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        return result.scalars().all()
    
    async def get_unindexed_chunks(self, library_id: str) -> List[Chunk]:
        """Get chunks in library that don't have embeddings yet"""
        result = await self.db.execute(
            select(Chunk).where(
                and_(
                    Chunk.library_id == library_id,
                    Chunk.has_embedding == False
                )
            )
        )
        return result.scalars().all()
    
    async def get_indexed_chunks(self, library_id: str) -> List[Chunk]:
        """Get chunks in library that have embeddings"""
        result = await self.db.execute(
            select(Chunk).where(
                and_(
                    Chunk.library_id == library_id,
                    Chunk.has_embedding == True
                )
            )
        )
        return result.scalars().all()
    
    async def update(self, chunk_id: str, update_data: ChunkUpdate) -> Optional[Chunk]:
        """Update chunk"""
        chunk = await self.get(chunk_id)
        if not chunk:
            return None
        
        # Update fields if provided
        if update_data.text is not None:
            chunk.text = update_data.text
        
        if update_data.metadata is not None:
            if update_data.metadata.source is not None:
                chunk.source = update_data.metadata.source
            if update_data.metadata.sentence_number is not None:
                chunk.sentence_number = update_data.metadata.sentence_number
            if update_data.metadata.extra:
                chunk.extra_metadata = update_data.metadata.extra
        
        await self.db.commit()
        await self.db.refresh(chunk)
        return chunk
    
    async def update_embedding(self, chunk_id: str, embedding_model: str) -> bool:
        """Update chunk embedding status"""
        result = await self.db.execute(
            update(Chunk)
            .where(Chunk.id == chunk_id)
            .values(has_embedding=True)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def delete(self, chunk_id: str) -> bool:
        """Delete chunk"""
        result = await self.db.execute(
            delete(Chunk).where(Chunk.id == chunk_id)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def delete_by_library(self, library_id: str) -> int:
        """Delete all chunks in a library. Returns count of deleted chunks."""
        result = await self.db.execute(
            delete(Chunk).where(Chunk.library_id == library_id)
        )
        await self.db.commit()
        return result.rowcount
    
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks in a document. Returns count of deleted chunks."""
        result = await self.db.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        await self.db.commit()
        return result.rowcount
    
    async def exists(self, chunk_id: str) -> bool:
        """Check if chunk exists"""
        result = await self.db.execute(
            select(Chunk.id).where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none() is not None
    
    async def count_by_library(self, library_id: str) -> int:
        """Count chunks in library"""
        result = await self.db.execute(
            select(Chunk.id).where(Chunk.library_id == library_id)
        )
        return len(result.scalars().all())
    
    async def count_by_document(self, document_id: str) -> int:
        """Count chunks in document"""
        result = await self.db.execute(
            select(Chunk.id).where(Chunk.document_id == document_id)
        )
        return len(result.scalars().all())
    
    async def get_library_chunk_ids(self, library_id: str) -> List[str]:
        """Get list of chunk IDs for a library"""
        result = await self.db.execute(
            select(Chunk.id).where(Chunk.library_id == library_id)
        )
        return [str(chunk_id) for chunk_id in result.scalars().all()]
    
    async def get_document_chunk_ids(self, document_id: str) -> List[str]:
        """Get list of chunk IDs for a document"""
        result = await self.db.execute(
            select(Chunk.id).where(Chunk.document_id == document_id)
        )
        return [str(chunk_id) for chunk_id in result.scalars().all()]
    
    async def filter_chunks(self, library_id: str, filters: Dict[str, Any]) -> List[Chunk]:
        """Filter chunks by metadata"""
        query = select(Chunk).where(Chunk.library_id == library_id)
        
        # Apply filters
        for key, value in filters.items():
            if key == "source":
                query = query.where(Chunk.source == value)
            elif key == "sentence_number":
                query = query.where(Chunk.sentence_number == value)
            # For extra_metadata, you'd need to use JSON operations
            # This is a simplified version
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_with_relationships(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk with library and document relationships loaded"""
        result = await self.db.execute(
            select(Chunk)
            .options(selectinload(Chunk.library), selectinload(Chunk.document))
            .where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def get_stats(self) -> Dict[str, int]:
        """Get repository statistics"""
        result = await self.db.execute(
            select(Chunk.has_embedding, Chunk.library_id, Chunk.document_id)
        )
        chunks = result.all()
        
        total_chunks = len(chunks)
        indexed_chunks = sum(1 for chunk in chunks if chunk.has_embedding)
        libraries_with_chunks = len(set(chunk.library_id for chunk in chunks))
        documents_with_chunks = len(set(chunk.document_id for chunk in chunks if chunk.document_id))
        
        return {
            "total_chunks": total_chunks,
            "indexed_chunks": indexed_chunks,
            "unindexed_chunks": total_chunks - indexed_chunks,
            "libraries_with_chunks": libraries_with_chunks,
            "documents_with_chunks": documents_with_chunks
        } 