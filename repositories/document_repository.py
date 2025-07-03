from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload
from database.models import Document
from schemas.document_schema import DocumentMetadata
from decorators import logger, timer
class DocumentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @logger
    @timer
    async def create_or_get(self, library_id: str, metadata: DocumentMetadata = None) -> Document:
        """Create a new document or get existing one based on metadata similarity"""
        # For auto-management, we could group by title if provided
        if metadata and metadata.title:
            existing = await self._find_by_title(library_id, metadata.title)
            if existing:
                return existing
        
        # Create new document
        document = Document(
            library_id=library_id,
            title=metadata.title if metadata else None,
            author=metadata.author if metadata else None
        )
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)
        return document
    
    @logger
    @timer
    async def get(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    @logger
    @timer
    async def get_by_library(self, library_id: str) -> List[Document]:
        """Get all documents in a library"""
        result = await self.db.execute(
            select(Document).where(Document.library_id == library_id)
        )
        return result.scalars().all()
    
    
    @logger
    @timer
    async def delete(self, document_id: str) -> bool:
        """Delete document"""
        result = await self.db.execute(
            delete(Document).where(Document.id == document_id)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    @logger
    @timer
    async def delete_by_library(self, library_id: str) -> int:
        """Delete all documents in a library. Returns count of deleted documents."""
        result = await self.db.execute(
            delete(Document).where(Document.library_id == library_id)
        )
        await self.db.commit()
        return result.rowcount
    
    @logger
    @timer
    async def count_by_library(self, library_id: str) -> int:
        """Count documents in library"""
        result = await self.db.execute(
            select(Document.id).where(Document.library_id == library_id)
        )
        return len(result.scalars().all())
    
    @logger
    @timer
    async def _find_by_title(self, library_id: str, title: str) -> Optional[Document]:
        """Find document by title within library"""
        result = await self.db.execute(
            select(Document).where(
                and_(
                    Document.library_id == library_id,
                    Document.title == title
                )
            )
        )
        return result.scalar_one_or_none()
    
    @logger
    @timer
    async def get_with_relationships(self, document_id: str) -> Optional[Document]:
        """Get document with library and chunks relationships loaded"""
        result = await self.db.execute(
            select(Document)
            .options(selectinload(Document.library), selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    @logger
    @timer
    async def update_metadata(self, document_id: str, metadata: DocumentMetadata) -> bool:
        """Update document metadata"""
        update_values = {}
        if metadata.title is not None:
            update_values["title"] = metadata.title
        if metadata.author is not None:
            update_values["author"] = metadata.author
        if metadata.extra:
            update_values["extra_metadata"] = metadata.extra
        
        if update_values:
            result = await self.db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(**update_values)
            )
            await self.db.commit()
            return result.rowcount > 0
        return False
    
    @logger
    @timer
    async def get_stats(self) -> Dict[str, int]:
        """Get repository statistics"""
        result = await self.db.execute(
            select(Document.library_id)
        )
        documents = result.all()
        
        total_documents = len(documents)
        libraries_with_documents = len(set(doc.library_id for doc in documents))
        
        return {
            "total_documents": total_documents,
            "libraries_with_documents": libraries_with_documents
        } 