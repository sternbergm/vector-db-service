from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from database.models import Library
from schemas.library_schema import LibraryCreate, LibraryUpdate

class LibraryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, library_data: LibraryCreate) -> Library:
        """Create a new library"""
        library = Library(
            name=library_data.name,
            description=library_data.metadata.description if library_data.metadata else None,
            extra_metadata=library_data.metadata.extra if library_data.metadata else {}
        )
        self.db.add(library)
        await self.db.commit()
        await self.db.refresh(library)
        return library
    
    async def get(self, library_id: str) -> Optional[Library]:
        """Get library by ID"""
        result = await self.db.execute(
            select(Library).where(Library.id == library_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self) -> List[Library]:
        """Get all libraries"""
        result = await self.db.execute(select(Library))
        return result.scalars().all()
    
    async def update(self, library_id: str, update_data: LibraryUpdate) -> Optional[Library]:
        """Update library"""
        library = await self.get(library_id)
        if not library:
            return None
        
        # Update fields if provided
        if update_data.name is not None:
            library.name = update_data.name
        if update_data.metadata is not None:
            if update_data.metadata.description is not None:
                library.description = update_data.metadata.description
            if update_data.metadata.extra:
                library.extra_metadata = update_data.metadata.extra
        
        await self.db.commit()
        await self.db.refresh(library)
        return library
    
    async def delete(self, library_id: str) -> bool:
        """Delete library"""
        result = await self.db.execute(
            delete(Library).where(Library.id == library_id)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def exists(self, library_id: str) -> bool:
        """Check if library exists"""
        result = await self.db.execute(
            select(Library.id).where(Library.id == library_id)
        )
        return result.scalar_one_or_none() is not None
    
    
    
    async def mark_indexed(self, library_id: str) -> bool:
        """Mark library as indexed"""
        result = await self.db.execute(
            update(Library)
            .where(Library.id == library_id)
            .values(indexed=True)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def mark_unindexed(self, library_id: str) -> bool:
        """Mark library as not indexed"""
        result = await self.db.execute(
            update(Library)
            .where(Library.id == library_id)
            .values(indexed=False)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    async def get_with_relationships(self, library_id: str) -> Optional[Library]:
        """Get library with documents and chunks loaded"""
        result = await self.db.execute(
            select(Library)
            .options(selectinload(Library.documents), selectinload(Library.chunks))
            .where(Library.id == library_id)
        )
        return result.scalar_one_or_none()
    
    async def get_stats(self) -> dict:
        """Get repository statistics"""
        result = await self.db.execute(
            select(Library.indexed)
        )
        libraries = result.all()
        
        total_libraries = len(libraries)
        indexed_libraries = sum(1 for lib in libraries if lib.indexed)
        
        return {
            "total_libraries": total_libraries,
            "indexed_libraries": indexed_libraries,
        } 