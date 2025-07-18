from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from database.models import Library, IndexAlgorithmEnum
from schemas.library_schema import LibraryCreate, LibraryUpdate
from schemas.search_schema import IndexAlgorithm
from decorators import logger, timer

class LibraryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @logger
    @timer
    async def create(self, library_data: LibraryCreate) -> Library:
        """Create a new library"""
        # Convert schema IndexAlgorithm to database enum
        preferred_algo = IndexAlgorithmEnum.FLAT
        if library_data.preferred_index_algorithm:
            preferred_algo = IndexAlgorithmEnum(library_data.preferred_index_algorithm.value)
        
        library = Library(
            name=library_data.name,
            preferred_index_algorithm=preferred_algo,
            description=library_data.metadata.description if library_data.metadata else None,
            extra_metadata=library_data.metadata.extra if library_data.metadata else {}
        )
        self.db.add(library)
        await self.db.commit()
        await self.db.refresh(library)
        return library
    
    @logger
    @timer
    async def get(self, library_id: str) -> Optional[Library]:
        """Get library by ID"""
        result = await self.db.execute(
            select(Library).where(Library.id == library_id)
        )
        return result.scalar_one_or_none()
    
    @logger
    @timer
    async def get_all(self) -> List[Library]:
        """Get all libraries"""
        result = await self.db.execute(select(Library))
        return list(result.scalars().all())
    
    @logger
    @timer
    async def update(self, library_id: str, update_data: LibraryUpdate) -> Optional[Library]:
        """Update library"""
        library = await self.get(library_id)
        if not library:
            return None
        
        # Update fields if provided
        if update_data.name is not None:
            library.name = update_data.name
        if update_data.preferred_index_algorithm is not None:
            library.preferred_index_algorithm = IndexAlgorithmEnum(update_data.preferred_index_algorithm.value)
        if update_data.metadata is not None:
            if update_data.metadata.description is not None:
                library.description = update_data.metadata.description
            if update_data.metadata.extra:
                library.extra_metadata = update_data.metadata.extra
        
        await self.db.commit()
        await self.db.refresh(library)
        return library
    
    @logger
    @timer
    async def delete(self, library_id: str) -> bool:
        """Delete library and cascade to documents and chunks"""
        # First, get the library object to trigger ORM cascade deletions
        library = await self.get_with_relationships(library_id)
        if not library:
            return False
        
        # Delete the library object - this will trigger cascade deletions
        await self.db.delete(library)
        await self.db.commit()
        return True
    
    @logger
    @timer
    async def exists(self, library_id: str) -> bool:
        """Check if library exists"""
        result = await self.db.execute(
            select(Library.id).where(Library.id == library_id)
        )
        return result.scalar_one_or_none() is not None
    
    @logger
    @timer
    async def get_preferred_algorithm(self, library_id: str) -> Optional[IndexAlgorithm]:
        """Get the preferred index algorithm for a library"""
        result = await self.db.execute(
            select(Library.preferred_index_algorithm).where(Library.id == library_id)
        )
        algorithm_enum = result.scalar_one_or_none()
        if algorithm_enum:
            return IndexAlgorithm(algorithm_enum.value)
        return None
    
    @logger
    @timer
    async def mark_indexed(self, library_id: str) -> bool:
        """Mark library as indexed"""
        result = await self.db.execute(
            update(Library)
            .where(Library.id == library_id)
            .values(indexed=True)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    @logger
    @timer
    async def mark_unindexed(self, library_id: str) -> bool:
        """Mark library as not indexed"""
        result = await self.db.execute(
            update(Library)
            .where(Library.id == library_id)
            .values(indexed=False)
        )
        await self.db.commit()
        return result.rowcount > 0
    
    @logger
    @timer
    async def get_with_relationships(self, library_id: str) -> Optional[Library]:
        """Get library with documents and chunks loaded"""
        result = await self.db.execute(
            select(Library)
            .options(selectinload(Library.documents), selectinload(Library.chunks))
            .where(Library.id == library_id)
        )
        return result.scalar_one_or_none()
    
    @logger
    @timer
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