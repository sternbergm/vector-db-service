from typing import List, Optional
from repositories import LibraryRepository
from schemas.library_schema import LibraryCreate, LibraryUpdate, LibraryResponse, LibraryMetadata
from exceptions import LibraryNotFoundError, DatabaseError
from decorators import logger
class LibraryService:
    def __init__(self, library_repository: LibraryRepository):
        self.library_repository = library_repository

    @logger
    async def create_library(self, data: LibraryCreate) -> LibraryResponse:
        """Create a new library"""
        try:
            # Validate and create library
            library = await self.library_repository.create(data)
            
            return LibraryResponse(
                id=str(library.id),
                name=library.name,
                indexed=library.indexed,
                metadata=LibraryMetadata(
                    description=library.description,
                    created_at=library.created_at,
                    updated_at=library.updated_at,
                    extra=library.extra_metadata or {}
                ),
            )
        except Exception as e:
            raise DatabaseError(f"Failed to create library: {str(e)}")

    @logger
    async def get_library(self, library_id: str) -> LibraryResponse:
        """Get library by ID"""
        try:
            library = await self.library_repository.get(library_id)
            if not library:
                raise LibraryNotFoundError(library_id)
                
            return LibraryResponse(
                id=str(library.id),
                name=library.name,
                indexed=library.indexed,
                metadata=LibraryMetadata(
                    description=library.description,
                    created_at=library.created_at,
                    updated_at=library.updated_at,
                    extra=library.extra_metadata or {}
                ),
            )
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to fetch library: {str(e)}")

    @logger
    async def get_all_libraries(self) -> List[LibraryResponse]:
        """Get all libraries"""
        try:
            libraries = await self.library_repository.get_all()
            
            return [
                LibraryResponse(
                    id=str(library.id),
                    name=library.name,
                    indexed=library.indexed,
                    metadata=LibraryMetadata(
                        description=library.description,
                        created_at=library.created_at,
                        updated_at=library.updated_at,
                        extra=library.extra_metadata or {}
                    ),
                )
                for library in libraries
            ]
        except Exception as e:
            raise DatabaseError(f"Failed to fetch libraries: {str(e)}")

    @logger
    async def update_library(self, library_id: str, data: LibraryUpdate) -> LibraryResponse:
        """Update library"""
        try:
            # Check if library exists
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
                
            library = await self.library_repository.update(library_id, data)
            if not library:
                raise DatabaseError("Library update failed unexpectedly")
                
            return LibraryResponse(
                id=str(library.id),
                name=library.name,
                indexed=library.indexed,
                metadata=LibraryMetadata(
                    description=library.description,
                    created_at=library.created_at,
                    updated_at=library.updated_at,
                    extra=library.extra_metadata or {}
                ),
            )
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to update library: {str(e)}")

    @logger
    async def delete_library(self, library_id: str) -> None:
        """Delete library and all its chunks/documents"""
        try:
            # Check if library exists
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
                
            # Delete library (cascades to chunks and documents)
            deleted = await self.library_repository.delete(library_id)
            if not deleted:
                raise DatabaseError("Library deletion failed unexpectedly")
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to delete library: {str(e)}")

    @logger
    async def library_exists(self, library_id: str) -> bool:
        """Check if library exists"""
        try:
            return await self.library_repository.exists(library_id)
        except Exception as e:
            raise DatabaseError(f"Failed to check library existence: {str(e)}")

    @logger
    async def get_library_stats(self) -> dict:
        """Get aggregated library statistics"""
        try:
            return await self.library_repository.get_stats()
        except Exception as e:
            raise DatabaseError(f"Failed to fetch library statistics: {str(e)}")

    @logger
    async def mark_library_indexed(self, library_id: str) -> None:
        """Mark library as indexed (all chunks have embeddings)"""
        try:
            # Check if library exists first
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
            
            success = await self.library_repository.mark_indexed(library_id)
            if not success:
                raise DatabaseError("Failed to mark library as indexed")
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to mark library as indexed: {str(e)}")

    @logger
    async def mark_library_unindexed(self, library_id: str) -> None:
        """Mark library as not indexed"""
        try:
            # Check if library exists first
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
            
            success = await self.library_repository.mark_unindexed(library_id)
            if not success:
                raise DatabaseError("Failed to mark library as unindexed")
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to mark library as unindexed: {str(e)}")