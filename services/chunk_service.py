from typing import List, Optional
from repositories import ChunkRepository, DocumentRepository, LibraryRepository
from schemas.chunk_schema import ChunkCreate, ChunkUpdate, ChunkResponse, ChunkMetadata
from schemas.document_schema import DocumentMetadata
from exceptions import ChunkNotFoundError, LibraryNotFoundError, DatabaseError, ChunkNotInLibraryError

class ChunkService:
    def __init__(self, 
                 chunk_repository: ChunkRepository,
                 document_repository: DocumentRepository,
                 library_repository: LibraryRepository):
        self.chunk_repository = chunk_repository
        self.document_repository = document_repository
        self.library_repository = library_repository

    async def create_chunk(self, data: ChunkCreate) -> ChunkResponse:
        """Create a new chunk with automatic document management"""
        try:
            # 1. Validate library exists
            if not await self.library_repository.exists(data.library_id):
                raise LibraryNotFoundError(data.library_id)
            
            # 2. Auto-manage document creation based on source
            document_metadata = None
            if data.metadata and data.metadata.source:
                # Use source as document title for grouping
                document_metadata = DocumentMetadata(
                    title=data.metadata.source,
                    author=None
                )
            
            # Create or get document (creates default one if no metadata)
            document = await self.document_repository.create_or_get(
                data.library_id, 
                document_metadata
            )
            
            # 3. Set document_id in chunk data
            chunk_data_with_doc = ChunkCreate(
                text=data.text,
                library_id=data.library_id,
                document_id=str(document.id),
                metadata=data.metadata
            )
            
            # 4. Create the chunk
            chunk = await self.chunk_repository.create(chunk_data_with_doc)
            
            # 5. Return response
            return self._build_chunk_response(chunk)
        except (LibraryNotFoundError, ChunkNotFoundError):
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to create chunk: {str(e)}")

    async def get_chunk(self, chunk_id: str) -> ChunkResponse:
        """Get chunk by ID"""
        try:
            chunk = await self.chunk_repository.get(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(chunk_id)
                
            return self._build_chunk_response(chunk)
        except ChunkNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to fetch chunk: {str(e)}")

    async def get_chunks_by_library(self, library_id: str) -> List[ChunkResponse]:
        """Get all chunks in a library"""
        try:
            # Validate library exists
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
                
            chunks = await self.chunk_repository.get_by_library(library_id)
            return [self._build_chunk_response(chunk) for chunk in chunks]
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to fetch chunks: {str(e)}")

    async def update_chunk(self, chunk_id: str, data: ChunkUpdate) -> ChunkResponse:
        """Update chunk with automatic document re-management if needed"""
        try:
            # Check if chunk exists
            if not await self.chunk_repository.exists(chunk_id):
                raise ChunkNotFoundError(chunk_id)
                
            # Get current chunk to compare
            current_chunk = await self.chunk_repository.get(chunk_id)
            if not current_chunk:
                raise ChunkNotFoundError(chunk_id)
                
            # Update the chunk
            updated_chunk = await self.chunk_repository.update(chunk_id, data)
            if not updated_chunk:
                raise DatabaseError("Chunk update failed unexpectedly")
            
            # If text changed, mark library as unindexed
            if data.text is not None:
                await self.library_repository.mark_unindexed(updated_chunk.library_id)
            
            return self._build_chunk_response(updated_chunk)
        except ChunkNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to update chunk: {str(e)}")

    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk and update document/library statistics"""
        try:
            # Get chunk first to access its relationships
            chunk = await self.chunk_repository.get(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(chunk_id)
            
            # Delete the chunk
            deleted = await self.chunk_repository.delete(chunk_id)
            if not deleted:
                raise DatabaseError("Chunk deletion failed unexpectedly")
        except ChunkNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to delete chunk: {str(e)}")

    async def get_unindexed_chunks(self, library_id: str) -> List[ChunkResponse]:
        """Get chunks that don't have embeddings yet"""
        try:
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
                
            chunks = await self.chunk_repository.get_unindexed_chunks(library_id)
            return [self._build_chunk_response(chunk) for chunk in chunks]
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to fetch unindexed chunks: {str(e)}")

    async def get_chunk_stats(self) -> dict:
        """Get chunk statistics"""
        try:
            return await self.chunk_repository.get_stats()
        except Exception as e:
            raise DatabaseError(f"Failed to fetch chunk statistics: {str(e)}")

    async def verify_chunk_in_library(self, chunk_id: str, library_id: str) -> ChunkResponse:
        """Verify chunk exists and belongs to the specified library"""
        try:
            chunk = await self.get_chunk(chunk_id)  # This will raise ChunkNotFoundError if not found
            
            if chunk.library_id != library_id:
                raise ChunkNotInLibraryError(chunk_id, library_id)
            
            return chunk
        except (ChunkNotFoundError, ChunkNotInLibraryError):
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to verify chunk in library: {str(e)}")

    def _build_chunk_response(self, chunk) -> ChunkResponse:
        """Helper method to build ChunkResponse from database model"""
        metadata = ChunkMetadata(
            source=chunk.source,
            sentence_number=chunk.sentence_number,
            created_at=chunk.created_at,
            extra=chunk.extra_metadata or {}
        )
        
        return ChunkResponse(
            id=str(chunk.id),
            text=chunk.text,
            embedding=None,  # Embeddings stored in memory, not in DB
            document_id=chunk.document_id,
            library_id=chunk.library_id,
            metadata=metadata,
            similarity_score=None
        ) 