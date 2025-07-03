from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional
from pydantic import BaseModel

from services.chunk_service import ChunkService
from repositories.chunk_repository import ChunkRepository
from repositories.document_repository import DocumentRepository  
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.chunk_schema import ChunkCreate, ChunkUpdate, ChunkResponse, ChunkMetadata
from exceptions import ChunkNotFoundError, LibraryNotFoundError, DatabaseError, ValidationError, ChunkNotInLibraryError

# Create a request model without library_id since it comes from the URL
class ChunkCreateRequest(BaseModel):
    text: str
    metadata: Optional[ChunkMetadata] = None

async def get_chunk_service(db: AsyncSession = Depends(get_db)) -> ChunkService:
    chunk_repository = ChunkRepository(db)
    document_repository = DocumentRepository(db)
    library_repository = LibraryRepository(db)
    return ChunkService(chunk_repository, document_repository, library_repository)

chunk_service_dependency = Annotated[ChunkService, Depends(get_chunk_service)]

router = APIRouter(prefix="/libraries/{library_id}/chunks", tags=["chunks"])

@router.post("/", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED)
async def create_chunk(
    library_id: str,
    chunk_data: ChunkCreateRequest,
    chunk_service: chunk_service_dependency
):
    """Create a new chunk in the specified library"""
    try:
        # Create the full ChunkCreate object with library_id from URL
        full_chunk_data = ChunkCreate(
            text=chunk_data.text,
            library_id=library_id,
            document_id=None,  # Will be auto-managed by service
            metadata=chunk_data.metadata
        )
        
        chunk = await chunk_service.create_chunk(full_chunk_data)
        return chunk
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.get("/", response_model=List[ChunkResponse])
async def get_chunks_by_library(
    library_id: str,
    chunk_service: chunk_service_dependency
):
    """Get all chunks in the specified library"""
    try:
        chunks = await chunk_service.get_chunks_by_library(library_id)
        return chunks
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    library_id: str,
    chunk_id: str,
    chunk_service: chunk_service_dependency
):
    """Get chunk by ID"""
    try:
        # Use the new verify method that checks both existence and library membership
        chunk = await chunk_service.verify_chunk_in_library(chunk_id, library_id)
        return chunk
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ChunkNotInLibraryError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.put("/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(
    library_id: str,
    chunk_id: str,
    update_data: ChunkUpdate,
    chunk_service: chunk_service_dependency
):
    """Update chunk"""
    try:
        # First verify the chunk exists and belongs to the library
        await chunk_service.verify_chunk_in_library(chunk_id, library_id)
        
        # Then update the chunk
        chunk = await chunk_service.update_chunk(chunk_id, update_data)
        return chunk
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ChunkNotInLibraryError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(
    library_id: str,
    chunk_id: str,
    chunk_service: chunk_service_dependency
):
    """Delete chunk"""
    try:
        # First verify the chunk exists and belongs to the library
        await chunk_service.verify_chunk_in_library(chunk_id, library_id)
        
        # Then delete the chunk
        await chunk_service.delete_chunk(chunk_id)
        return None
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ChunkNotInLibraryError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.get("/unindexed", response_model=List[ChunkResponse])
async def get_unindexed_chunks(
    library_id: str,
    chunk_service: chunk_service_dependency
):
    """Get chunks that don't have embeddings yet"""
    try:
        chunks = await chunk_service.get_unindexed_chunks(library_id)
        return chunks
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.get("/stats")
async def get_chunk_stats(
    library_id: str,
    chunk_service: chunk_service_dependency
):
    """Get chunk statistics"""
    try:
        stats = await chunk_service.get_chunk_stats()
        return stats
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        ) 