from fastapi import APIRouter, HTTPException, Depends, status, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional
from pydantic import BaseModel
from uuid import UUID

from services.chunk_service import ChunkService
from repositories.chunk_repository import ChunkRepository
from repositories.document_repository import DocumentRepository  
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.chunk_schema import ChunkCreate, ChunkUpdate, ChunkResponse, ChunkMetadata
from exceptions import ChunkNotFoundError, LibraryNotFoundError, DatabaseError, ValidationError, ChunkNotInLibraryError
from decorators import logger, timer

# Import background tasks and vector service dependency
from services.background_tasks import (
    generate_and_index_embedding,
    update_and_reindex_embedding,
    remove_from_vector_index
)
from services.dependencies import get_vector_service_dependency

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
@logger
@timer
async def create_chunk(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_data: ChunkCreateRequest,
    background_tasks: BackgroundTasks,
    chunk_service: chunk_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Create a new chunk in the specified library with background embedding generation"""
    try:
        # Create the full ChunkCreate object with library_id from URL
        full_chunk_data = ChunkCreate(
            text=chunk_data.text,
            library_id=str(library_id),
            document_id=None,  # Will be auto-managed by service
            metadata=chunk_data.metadata
        )
        
        # 1. Create chunk in database
        chunk = await chunk_service.create_chunk(full_chunk_data)
        
        # 2. Add background task to generate embedding and add to vector index
        background_tasks.add_task(generate_and_index_embedding, chunk, vector_service)
        
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
@logger
@timer
async def get_chunks_by_library(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_service: chunk_service_dependency
):
    """Get all chunks in the specified library"""
    try:
        chunks = await chunk_service.get_chunks_by_library(str(library_id))
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
@logger
@timer
async def get_chunk(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_id: Annotated[UUID, Path(description="The UUID of the chunk")],
    chunk_service: chunk_service_dependency
):
    """Get chunk by ID"""
    try:
        chunk = await chunk_service.verify_chunk_in_library(str(chunk_id), str(library_id))
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
@logger
@timer
async def update_chunk(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_id: Annotated[UUID, Path(description="The UUID of the chunk")],
    update_data: ChunkUpdate,
    background_tasks: BackgroundTasks,
    chunk_service: chunk_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Update chunk with background embedding re-generation if text changed"""
    try:
        # First verify the chunk exists and belongs to the library
        await chunk_service.verify_chunk_in_library(str(chunk_id), str(library_id))
        
        # Update the chunk
        chunk = await chunk_service.update_chunk(str(chunk_id), update_data)
        
        # If text was updated, trigger background embedding update
        if update_data.text is not None:
            background_tasks.add_task(update_and_reindex_embedding, chunk, vector_service)
        
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
@logger
@timer
async def delete_chunk(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_id: Annotated[UUID, Path(description="The UUID of the chunk")],
    background_tasks: BackgroundTasks,
    chunk_service: chunk_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Delete chunk with background vector index cleanup"""
    try:
        # First verify the chunk exists and belongs to the library
        await chunk_service.verify_chunk_in_library(str(chunk_id), str(library_id))
        
        # Delete the chunk from database
        await chunk_service.delete_chunk(str(chunk_id))
        
        # Add background task to remove from vector index
        background_tasks.add_task(
            remove_from_vector_index,
            str(chunk_id),
            str(library_id),
            vector_service
        )
        
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
@logger
@timer
async def get_unindexed_chunks(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    chunk_service: chunk_service_dependency
):
    """Get chunks that don't have embeddings yet"""
    try:
        chunks = await chunk_service.get_unindexed_chunks(str(library_id))
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
@logger
@timer
async def get_chunk_stats(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
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