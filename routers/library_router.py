from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List

from services.library_service import LibraryService
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.library_schema import LibraryCreate, LibraryUpdate, LibraryResponse
from exceptions import LibraryNotFoundError, DatabaseError, ValidationError

async def get_library_service(db: AsyncSession = Depends(get_db)) -> LibraryService:
    library_repository = LibraryRepository(db)
    return LibraryService(library_repository)

library_service_dependency = Annotated[LibraryService, Depends(get_library_service)]

router = APIRouter(prefix="/libraries", tags=["libraries"])

@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(
    library_data: LibraryCreate,
    library_service: library_service_dependency
):
    """Create a new library"""
    try:
        library = await library_service.create_library(library_data)
        return library
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

@router.get("/", response_model=List[LibraryResponse])
async def get_all_libraries(library_service: library_service_dependency):
    """Get all libraries"""
    try:
        libraries = await library_service.get_all_libraries()
        return libraries
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.get("/{library_id}", response_model=LibraryResponse)
async def get_library(
    library_id: str,
    library_service: library_service_dependency
):
    """Get library by ID"""
    try:
        library = await library_service.get_library(library_id)
        return library
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

@router.put("/{library_id}", response_model=LibraryResponse)
async def update_library(
    library_id: str,
    update_data: LibraryUpdate,
    library_service: library_service_dependency
):
    """Update library"""
    try:
        library = await library_service.update_library(library_id, update_data)
        return library
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

@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(
    library_id: str,
    library_service: library_service_dependency
):
    """Delete library and all its chunks/documents"""
    try:
        await library_service.delete_library(library_id)
        return None
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

@router.get("/{library_id}/stats")
async def get_library_stats(library_service: library_service_dependency):
    """Get aggregated library statistics"""
    try:
        stats = await library_service.get_library_stats()
        return stats
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.post("/{library_id}/knn-search")
async def knn_search(library_id: str):
    """Perform k-NN search (placeholder for vector search implementation)"""
    # TODO: Implement k-NN search once vector indexing is complete
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Vector search not yet implemented"
    ) 