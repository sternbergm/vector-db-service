from fastapi import APIRouter, HTTPException, Depends, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List
from uuid import UUID

from services.library_service import LibraryService
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.library_schema import LibraryCreate, LibraryUpdate, LibraryResponse
from schemas.search_schema import SearchRequest, SearchResponse
from exceptions import LibraryNotFoundError, DatabaseError, ValidationError
from decorators import logger, timer

# Import the vector service dependency from dependencies module
from services.dependencies import get_vector_service_dependency

async def get_library_service(db: AsyncSession = Depends(get_db)) -> LibraryService:
    library_repository = LibraryRepository(db)
    return LibraryService(library_repository)

library_service_dependency = Annotated[LibraryService, Depends(get_library_service)]

router = APIRouter(prefix="/libraries", tags=["libraries"])

@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
@logger
@timer
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
@logger
@timer
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
@logger
@timer
async def get_library(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    library_service: library_service_dependency
):
    """Get library by ID"""
    try:
        library = await library_service.get_library(str(library_id))
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
@logger
@timer
async def update_library(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    update_data: LibraryUpdate,
    library_service: library_service_dependency
):
    """Update library"""
    try:
        library = await library_service.update_library(str(library_id), update_data)
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
@logger
@timer
async def delete_library(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    library_service: library_service_dependency
):
    """Delete library and all its chunks/documents"""
    try:
        await library_service.delete_library(str(library_id))
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
@logger
@timer
async def get_library_stats(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    library_service: library_service_dependency
):
    """Get aggregated library statistics"""
    try:
        stats = await library_service.get_library_stats()
        return stats
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.post("/{library_id}/knn-search", response_model=SearchResponse)
@logger
@timer
async def knn_search(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    search_request: SearchRequest,
    vector_service = Depends(get_vector_service_dependency)
):
    """Perform k-NN similarity search in the specified library"""
    try:
        # Perform the search
        search_response = await vector_service.search_similar_chunks(
            query_text=search_request.query,
            library_id=str(library_id),
            k=search_request.k
        )
        
        return search_response
        
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        ) 