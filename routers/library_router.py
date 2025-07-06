from fastapi import APIRouter, HTTPException, Depends, status, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Optional
from uuid import UUID

from services.library_service import LibraryService
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.library_schema import LibraryCreate, LibraryUpdate, LibraryResponse
from schemas.search_schema import SearchRequest, SearchResponse, IndexAlgorithm, IndexAlgorithmRequest, IndexAlgorithmResponse
from exceptions import LibraryNotFoundError, DatabaseError, ValidationError
from decorators import logger, timer

# Import the vector service dependency from dependencies module
from services.dependencies import get_vector_service_dependency

# Import background tasks
from services.background_tasks import create_library_index_with_algorithm

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
    background_tasks: BackgroundTasks,
    library_service: library_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Create a new library"""
    try:
        library = await library_service.create_library(library_data)
        
        # Add background task to create index with preferred algorithm
        background_tasks.add_task(
            create_library_index_with_algorithm,
            library.id,
            library.preferred_index_algorithm,
            vector_service,
            library_service
        )
        
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
    background_tasks: BackgroundTasks,
    library_service: library_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Update library"""
    try:
        library = await library_service.update_library(str(library_id), update_data)
        
        # If preferred algorithm was changed, trigger background reindexing
        if update_data.preferred_index_algorithm is not None:
            background_tasks.add_task(
                create_library_index_with_algorithm,
                str(library_id),
                update_data.preferred_index_algorithm,
                vector_service,
                library_service
            )
        
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
    library_service: library_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Delete library and all its chunks/documents"""
    try:
        # Delete vector index first
        await vector_service.delete_library_index(str(library_id))
        
        # Delete library from database
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

@router.post("/{library_id}/index-algorithm", response_model=IndexAlgorithmResponse)
@logger
@timer
async def set_library_index_algorithm(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    algorithm_request: IndexAlgorithmRequest,
    background_tasks: BackgroundTasks,
    library_service: library_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Set or change the index algorithm for a library"""
    try:
        # Update library's preferred algorithm
        update_data = LibraryUpdate(preferred_index_algorithm=algorithm_request.algorithm)
        library = await library_service.update_library(str(library_id), update_data)
        
        # Add background task to rebuild index with new algorithm
        background_tasks.add_task(
            create_library_index_with_algorithm,
            str(library_id),
            algorithm_request.algorithm,
            vector_service,
            library_service
        )
        
        return IndexAlgorithmResponse(
            library_id=str(library_id),
            algorithm=algorithm_request.algorithm.value,
            success=True,
            message=f"Index algorithm set to {algorithm_request.algorithm.value}. Reindexing in background.",
            index_info=None  # Will be available after background task completes
        )
        
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

@router.get("/{library_id}/index-info")
@logger
@timer
async def get_library_index_info(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    vector_service = Depends(get_vector_service_dependency)
):
    """Get information about a library's current index"""
    try:
        index_info = await vector_service.get_library_index_info(str(library_id))
        if index_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No index found for library {library_id}"
            )
        return index_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index info: {str(e)}"
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
        # Use algorithm from search request if provided
        search_response = await vector_service.search_similar_chunks(
            query_text=search_request.query,
            library_id=str(library_id),
            k=search_request.k,
            algorithm=search_request.algorithm
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