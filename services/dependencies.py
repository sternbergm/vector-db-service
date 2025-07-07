from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from services.vector_service import initialize_vector_service
from services.chunk_service import ChunkService
from services.library_service import LibraryService
from repositories.chunk_repository import ChunkRepository
from repositories.document_repository import DocumentRepository
from repositories.library_repository import LibraryRepository
from services.vector_service import VectorService

# Global service instances
vector_service_instance = None
library_service_instance = None
chunk_service_instance = None

async def get_vector_service_dependency(db: AsyncSession = Depends(get_db)) -> VectorService:
    """Get the vector service instance with lazy initialization."""
    global vector_service_instance, library_service_instance, chunk_service_instance
    
    if vector_service_instance is None:
        # Create repositories and services
        chunk_repository = ChunkRepository(db)
        document_repository = DocumentRepository(db)
        library_repository = LibraryRepository(db)
        chunk_service_instance = ChunkService(chunk_repository, document_repository, library_repository)
        library_service_instance = LibraryService(library_repository)
        
        # Initialize the vector service (this sets the global instance in vector_service.py too)
        vector_service_instance = initialize_vector_service(chunk_service_instance, library_service_instance)
        print("Vector service initialized successfully")
    
    return vector_service_instance

async def get_library_service_dependency(db: AsyncSession = Depends(get_db)):
    """Get the library service instance with lazy initialization."""
    global library_service_instance
    
    if library_service_instance is None:
        # This will be initialized in get_vector_service_dependency
        await get_vector_service_dependency(db)
    
    return library_service_instance

async def get_chunk_service_dependency(db: AsyncSession = Depends(get_db)):
    """Get the chunk service instance with lazy initialization."""
    global chunk_service_instance
    
    if chunk_service_instance is None:
        # This will be initialized in get_vector_service_dependency
        await get_vector_service_dependency(db)
    
    return chunk_service_instance 