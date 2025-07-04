from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from services.vector_service import initialize_vector_service
from services.chunk_service import ChunkService
from repositories.chunk_repository import ChunkRepository
from repositories.document_repository import DocumentRepository
from repositories.library_repository import LibraryRepository

# Global vector service instance
vector_service_instance = None

async def get_vector_service_dependency(db: AsyncSession = Depends(get_db)):
    """Get the vector service instance with lazy initialization."""
    global vector_service_instance
    
    if vector_service_instance is None:
        # Create repositories and services
        chunk_repository = ChunkRepository(db)
        document_repository = DocumentRepository(db)
        library_repository = LibraryRepository(db)
        chunk_service = ChunkService(chunk_repository, document_repository, library_repository)
        
        # Initialize the vector service (this sets the global instance in vector_service.py too)
        vector_service_instance = initialize_vector_service(chunk_service)
        print("Vector service initialized successfully")
    
    return vector_service_instance 