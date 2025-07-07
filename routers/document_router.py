from fastapi import APIRouter, HTTPException, Depends, status, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, List, Dict, Any
from uuid import UUID

from services.document_service import DocumentService
from repositories.document_repository import DocumentRepository
from repositories.chunk_repository import ChunkRepository
from repositories.library_repository import LibraryRepository
from database.database import get_db
from schemas.document_schema import DocumentCreateFromChunks, DocumentResponse, DocumentSummary
from schemas.chunk_schema import ChunkMetadata
from exceptions import DocumentNotFoundError, LibraryNotFoundError, DatabaseError, ValidationError
from decorators import logger, timer

# Import background tasks and vector service dependency
from services.background_tasks import batch_process_unindexed_chunks
from services.dependencies import get_vector_service_dependency

async def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    document_repository = DocumentRepository(db)
    chunk_repository = ChunkRepository(db)
    library_repository = LibraryRepository(db)
    return DocumentService(document_repository, chunk_repository, library_repository)

document_service_dependency = Annotated[DocumentService, Depends(get_document_service)]

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])

@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
@logger
@timer
async def create_document_from_chunks(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    document_data: DocumentCreateFromChunks,
    background_tasks: BackgroundTasks,
    document_service: document_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Create a new document from a list of chunks with background embedding generation"""
    try:
        # Create chunk metadata if source is provided
        chunk_metadata = None
        if document_data.chunk_source:
            chunk_metadata = ChunkMetadata(
                source=document_data.chunk_source,
                extra={}
            )
        
        # Create document with chunks
        result = await document_service.create_document_with_chunks(
            library_id=str(library_id),
            chunk_texts=document_data.chunk_texts,
            document_metadata=document_data.document_metadata,
            chunk_metadata=chunk_metadata
        )
        
        # Add background task to generate embeddings for all chunks
        background_tasks.add_task(
            batch_process_unindexed_chunks,
            str(library_id),
            result["chunks"],
            vector_service
        )
        
        return {
            "message": "Document created successfully",
            "document_id": result["document_id"],
            "library_id": result["library_id"],
            "chunks_created": result["chunks_created"],
            "chunk_ids": result["chunk_ids"],
            "background_task_scheduled": True
        }
        
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

@router.get("/", response_model=List[DocumentSummary])
@logger
@timer
async def get_documents_by_library(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    document_service: document_service_dependency
):
    """Get all documents in the specified library"""
    try:
        documents = await document_service.get_documents_by_library(str(library_id))
        return [
            DocumentSummary(**doc) for doc in documents
        ]
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

@router.get("/{document_id}", response_model=DocumentResponse)
@logger
@timer
async def get_document(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    document_id: Annotated[UUID, Path(description="The UUID of the document")],
    document_service: document_service_dependency
):
    """Get document by ID with all its chunks"""
    try:
        document = await document_service.get_document(str(document_id))
        
        # Verify document belongs to the specified library
        if document["library_id"] != str(library_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found in library '{library_id}'"
            )
        
        return DocumentResponse(**document)
        
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@logger
@timer
async def delete_document(
    library_id: Annotated[UUID, Path(description="The UUID of the library")],
    document_id: Annotated[UUID, Path(description="The UUID of the document")],
    background_tasks: BackgroundTasks,
    document_service: document_service_dependency,
    vector_service = Depends(get_vector_service_dependency)
):
    """Delete document and all its chunks with background vector cleanup and reindexing"""
    try:
        # First get document to verify it exists and belongs to the library
        document = await document_service.get_document(str(document_id))
        
        # Verify document belongs to the specified library
        if document["library_id"] != str(library_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found in library '{library_id}'"
            )
        
        # Delete the document and all its chunks
        await document_service.delete_document(str(document_id))
        
        # Add background task to reindex the library
        from services.background_tasks import reindex_library
        background_tasks.add_task(
            reindex_library,
            str(library_id),
            vector_service
        )
        
        return None
        
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )