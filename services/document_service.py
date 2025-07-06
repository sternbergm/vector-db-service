from typing import List, Optional, Dict, Any
from repositories.document_repository import DocumentRepository
from repositories.chunk_repository import ChunkRepository
from repositories.library_repository import LibraryRepository
from schemas.document_schema import DocumentMetadata
from schemas.chunk_schema import ChunkCreate, ChunkResponse, ChunkMetadata
from exceptions import DocumentNotFoundError, LibraryNotFoundError, DatabaseError
from decorators import logger
import uuid

class DocumentService:
    def __init__(self, 
                 document_repository: DocumentRepository,
                 chunk_repository: ChunkRepository,
                 library_repository: LibraryRepository):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.library_repository = library_repository

    @logger
    async def create_document_with_chunks(self, 
                                        library_id: str,
                                        chunk_texts: List[str],
                                        document_metadata: Optional[DocumentMetadata] = None,
                                        chunk_metadata: Optional[ChunkMetadata] = None) -> Dict[str, Any]:
        """
        Create a document with a batch of chunks.
        
        Args:
            library_id: ID of the library to create the document in
            chunk_texts: List of text content for chunks
            document_metadata: Optional metadata for the document
            chunk_metadata: Optional metadata template for all chunks
            
        Returns:
            Dict containing document info and created chunks
        """
        try:
            # 1. Validate library exists
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
            
            # 2. Create the document first
            document = await self.document_repository.create_or_get(
                library_id=library_id,
                metadata=document_metadata
            )
            
            # 3. Create chunks using batch functionality
            chunk_data_list = []
            for i, chunk_text in enumerate(chunk_texts):
                # Create chunk metadata with sentence number
                chunk_meta = ChunkMetadata(
                    source=chunk_metadata.source if chunk_metadata else None,
                    sentence_number=i + 1,
                    extra=chunk_metadata.extra if chunk_metadata else {}
                )
                
                chunk_data = ChunkCreate(
                    text=chunk_text,
                    library_id=library_id,
                    document_id=str(document.id),
                    metadata=chunk_meta
                )
                chunk_data_list.append(chunk_data)
            
            # 4. Batch create chunks
            chunks = await self.batch_create_chunks(chunk_data_list)
            
            # 5. Return document and chunks info
            return {
                "document_id": str(document.id),
                "library_id": library_id,
                "chunks_created": len(chunks),
                "chunk_ids": [chunk.id for chunk in chunks],
                "chunks": chunks
            }
            
        except (LibraryNotFoundError, DocumentNotFoundError):
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to create document with chunks: {str(e)}")

    @logger
    async def batch_create_chunks(self, chunk_data_list: List[ChunkCreate]) -> List[ChunkResponse]:
        """
        Create multiple chunks in a single transaction.
        
        Args:
            chunk_data_list: List of ChunkCreate objects
            
        Returns:
            List of created ChunkResponse objects
        """
        try:
            if not chunk_data_list:
                return []
            
            # Use the batch_create method from repository for efficiency
            created_chunks = await self.chunk_repository.batch_create(chunk_data_list)
            
            # Convert to response objects
            chunk_responses = []
            for chunk in created_chunks:
                chunk_response = self._build_chunk_response(chunk)
                chunk_responses.append(chunk_response)
            
            return chunk_responses
            
        except Exception as e:
            raise DatabaseError(f"Failed to batch create chunks: {str(e)}")

    @logger
    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get document with its chunks"""
        try:
            document = await self.document_repository.get(document_id)
            if not document:
                raise DocumentNotFoundError(document_id)
            
            # Get all chunks for this document
            chunks = await self.chunk_repository.get_by_document(document_id)
            chunk_responses = [self._build_chunk_response(chunk) for chunk in chunks]
            
            return {
                "document_id": str(document.id),
                "library_id": str(document.library_id),
                "title": document.title,
                "author": document.author,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "chunks": chunk_responses,
                "chunk_count": len(chunk_responses)
            }
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to get document: {str(e)}")

    @logger
    async def get_documents_by_library(self, library_id: str) -> List[Dict[str, Any]]:
        """Get all documents in a library"""
        try:
            if not await self.library_repository.exists(library_id):
                raise LibraryNotFoundError(library_id)
            
            documents = await self.document_repository.get_by_library(library_id)
            
            result = []
            for doc in documents:
                # Get chunk count for each document
                chunk_count = await self.chunk_repository.count_by_document(str(doc.id))
                
                result.append({
                    "document_id": str(doc.id),
                    "library_id": str(doc.library_id),
                    "title": doc.title,
                    "author": doc.author,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at,
                    "chunk_count": chunk_count
                })
            
            return result
            
        except LibraryNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to get documents by library: {str(e)}")

    @logger
    async def delete_document(self, document_id: str) -> None:
        """Delete document and all its chunks"""
        try:
            # Check if document exists
            document = await self.document_repository.get(document_id)
            if not document:
                raise DocumentNotFoundError(document_id)
            
            # Delete all chunks first
            await self.chunk_repository.delete_by_document(document_id)
            
            # Delete the document
            deleted = await self.document_repository.delete(document_id)
            if not deleted:
                raise DatabaseError("Document deletion failed unexpectedly")
                
        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to delete document: {str(e)}")

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
            document_id=str(chunk.document_id) if chunk.document_id else None,
            library_id=str(chunk.library_id),
            metadata=metadata,
            similarity_score=None
        )