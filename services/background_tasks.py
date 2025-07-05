import asyncio
import logging
from typing import Optional, List

from services.vector_service import VectorService
from services.library_service import LibraryService
from services.embedding_service import EmbeddingError
from schemas.chunk_schema import ChunkResponse
from schemas.search_schema import IndexAlgorithm
from exceptions import ChunkNotFoundError, DatabaseError, LibraryNotFoundError

logger = logging.getLogger(__name__)


async def generate_and_index_embedding(chunk: ChunkResponse, vector_service: VectorService):
    """
    Background task to generate embedding and add to vector index.
    
    This task is triggered when a new chunk is created.
    
    Args:
        chunk: ChunkResponse object with all chunk data
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting background embedding generation for chunk {chunk.id}")
        
        # Generate embedding and add to vector index
        success = await vector_service.add_chunk_vector(chunk)
        
        if success:
            logger.info(f"Successfully generated and indexed embedding for chunk {chunk.id}")
        else:
            logger.error(f"Failed to generate or index embedding for chunk {chunk.id}")
            
    except EmbeddingError as e:
        logger.error(f"Embedding generation failed for chunk {chunk.id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during background embedding generation for chunk {chunk.id}: {str(e)}")


async def update_and_reindex_embedding(chunk: ChunkResponse, vector_service: VectorService):
    """
    Background task to update embedding and rebuild index.
    
    This task is triggered when a chunk's text is updated.
    
    Args:
        chunk: ChunkResponse object with updated chunk data
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting background embedding update for chunk {chunk.id}")
        
        # Update embedding and rebuild index
        success = await vector_service.update_chunk_vector(chunk)
        
        if success:
            logger.info(f"Successfully updated and reindexed embedding for chunk {chunk.id}")
        else:
            logger.error(f"Failed to update or reindex embedding for chunk {chunk.id}")
            
    except EmbeddingError as e:
        logger.error(f"Embedding update failed for chunk {chunk.id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during background embedding update for chunk {chunk.id}: {str(e)}")


async def remove_from_vector_index(chunk_id: str, library_id: str, vector_service: VectorService):
    """
    Background task to remove vector from index.
    
    This task is triggered when a chunk is deleted.
    
    Args:
        chunk_id: ID of the chunk to remove
        library_id: ID of the library the chunk belonged to
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting background vector removal for chunk {chunk_id} from library {library_id}")
        
        # Remove vector from index
        success = await vector_service.remove_chunk_vector(chunk_id, library_id)
        
        if success:
            logger.info(f"Successfully removed vector for chunk {chunk_id} from library {library_id}")
        else:
            logger.warning(f"Vector for chunk {chunk_id} was not found in library {library_id} index")
            
    except Exception as e:
        logger.error(f"Unexpected error during background vector removal for chunk {chunk_id}: {str(e)}")


async def reindex_library(library_id: str, vector_service: VectorService):
    """
    Background task to completely rebuild a library's vector index.
    
    This can be triggered manually or when there are major changes.
    
    Args:
        library_id: ID of the library to reindex
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting background library reindexing for library {library_id}")
        
        # Force rebuild of the library index
        await vector_service._rebuild_library_index(library_id)
        
        logger.info(f"Successfully reindexed library {library_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error during background library reindexing for library {library_id}: {str(e)}")


async def create_library_index_with_algorithm(library_id: str, 
                                            algorithm: IndexAlgorithm,
                                            vector_service: VectorService,
                                            library_service: LibraryService):
    """
    Background task to create a library index with the specified algorithm.
    
    This task is triggered when a library's preferred algorithm is set or changed.
    
    Args:
        library_id: ID of the library to index
        algorithm: Index algorithm to use
        vector_service: VectorService instance
        library_service: LibraryService instance
    """
    try:
        logger.info(f"Starting background library index creation for library {library_id} with algorithm {algorithm.value}")
        
        # Set the library algorithm in vector service
        success = await vector_service.set_library_algorithm(library_id, algorithm)
        
        if success:
            # Mark library as indexed
            await library_service.mark_library_indexed(library_id)
            logger.info(f"Successfully created {algorithm.value} index for library {library_id}")
        else:
            logger.error(f"Failed to create {algorithm.value} index for library {library_id}")
            
    except Exception as e:
        logger.error(f"Unexpected error during background library index creation for library {library_id}: {str(e)}")


async def batch_process_unindexed_chunks(library_id: str, 
                                       unindexed_chunks: List[ChunkResponse], 
                                       vector_service: VectorService):
    """
    Background task to process a list of unindexed chunks with batch embedding generation.
    
    This is useful for initial indexing or recovery scenarios.
    
    Args:
        library_id: ID of the library to process
        unindexed_chunks: List of ChunkResponse objects to process
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting batch processing of {len(unindexed_chunks)} unindexed chunks for library {library_id}")
        
        if not unindexed_chunks:
            logger.info(f"No unindexed chunks provided for library {library_id}")
            return
        
        # Use batch embedding generation for efficiency
        chunk_texts = [chunk.text for chunk in unindexed_chunks]
        
        # Generate embeddings in batch
        from services.embedding_service import embedding_service
        embeddings = await embedding_service.generate_embeddings_batch(chunk_texts)
        
        # Process each chunk with its embedding
        success_count = 0
        for chunk, embedding in zip(unindexed_chunks, embeddings):
            try:
                # Add embedding to vector storage
                from vector_db.storage import vector_storage
                success = vector_storage.add_vector(chunk.id, embedding, chunk.library_id, chunk.metadata)
                
                if success:
                    success_count += 1
                else:
                    logger.warning(f"Failed to add vector for chunk {chunk.id}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
                continue
        
        # Rebuild library index once at the end
        if success_count > 0:
            await vector_service._rebuild_library_index(library_id)
            
        logger.info(f"Batch processing completed for library {library_id}: {success_count}/{len(unindexed_chunks)} chunks processed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error during batch processing for library {library_id}: {str(e)}")


async def initialize_all_library_embeddings_and_indexes(vector_service: VectorService, 
                                                       library_service: LibraryService,
                                                       chunk_service):
    """
    Background task to initialize all library embeddings and indexes on startup.
    
    This loads all chunks from the database, creates embeddings for unindexed chunks,
    and builds indexes for all libraries.
    
    Args:
        vector_service: VectorService instance
        library_service: LibraryService instance
        chunk_service: ChunkService instance
    """
    try:
        logger.info("Starting initialization of all library embeddings and indexes")
        
        # Get all libraries
        libraries = await library_service.get_all_libraries()
        
        for library in libraries:
            try:
                library_id = library.id
                logger.info(f"Processing library {library_id}")
                
                # Get all chunks for this library
                all_chunks = await chunk_service.get_chunks_by_library(library_id)
                
                if not all_chunks:
                    logger.info(f"No chunks found in library {library_id}")
                    continue
                
                # Get unindexed chunks
                unindexed_chunks = await chunk_service.get_unindexed_chunks(library_id)
                
                if unindexed_chunks:
                    logger.info(f"Found {len(unindexed_chunks)} unindexed chunks in library {library_id}")
                    
                    # Process unindexed chunks in batch
                    await batch_process_unindexed_chunks(library_id, unindexed_chunks, vector_service)
                
                # Create index with preferred algorithm
                preferred_algorithm = library.preferred_index_algorithm
                logger.info(f"Creating {preferred_algorithm.value} index for library {library_id}")
                
                success = await vector_service.set_library_algorithm(library_id, preferred_algorithm)
                
                if success:
                    await library_service.mark_library_indexed(library_id)
                    logger.info(f"Successfully initialized library {library_id}")
                else:
                    logger.error(f"Failed to create index for library {library_id}")
                    
            except Exception as e:
                logger.error(f"Error processing library {library.id}: {str(e)}")
                continue
        
        logger.info("Completed initialization of all library embeddings and indexes")
        
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {str(e)}") 