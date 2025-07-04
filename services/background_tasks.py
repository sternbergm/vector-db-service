import asyncio
import logging
from typing import Optional

from services.vector_service import VectorService
from services.embedding_service import EmbeddingError
from schemas.chunk_schema import ChunkResponse
from exceptions import ChunkNotFoundError, DatabaseError

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


async def batch_process_unindexed_chunks(library_id: str, unindexed_chunks: list[ChunkResponse], vector_service: VectorService):
    """
    Background task to process a list of unindexed chunks.
    
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
        
        # Process each chunk
        success_count = 0
        for chunk in unindexed_chunks:
            try:
                # Generate embedding and add to vector index
                success = await vector_service.add_chunk_vector(chunk)
                
                if success:
                    success_count += 1
                else:
                    logger.warning(f"Failed to process chunk {chunk.id}")
                    
                # Add small delay to avoid overwhelming the embedding service
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed for library {library_id}: {success_count}/{len(unindexed_chunks)} chunks processed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error during batch processing for library {library_id}: {str(e)}") 