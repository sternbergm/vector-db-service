import asyncio
import logging
from typing import Optional, List

from services.vector_service import VectorService
from services.library_service import LibraryService
from services.embedding_service import EmbeddingError
from schemas.chunk_schema import ChunkResponse
from schemas.search_schema import IndexAlgorithm

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


async def cleanup_orphaned_vectors(library_id: str, vector_service: VectorService):
    """
    Background task to clean up orphaned vectors that exist in vector storage 
    but not in the database.
    
    This is typically called before reindexing after document/chunk deletions.
    
    Args:
        library_id: ID of the library to clean up
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting orphaned vector cleanup for library {library_id}")
        
        # Get chunk IDs from vector storage
        from vector_db.storage import vector_storage
        vector_chunk_ids = set(vector_storage.filter_by_library(library_id))
        
        # Get chunk IDs from database
        db_chunk_ids = set()
        try:
            # Get all chunks from database for this library
            chunks = await vector_service.chunk_service.get_chunks_by_library(library_id)
            db_chunk_ids = {chunk.id for chunk in chunks}
        except Exception as e:
            logger.error(f"Failed to get chunks from database for library {library_id}: {str(e)}")
            return
        
        # Find orphaned vectors (exist in vector storage but not in database)
        orphaned_chunk_ids = vector_chunk_ids - db_chunk_ids
        
        if orphaned_chunk_ids:
            logger.info(f"Found {len(orphaned_chunk_ids)} orphaned vectors in library {library_id}")
            
            # Remove orphaned vectors
            removed_count = 0
            for chunk_id in orphaned_chunk_ids:
                try:
                    success = vector_storage.remove_vector(chunk_id)
                    if success:
                        removed_count += 1
                        logger.debug(f"Removed orphaned vector for chunk {chunk_id}")
                    else:
                        logger.warning(f"Failed to remove orphaned vector for chunk {chunk_id}")
                except Exception as e:
                    logger.error(f"Error removing orphaned vector for chunk {chunk_id}: {str(e)}")
                    continue
            
            logger.info(f"Removed {removed_count}/{len(orphaned_chunk_ids)} orphaned vectors from library {library_id}")
        else:
            logger.info(f"No orphaned vectors found in library {library_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error during orphaned vector cleanup for library {library_id}: {str(e)}")


async def reindex_library(library_id: str, vector_service: VectorService):
    """
    Background task to completely rebuild a library's vector index.
    
    This can be triggered manually or when there are major changes.
    This includes cleaning up orphaned vectors before rebuilding the index.
    
    Args:
        library_id: ID of the library to reindex
        vector_service: VectorService instance
    """
    try:
        logger.info(f"Starting background library reindexing for library {library_id}")
        
        # Clean up orphaned vectors before rebuilding
        await cleanup_orphaned_vectors(library_id, vector_service)
        
        # Force rebuild of the library index
        await vector_service._rebuild_library_index(library_id)
        
        logger.info(f"Successfully reindexed library {library_id}")
        
    except Exception as e:
        logger.error(f"Unexpected error during background library reindexing for library {library_id}: {str(e)}")


async def cleanup_all_orphaned_vectors(vector_service: VectorService):
    """
    Background task to clean up orphaned vectors across all libraries.
    
    This is useful for maintenance operations to ensure vector storage
    is in sync with the database.
    
    Args:
        vector_service: VectorService instance
    """
    try:
        logger.info("Starting global orphaned vector cleanup")
        
        # Get all library IDs that have vectors
        from vector_db.storage import vector_storage
        all_library_ids = vector_storage.get_library_ids()
        
        if not all_library_ids:
            logger.info("No libraries found with vectors")
            return
        
        logger.info(f"Found {len(all_library_ids)} libraries with vectors")
        
        # Clean up each library
        total_removed = 0
        for library_id in all_library_ids:
            try:
                # Get initial count
                initial_count = len(vector_storage.filter_by_library(library_id))
                
                # Clean up orphaned vectors
                await cleanup_orphaned_vectors(library_id, vector_service)
                
                # Get final count
                final_count = len(vector_storage.filter_by_library(library_id))
                
                removed_count = initial_count - final_count
                total_removed += removed_count
                
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} orphaned vectors from library {library_id}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up library {library_id}: {str(e)}")
                continue
        
        logger.info(f"Global orphaned vector cleanup completed. Total removed: {total_removed}")
        
    except Exception as e:
        logger.error(f"Unexpected error during global orphaned vector cleanup: {str(e)}")


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
        logger.info("Getting all libraries from database...")
        libraries = await library_service.get_all_libraries()
        logger.info(f"Found {len(libraries)} libraries in database")
        
        if not libraries:
            logger.warning("No libraries found in database during startup initialization")
            return
        
        for library in libraries:
            try:
                library_id = library.id
                logger.info(f"Processing library {library_id} - {library.name}")
                
                # Get all chunks for this library
                logger.info(f"Getting chunks for library {library_id}...")
                all_chunks = await chunk_service.get_chunks_by_library(library_id)
                logger.info(f"Found {len(all_chunks)} total chunks in library {library_id}")
                
                if not all_chunks:
                    logger.info(f"No chunks found in library {library_id}, skipping")
                    continue
                
                # Get unindexed chunks
                logger.info(f"Getting unindexed chunks for library {library_id}...")
                unindexed_chunks = await chunk_service.get_unindexed_chunks(library_id)
                logger.info(f"Found {len(unindexed_chunks)} unindexed chunks in library {library_id}")
                
                if unindexed_chunks:
                    logger.info(f"Processing {len(unindexed_chunks)} unindexed chunks in library {library_id}")
                    
                    # Process unindexed chunks in batch
                    await batch_process_unindexed_chunks(library_id, unindexed_chunks, vector_service)
                
                # Create index with preferred algorithm
                preferred_algorithm = library.preferred_index_algorithm
                logger.info(f"Creating {preferred_algorithm.value} index for library {library_id}")
                
                success = await vector_service.set_library_algorithm(library_id, preferred_algorithm)
                logger.info(f"Index creation result for library {library_id}: {success}")
                
                if success:
                    await library_service.mark_library_indexed(library_id)
                    logger.info(f"Successfully initialized library {library_id}")
                    
                    # VERIFY: Check if index is actually accessible
                    index_info = await vector_service.get_library_index_info(library_id)
                    if index_info:
                        logger.info(f"VERIFICATION: Index for library {library_id} is accessible - {index_info.algorithm} with {index_info.vector_count} vectors")
                    else:
                        logger.warning(f"VERIFICATION FAILED: Index for library {library_id} is not accessible after creation!")
                        
                else:
                    logger.error(f"Failed to create index for library {library_id}")
                    
            except Exception as e:
                logger.error(f"Error processing library {library.id}: {str(e)}")
                import traceback
                logger.error(f"Library processing traceback: {traceback.format_exc()}")
                continue
        
        # FINAL VERIFICATION: Check all indexes
        logger.info("FINAL VERIFICATION: Checking all library indexes...")
        all_indexes = await vector_service.get_all_library_indexes_info()
        logger.info(f"VERIFICATION COMPLETE: Found {len(all_indexes)} accessible indexes:")
        for lib_id, index_info in all_indexes.items():
            logger.info(f"  - Library {lib_id}: {index_info.algorithm} index with {index_info.vector_count} vectors")
        
        logger.info("Completed initialization of all library embeddings and indexes")
        
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {str(e)}")
        import traceback
        logger.error(f"Initialization traceback: {traceback.format_exc()}") 