from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import asyncio
import logging

from database.database import create_tables, close_db
from routers.library_router import router as library_router
from routers.chunk_router import router as chunk_router
from routers.document_router import router as document_router
from services.dependencies import get_vector_service_dependency, get_library_service_dependency, get_chunk_service_dependency

# Import background tasks
from services.background_tasks import initialize_all_library_embeddings_and_indexes

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_tables()
    logger.info("Database tables created")
    
    # Initialize vector service
    # Note: We'll initialize the vector service on first request to avoid
    # database session issues during startup
    logger.info("Vector service will be initialized on first request")
    
    # Add startup task to initialize embeddings and indexes
    # We'll run this in the background after a short delay to allow the service to start
    async def startup_initialization():
        try:
            # Wait a bit for the service to fully start
            await asyncio.sleep(5)
            
            logger.info("Starting background initialization of embeddings and indexes...")
            
            # Get service instances
            from services.dependencies import get_vector_service, get_library_service, get_chunk_service
            
            vector_service = get_vector_service()
            library_service = get_library_service()
            chunk_service = get_chunk_service()
            
            # Initialize all library embeddings and indexes
            await initialize_all_library_embeddings_and_indexes(
                vector_service=vector_service,
                library_service=library_service,
                chunk_service=chunk_service
            )
            
            logger.info("Completed background initialization of embeddings and indexes")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings and indexes: {str(e)}")
    
    # Start the background initialization task
    asyncio.create_task(startup_initialization())
    
    yield
    
    # Shutdown  
    await close_db()
    logger.info("Database connection closed")

app = FastAPI(
    title="Vector DB Service",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(library_router, prefix="/api")
app.include_router(chunk_router, prefix="/api")
app.include_router(document_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "vector-db-svc"}

@app.get("/")
def root():
    return {
        "message": "Vector DB Service API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/vector-service/status")
async def vector_service_status(vector_service = Depends(get_vector_service_dependency)):
    """Get vector service status and statistics."""
    try:
        # Get all library indexes info
        indexes_info = await vector_service.get_all_library_indexes_info()
        
        # Get vector storage stats
        from vector_db.storage import vector_storage
        storage_stats = vector_storage.get_stats()
        memory_usage = vector_storage.get_memory_usage()
        
        return {
            "status": "initialized",
            "libraries_indexed": len(indexes_info),
            "indexes_info": indexes_info,
            "storage_stats": storage_stats,
            "memory_usage": memory_usage
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)