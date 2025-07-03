from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.database import create_tables, close_db
from routers.library_router import router as library_router
from routers.chunk_router import router as chunk_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_tables()
    print("Database tables created")
    yield
    # Shutdown  
    await close_db()
    print("Error creating tables")

app = FastAPI(
    title="Vector DB Service",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(library_router, prefix="/api")
app.include_router(chunk_router, prefix="/api")

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