from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import os
from .models import Base
from dotenv import load_dotenv

load_dotenv()

# Database URL - using asyncpg for PostgreSQL async support
DATABASE_URL = os.getenv("DATABASE_URL")

# Create async SQLAlchemy engine
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=False,          # Set to True for SQL logging in development
    # Important: These settings help with concurrent session safety
    pool_size=20,        # Number of connections to maintain in pool
    max_overflow=30,     # Additional connections beyond pool_size
    future=True          # Use SQLAlchemy 2.0 style
)


# Create async SessionLocal class with proper scoping
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autoflush=False,         # Don't auto-flush before queries
    autocommit=False         # Explicit transaction control
)



async def get_db() -> AsyncSession:
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

# Function to create all tables (for development/testing)
async def create_tables():
    """Create all tables - useful for development/testing"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Function to drop all tables (for testing)
async def drop_tables():
    """Drop all tables - useful for testing"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Function to close engine (for cleanup)
async def close_db():
    """Close database engine - call during application shutdown"""
    await engine.dispose()
