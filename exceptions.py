"""
Custom exceptions for the Vector DB Service
"""

class VectorDBException(Exception):
    """Base exception for all vector DB service errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class NotFoundError(VectorDBException):
    """Raised when a requested resource is not found"""
    pass

class ValidationError(VectorDBException):
    """Raised when data validation fails"""
    pass

class DatabaseError(VectorDBException):
    """Raised when database operations fail"""
    pass

class DuplicateError(VectorDBException):
    """Raised when trying to create a resource that already exists"""
    pass

class InvalidOperationError(VectorDBException):
    """Raised when an operation is not allowed in the current state"""
    pass

class LibraryNotFoundError(NotFoundError):
    """Raised when a library is not found"""
    def __init__(self, library_id: str):
        super().__init__(f"Library with ID '{library_id}' not found")
        self.library_id = library_id

class ChunkNotFoundError(NotFoundError):
    """Raised when a chunk is not found"""
    def __init__(self, chunk_id: str):
        super().__init__(f"Chunk with ID '{chunk_id}' not found")
        self.chunk_id = chunk_id

class DocumentNotFoundError(NotFoundError):
    """Raised when a document is not found"""
    def __init__(self, document_id: str):
        super().__init__(f"Document with ID '{document_id}' not found")
        self.document_id = document_id

class ChunkNotInLibraryError(NotFoundError):
    """Raised when a chunk doesn't belong to the specified library"""
    def __init__(self, chunk_id: str, library_id: str):
        super().__init__(f"Chunk '{chunk_id}' not found in library '{library_id}'")
        self.chunk_id = chunk_id
        self.library_id = library_id 