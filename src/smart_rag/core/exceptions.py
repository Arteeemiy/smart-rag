class ServiceError(Exception):
    """Base service exception."""


class RetrievalError(ServiceError):
    """Raised when retrieval fails."""


class LLMGenerationError(ServiceError):
    """Raised when the LLM call fails."""
