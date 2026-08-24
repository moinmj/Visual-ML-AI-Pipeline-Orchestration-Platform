from fastapi import HTTPException, status
from typing import Any, Optional, Dict


class PlatformException(Exception):
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundException(PlatformException):
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            message=f"{resource} with ID '{resource_id}' not found.",
            code="NOT_FOUND",
            details={"resource": resource, "id": str(resource_id)}
        )


class ValidationException(PlatformException):
    def __init__(self, message: str, errors: Optional[Any] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"validation_errors": errors}
        )


class ExecutionException(PlatformException):
    def __init__(self, message: str, node_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="EXECUTION_ERROR",
            details={"node_id": node_id, **(details or {})}
        )
