from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


class OperationStatus:
    """
    Statuses for long running operations
    """

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


# turned out to be unnecessary
@dataclass
class PredictionResponse:
    predictions: List[str]
    status: OperationStatus
    error: Optional[str]


@dataclass
class TaskResponse:
    task_id: str
    status: OperationStatus
    created_at: datetime


@dataclass
class ErrorResponse:
    error: str
