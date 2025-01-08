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
