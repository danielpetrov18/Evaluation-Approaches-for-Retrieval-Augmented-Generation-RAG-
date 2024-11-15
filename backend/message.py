from typing import List
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

class Message(BaseModel):
    """
    Represents a chat message with metadata and embedding information.
    
    Attributes:
        id: Unique identifier for the message
        role: Role of the message sender (e.g., 'user', 'assistant')
        content: The actual message content
        embedding: Vector embedding of fixed length (1024 dimensions if using the mxbai-embed-large model)
        timestamp: UTC timestamp of when the message was created
    """
    id: UUID = Field(default_factory=uuid4)
    role: str = Field(..., min_length=1)  # ... means required/not nullable
    content: str = Field(..., min_length=1)  # min_length=1 ensures non-empty string
    embedding: List[float] = Field(..., max_length=1024, min_length=1024)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # https://docs.pydantic.dev/1.10/usage/model_config/
    class Config:
        frozen = True  # Makes instances immutable
        extra = "forbid" # Prevents additional fields not defined in model
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }