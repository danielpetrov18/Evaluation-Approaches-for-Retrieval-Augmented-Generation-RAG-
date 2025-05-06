# pylint: disable=C0114
# pylint: disable=R0903

from typing import List

from pydantic import BaseModel, Field

class Message(BaseModel):
    """
    Represents a chat message with embedding information.
    This class is relevant for my custom implementation of the history.
    It holds embedding information. When searching for relevant messages from previous
    interactions we will perform semantic similarity search on those embeddings.
    
    Attributes:
        id: Unique identifier for the message
        role: Role of the message sender (e.g., 'user', 'assistant')
        content: The actual message content
        embedding: Vector embedding of fixed length
    """
    id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)  # ... means required/not nullable
    content: str = Field(..., min_length=1)  # min_length=1 ensures non-empty string
    embedding: List[float] = Field(...)

    # https://docs.pydantic.dev/1.10/usage/model_config/
    class Config:
        """Configuration for the message. A message cannot be modified once created."""
        frozen = True  # Makes instances immutable
        extra = "forbid" # Prevents additional fields not defined in model
