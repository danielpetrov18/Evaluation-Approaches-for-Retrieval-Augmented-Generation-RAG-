from uuid import UUID
from message import Message
from collections import deque
from typing import Dict, Optional, Iterator, List

class ChatHistory:
    """
    Manages a collection of chat messages with efficient search and retrieval capabilities.
    
    The ChatHistory class provides operations for managing message history
    with constant-time access to messages by ID and efficient iteration over messages
    in chronological order.
    
    Attributes:
        max_size (int): Maximum number of messages to store in history
        
    Example:
        >>> history = ChatHistory(max_size=100)
        >>> msg = Message(role="user", content="Hello", embedding=[0.1]*1024)
        >>> history.add_message(msg)
        >>> retrieved = history.get_message(msg.id)
    """
    
    def __init__(self, max_size: int = 30) -> None:
        """
        Initialize a new ChatHistory instance.
        
        Args:
            max_size: Maximum number of messages to retain in history
            
        Raises:
            ValueError: If max_size is less than 5
        """
        if max_size < 5:
            raise ValueError("History max size must be at least 5!")
            
        self._max_size = max_size
        self._message_index: Dict[UUID, Message] = {}
        self._messages: deque[Message] = deque(maxlen=max_size)
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the history, maintaining the size limit.
        
        If the history has reached its maximum size, the oldest message will be
        removed before adding the new one.
        
        Args:
            message: Message instance to add
        """ 
        if len(self._messages) == self._max_size:
            oldest = self._messages[0]
            self._remove_message(oldest.id)
        
        self._messages.append(message)
        self._message_index[message.id] = message
    
    def _remove_message(self, message_id: UUID) -> None:
        """
        Remove a message from history by its ID.
        
        Args:
            message_id: UUID of the message to remove     
        """
        if message_id in self._message_index:    
            message = self._message_index[message_id]
            self._messages.remove(message)
            del self._message_index[message_id]
    
    def get_message(self, message_id: UUID) -> Optional[Message]:
        """
        Retrieve a message by its ID.
        
        Args:
            message_id: UUID of the message to retrieve
            
        Returns:
            The Message instance or None if not found
        """
        return self._message_index.get(message_id)
    
    def get_all_messages(self) -> List[Message]:
        """
        Retrieve all messages in chronological order.
        
        Returns:
            List of all messages in the history
        """
        return list(self._messages)
    
    def clear(self):
        self._messages.clear()
        self._message_index.clear()
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)