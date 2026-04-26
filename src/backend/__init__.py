"""
Backend module for chat models.

Provides the main exports for the backend module including
ChatModel base class, types, and factory methods.
"""

from .chat import ChatModel, ChatModelOutput, Message
from .types import ChatModelParameters, ProtocolType, ProviderType

__all__ = [
    "ChatModel",
    "ChatModelParameters",
    "ChatModelOutput",
    "Message",
    "ProtocolType",
    "ProviderType",
]

# Made with Bob
