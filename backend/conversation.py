"""
This modules helps the user to keep track of conversations and manage them.
"""

import io
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from r2r import R2RAsyncClient, R2RException

class ConversationHandler:
    """
    This class supports functionality for creating, listing conversations and their metadata.
    One can also export messages and or conversations.
    """

    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        load_dotenv()
        self._export_dir = Path(os.getenv("EXPORT_DIRECTORY"))
        self._export_dir.mkdir(parents=True, exist_ok=True)

    async def list_conversations(self, ids: list[str] = None, offset: int = 0, limit: int = 100):
        """
        Retrieve a list of conversations from the R2R service.

        Args:
            ids (list[str], optional): List of conversation IDs to filter by.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            WrappedConversationsResponse: List of conversations in the R2R service.

        Raises:
            R2RException: If there is an error while fetching the list of conversations.
            Exception: If an unexpected error occurs.
        """
        try:
            conversatios_resp = await self._client.conversations.list(
                ids,
                offset,
                limit
            )
            return conversatios_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing conversations: %s [-]', e)
            raise

    async def create_conversation(self, name: str, bearer_token: str):
        """
        Create a new conversation in the R2R service.

        Args:
            name (str): The name of the conversation.
            bearer_token (str): The bearer token for authorization.

        Raises:
            R2RException: If there is an error while creating the conversation.
            Exception: If an unexpected error occurs.
        """
        try:
            response = requests.post(
                url='http://127.0.0.1:7272/v3/conversations',
                headers={
                    'Authorization': f'Bearer {bearer_token}'
                },
                json={
                    'name': name
                },
                timeout=5
            )

            if response.status_code != 200:
                self._logger.error('[-] Failed to create conversation: %s [-]', response.text)
                raise R2RException(response.text, response.status_code)

        except Exception as e:
            self._logger.error('[-] Unexpected error while creating conversation: %s [-]', e)
            raise

    async def export_conversations_to_csv(
        self,
        bearer_token: str,
        out: str,
        filters: dict = None
    ):
        """
        Export conversations to a CSV file.

        This function sends a request to the R2R service to export conversations, 
        filters them based on the provided criteria, and writes the resulting CSV 
        data to a file.

        Args:
            bearer_token (str): The bearer token for authorization.
            out (str): The output filename for the exported CSV.
            filters (dict, optional): Filters to apply to the exported conversations. 

        Raises:
            R2RException: If there is an error while exporting the conversation.
            Exception: If an unexpected error occurs.
        """

        try:
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json',
                'Accept': 'text/csv'
            }

            payload = {
                'include_header': 'true',
                'columns': [
                    'id',
                    'created_at',
                    'name'
                ]
            }

            if filters is not None:
                payload['filters'] = filters

            response = requests.post(
                url='http://127.0.0.1:7272/v3/conversations/export',
                headers=headers,
                json=payload,
                timeout=5
            )

            if response.status_code != 200:
                self._logger.error('[-] Error exporting conversation: %s [-]', response.text)
                raise R2RException(response.text, response.status_code)

            df = pd.read_csv(io.BytesIO(response.content))
            if df.shape[0] == 0: # If the dataframe is empty (no rows)
                raise R2RException('No conversations found', 404)

            out = Path(
                self._export_dir,
                f'{out}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
            )
            df.to_csv(out, index=False)

        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while exporting conversation: %s [-]', e)
            raise

    async def export_messages_to_csv(
        self,
        bearer_token: str,
        out: str,
        filters: dict = None
    ):
        """
        Export messages to a CSV file.

        This function sends a request to the R2R service to export messages, 
        filters them based on the provided criteria, and writes the resulting CSV 
        data to a file.

        Args:
            bearer_token (str): The bearer token for authorization.
            out (str): The output filename for the exported CSV.
            filters (dict, optional): Filters to apply to the exported messages. 

        Raises:
            R2RException: If there is an error while exporting the messages.
            Exception: If an unexpected error occurs.
        """
        try:
            payload = {
                 "include_header": "true"
            }

            if filters:
                payload['filters'] = filters

            response = requests.post(
                url='http://127.0.0.1:7272/v3/conversations/export_messages',
                headers={
                    'Authorization': f'Bearer {bearer_token}',
                    "Content-Type": "application/json",
                    "Accept": "text/csv"
                },
                json=payload,
                timeout=5
            )

            if response.status_code != 200:
                self._logger.error('[-] Error exporting messages: %s [-]', response.text)
                raise R2RException(response.text, response.status_code)

            df = pd.read_csv(io.BytesIO(response.content))
            if df.shape[0] == 0: # If the dataframe is empty (no rows)
                raise R2RException('No messages found', 404)

            df['content'] = df['content'].apply(json.loads)
            df['metadata'] = df['metadata'].apply(json.loads)

            # Extract useful fields from JSON columns
            df['role'] = df['content'].apply(lambda x: x.get('role', None))
            df['message'] = df['content'].apply(lambda x: x.get('content', None))
            df['timestamp'] = df['metadata'].apply(lambda x: x.get('timestamp', None))

            # Select relevant columns
            df = df[
                [
                    'id', 
                    'conversation_id', 
                    'created_at', 
                    'parent_id', 
                    'role', 
                    'message', 
                    'timestamp'
                ]
            ]

            out = Path(
                self._export_dir,
                f'{out}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
            )
            df.to_csv(out, index=False)

        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while exporting messages: %s [-]', e)
            raise

    async def get_conversation(self, conversation_id: str):
        """
        Retrieve a conversation from the R2R service by its ID.

        Args:
            conversation_id (str): The ID of the conversation to retrieve.

        Returns:
            WrappedConversationMessagesResponse: The retrieved conversation.

        Raises:
            R2RException: If there is an error while retrieving the conversation.
            Exception: If an unexpected error occurs.
        """
        try:
            conversation_data = await self._client.conversations.retrieve(conversation_id)
            return conversation_data
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving conversation: %s [-]', e)
            raise

    async def delete_conversation(self, conversation_id: str):
        """
        Delete a conversation from the R2R service by its ID.

        Args:
            conversation_id (str): The ID of the conversation to delete.

        Returns:
            WrappedBooleanResponse: The deletion result.

        Raises:
            R2RException: If there is an error while deleting the conversation.
            Exception: If an unexpected error occurs.
        """

        try:
            deletion_resp = await self._client.conversations.delete(conversation_id)
            return deletion_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting conversation: %s [-]', e)
            raise

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        parent_id: str = None,
        metadata: dict = None
    ):
        """
        Add a message to a conversation in the R2R service.

        Args:
            conversation_id (str): The ID of the conversation to which the message will be added.
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
            parent_id (str, optional): The ID of the parent message, if applicable.
            metadata (dict, optional): Additional metadata associated with the message.

        Returns:
            WrappedMessageResponse: The response containing the added message.

        Raises:
            R2RException: If there is an error while adding the message.
            Exception: If an unexpected error occurs.
        """

        try:
            response = self._client.conversations.add_message(
                id=conversation_id,
                content=content,
                role=role,
                metadata=metadata,
                parent_id=parent_id
            )
            return response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while adding message: %s [-]', e)
            raise

    async def update_message(self, conversation_id: str, message_id: str, metadata: dict):
        """
        Update a message's metadata in a conversation in the R2R service.

        Args:
            conversation_id (str): The ID of the conversation containing the message.
            message_id (str): The ID of the message to update.
            metadata (dict): The new metadata for the message.

        Returns:
            WrappedMessageResponse: The response containing the updated message.

        Raises:
            R2RException: If there is an error while updating the message.
            Exception: If an unexpected error occurs.
        """

        try:
            response = self._client.conversations.update_message(
                id=conversation_id,
                message_id=message_id,
                metadata=metadata
            )
            return response
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while updating message: %s [-]', e)
            raise
