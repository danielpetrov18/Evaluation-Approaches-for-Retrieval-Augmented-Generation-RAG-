"""
This module enables users to create and or interact with indices.
"""

import logging
from r2r import R2RAsyncClient, R2RException

class IndexHandler:
    """One can create, list, remove and refer to a given index."""

    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    async def list_indices(self, filters: dict = None, offset: int = 0, limit: int = 100):
        """
        Retrieve a list of indices from the R2R service. 
        Filters can be applied to the list of indices to narrow down the results.
        For example one can filter based on index name or table name.

        Args:
            filters (dict, optional): Filters to apply to the list of indices.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            WrappedVectorIndicesResponse: Index metadata from the R2R service.

        Raises:
            R2RException: If there is an error while listing the indices.
            Exception: If an unexpected error occurs.
        """
        try:
            indices_resp = await self._client.indices.list(
                filters=filters,
                offset=offset,
                limit=limit
            )
            return indices_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while listing indices: %s [-]', e)
            raise

    async def create_index(
        self,
        index_method: str,
        index_name: str,
        index_measure: str,
        index_arguments: dict = None
    ):
        """
        Create a new index in the R2R service.

        Example:
            my_new_index:
              index_method: hnsw
              index_measure: cosine_distance
              index_arguments:
                M: 16
                ef_construction: 200
              concurrently: true

        Args:
            index_method: Only hnsw and ivf_flat are supported.
            index_name: Name of the index.
            index_measure: Only [ip_distance, l2_distance and cosine_distance] are supported.
            index_arguments: Optional arguments to pass to the index method.

        Returns:
            A message indicating the result of the index creation.

        Raises:
            R2RException: If there is an error while creating the index.
            ValueError: If the index arguments are invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            config = self._construct_index_config(
                index_method,
                index_name,
                index_measure,
                index_arguments
            )
            index_result = await self._client.indices.create(
                config=config,
                run_with_orchestration=True
            )
            return index_result
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 500) from r2re
        except ValueError as ve:
            self._logger.error(str(ve))
            raise ValueError(str(ve)) from ve
        except Exception as e:
            self._logger.error('[-] Unexpected error while creating index: %s [-]', e)
            raise

    def _construct_index_config(
        self,
        index_method: str,
        index_name: str,
        index_measure: str,
        index_arguments: dict
    ) -> dict:
        # https://medium.com/@emreks/comparing-ivfflat-and-hnsw-with-pgvector-performance-analysis-on-diverse-datasets-e1626505bc9a
        # https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37/

        if index_method not in ('hnsw', 'ivf_flat'):
            raise ValueError('[-] Invalid index method, only hnsw and ivf_flat are supported! [-]')

        if index_measure not in ('ip_distance', 'l2_distance', 'cosine_distance'):
            raise ValueError('[-] Only ip_distance, l2_distance and cosine_distance are supported!')

        config = {
            # According to the documentation it should be vectors. However, it doesn't work.
            'table_name': 'chunks',
            'index_method': index_method,
            'index_measure': index_measure,
            'index_arguments': index_arguments,
            'index_name': index_name,
            # According documentaition it should be 'embedding', however it doesn't work.
            # I've established connection to the pgvector container. It should be 'vec'.
            'index_column': 'vec', 
            'concurrently': True
        }
        return config

    async def get_index_details(self, index_name: str):
        """
        Retrieve details of an index in the R2R service. 
        Information like performance statistics can be retrieved.

        Args:
            index_name: Name of the index to retrieve details for.

        Returns:
            WrappedVectorIndexResponse: Index metadata from the R2R service.

        Raises:
            R2RException: If there is an error while retrieving the index details.
            Exception: If an unexpected error occurs.
        """
        try:
            index_resp = await self._client.indices.retrieve(
                index_name=index_name,
                table_name="chunks"
            )
            return index_resp
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while retrieving index details: %s [-]', e)
            raise

    async def delete_index_by_name(self, index_name: str):
        """
        Delete an index in the R2R service.

        Args:
            index_name: Name of the index to delete.

        Returns:
            WrappedGenericMessageResponse: A message indicating the result of the index deletion. 

        Raises:
            R2RException: If there is an error while deleting the index.
            Exception: If an unexpected error occurs.
        """
        try:
            index = await self._client.indices.delete(
                index_name=index_name,
                table_name="chunks"
            )
            return index
        except R2RException as r2re:
            self._logger.error(str(r2re))
            raise R2RException(str(r2re), 404) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while deleting index: %s [-]', e)
            raise
