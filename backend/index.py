import logging
from typing import Optional, List
from r2r import R2RAsyncClient, R2RException

class IndexHandler:
    
    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        
    async def list_indices(self, filters: Optional[dict] = None, offset: Optional[int] = 0, limit: Optional[int] = 100) -> List[dict]:   
        """
        Retrieve a list of indices from the R2R service. Filters can be applied to the list of indices to narrow down the results.
        For example one can filter based on index name, table name or index method (cosine_distance, l2_distance, ip_distance).

        Args:
            filters: Optional filters to apply to the list of indices.
            offset: Optional offset for pagination.
            limit: Optional limit for pagination.

        Returns:
            List of index metadata from the R2R service.

        Raises:
            R2RException: If there is an error while listing the indices.
            Exception: If an unexpected error occurs.
        """
        try:
            indices = await self._client.indices.list(
                filters=filters, 
                offset=offset, 
                limit=limit
            )
            return indices['results']['indices']
        except R2RException as r2re:
            err_msg = f'[-] Error while listing indices: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while listing indices: {e} [-]')
            raise Exception(str(e)) from e
        
    async def create_index(self, index_method: str, index_name: str, index_measure: str, index_arguments: Optional[dict] = None, concurrently: bool = True) -> str:   
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
            index_method: Index method to use for the index. Only [hnsw and ivf_flat] are supported.
            index_name: Name of the index.
            index_measure: Measure to use for the index. Only [ip_distance, l2_distance and cosine_distance] are supported.
            index_arguments: Optional arguments to pass to the index method.
            concurrently: Optional flag to indicate whether to create the index concurrently. Defaults to True.

        Returns:
            A message indicating the result of the index creation.

        Raises:
            R2RException: If there is an error while creating the index.
            ValueError: If the index arguments are invalid.
            Exception: If an unexpected error occurs.
        """
        try:
            config = self._construct_index_config(index_method, index_name, index_measure, index_arguments, concurrently)
            index_result = await self._client.indices.create(
                config=config, 
                run_with_orchestration=False
            )
            return index_result['results']['message']
        except R2RException as r2re:
            err_msg = f'[-] Error while creating index: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except ValueError as ve:
            err_msg = f'[-] Invalid index arguments: {ve} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from ve
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while creating index: {e} [-]')
            raise Exception(str(e)) from e
        
    def _construct_index_config(self, index_method: str, index_name: str, index_measure: str, index_arguments: dict, concurrently: bool = True) -> dict:
        if index_method not in {'hnsw', 'ivf_flat'}:
            raise ValueError(f'[-] Invalid index method: {index_method} [-]! Only [hnsw and ivf_flat] are supported!')

        if index_measure not in {'ip_distance', 'l2_distance', 'cosine_distance'}:
            raise ValueError(f'[-] Invalid index measure: {index_measure} [-]! Only [ip_distance, l2_distance and cosine_distance] are supported!')
        
        config = {
            # According to the documentation it should be vectors. However, it doesn't work. 
            'table_name': 'chunks',
            'index_method': index_method,
            'index_measure': index_measure,
            'index_arguments': index_arguments,
            'index_name': index_name,
            # According documentaition it should be 'embedding', however it doesn't work. 
            # I've established connection to the pgvector container and found out the table structure. It should be 'vec'.
            'index_column': 'vec', 
            'concurrently': concurrently
        }
        return config
        
    async def get_index_details(self, index_name: str) -> dict: 
        """
        Retrieve details of an index in the R2R service. Information like performance statistics can be retrieved.

        Args:
            index_name: Name of the index to retrieve details for.

        Returns:
            Index metadata from the R2R service.

        Raises:
            R2RException: If there is an error while retrieving the index details.
            Exception: If an unexpected error occurs.
        """
        try:
            index = await self._client.indices.retrieve(
                index_name=index_name,
                table_name="chunks"
            )
            return index['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while retrieving index details: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while retrieving index details: {e} [-]')
            raise Exception(str(e)) from e
     
    async def delete_index_by_name(self, index_name: str) -> str: 
        """
        Delete an index in the R2R service.

        Args:
            index_name: Name of the index to delete.

        Returns:
            A message indicating the result of the index deletion. 

        Raises:
            R2RException: If there is an error while deleting the index.
            Exception: If an unexpected error occurs.
        """
        try:
            index = await self._client.indices.delete(
                index_name=index_name,
                table_name="chunks"
            )
            return index['results']['message']
        except R2RException as r2re:
            err_msg = f'[-] Error while deleting index: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 404) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while deleting index: {e} [-]')
            raise Exception(str(e)) from e