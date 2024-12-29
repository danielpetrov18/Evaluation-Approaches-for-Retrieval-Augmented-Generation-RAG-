import logging
from typing import Optional, List
from r2r import R2RAsyncClient, R2RException

class SystemHandler:
    
    def __init__(self, client: R2RAsyncClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        
    async def health(self) -> str:   
        """
        Check the health of the R2R service.

        Returns:
            str: Returns OK if the service is healthy.
            
        Raises:
            R2RException: If the service is not healthy.
            Exception: If an unexpected error occurs.
        """
        try:
            health = await self._client.system.health()
            return health['results']['message']
        except R2RException as r2re:
            err_msg = f'[-] Error while checking health: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while checking health: {e} [-]')
            raise Exception(str(e)) from e
    
    async def settings(self) -> dict: 
        """
        Get the current R2R settings. The settings are the ones set in the config file (with toml extension in the /backend folder).

        Returns:
            dict: Settings from R2R service.

        Raises:
            R2RException: If the service encounters an error while getting the settings.
            Exception: If an unexpected error occurs.
        """
        try:
            settings = await self._client.system.settings()
            return settings['results']['config']
        except R2RException as r2re:
            err_msg = f'[-] Error while getting settings: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while getting settings: {e} [-]')
            raise Exception(str(e)) from e
        
    async def status(self) -> dict: 
        """
        Get the current R2R status. The status includes information such as the start time, uptime, CPU usage, 
        memory usage, run ID, run type, entries, and timestamp.

        Returns:
            dict: Status from R2R service.

        Raises:
            R2RException: If the service encounters an error while getting the status.
            Exception: If an unexpected error occurs.
        """
        try:
            status = await self._client.system.status()
            return status['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while getting status: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while getting status: {e} [-]')
            raise Exception(str(e)) from e
    
    async def logs(self, offset: Optional[int] = 0, limit: Optional[int] = 100) -> List[dict]:     
        """
        Retrieve logs from the R2R service.

        Returns:
            dict: Logs from the R2R service.

        Raises:
            R2RException: If there is an error while fetching logs.
            Exception: If an unexpected error occurs.
        """
        try:
            logs = await self._client.system.logs(
                offset=offset, 
                limit=limit
            ) 
            return logs['results']
        except R2RException as r2re:
            err_msg = f'[-] Error while getting logs: {r2re} [-]'
            self._logger.error(err_msg)
            raise R2RException(err_msg, 500) from r2re
        except Exception as e:
            self._logger.error(f'[-] Unexpected error while getting logs: {e} [-]')
            raise Exception(str(e)) from e