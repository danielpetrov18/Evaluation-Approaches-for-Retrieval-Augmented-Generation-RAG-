"""
This module provides the user with the ability to interact with the system. 
One can load logs, check out the status and check out the current settings.
"""

import logging
from r2r import R2RClient, R2RException

class Settings:
    """
    This class encapsulates the system endpoints of the R2R service.
    """

    def __init__(self, client: R2RClient):
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    def health(self):
        """
        Check the health of the R2R service.

        Returns:
            WrappedGenericMessageResponse: Returns OK if the service is healthy.
            
        Raises:
            R2RException: If the service is not healthy.
            Exception: If an unexpected error occurs.
        """
        try:
            health_resp = self._client.system.health()
            return health_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while checking health: %s [-]', str(e))
            raise

    def settings(self):
        """
        Get the current R2R settings. 
        The settings are the ones set in the config file (.toml extension).

        Returns:
            WrappedSettingsResponse: Settings from R2R service.

        Raises:
            R2RException: If the service encounters an error while getting the settings.
            Exception: If an unexpected error occurs.
        """
        try:
            settings_resp = self._client.system.settings()
            return settings_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while getting settings: %s [-]', str(e))
            raise

    def status(self):
        """
        Get the current R2R status. 
        The status includes information such as the start time, uptime, CPU usage and so on.

        Returns:
            WrappedServerStatsResponse: Status from R2R service.

        Raises:
            R2RException: If the service encounters an error while getting the status.
            Exception: If an unexpected error occurs.
        """
        try:
            status_resp = self._client.system.status()
            return status_resp
        except R2RException as r2re:
            self._logger.error(r2re.message)
            raise R2RException(r2re.message, r2re.status_code) from r2re
        except Exception as e:
            self._logger.error('[-] Unexpected error while getting status: %s [-]', str(e))
            raise
