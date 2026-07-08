from typing import Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import Config
from ..exceptions import CMSFetchError
from ..logger import setup_logger

logger = setup_logger(__name__)


class CMSClient:
    """HTTP client for CMS API with retry logic and API key/secret authentication"""

    def __init__(self, config: Config, api_key: str, api_secret: str):
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy and custom headers only"""
        session = requests.Session()

        # ✅ Force ONLY these headers
        session.headers.clear()
        session.headers.update({
            "X-API-Key": self.api_key,
            "X-API-Secret": self.api_secret,
        })

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def fetch(self, endpoint: str) -> Dict[str, Any]:
        """
        Fetch data from CMS API endpoint using X-API-Key and X-API-Secret headers only
        """
        url = f"{self.config.cms_base_url}/{endpoint}"

        try:
            logger.info(f"Fetching from: {url}")
            resp = self.session.get(url, timeout=self.config.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Successfully fetched {endpoint}")
            return data

        except requests.exceptions.Timeout:
            error_msg = f"Timeout fetching {endpoint}"
            logger.error(error_msg)
            raise CMSFetchError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error fetching {endpoint}: {e.response.status_code}"
            logger.error(error_msg)
            raise CMSFetchError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed for {endpoint}: {str(e)}"
            logger.error(error_msg)
            raise CMSFetchError(error_msg)

        except ValueError as e:
            error_msg = f"Invalid JSON response from {endpoint}: {str(e)}"
            logger.error(error_msg)
            raise CMSFetchError(error_msg)

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()