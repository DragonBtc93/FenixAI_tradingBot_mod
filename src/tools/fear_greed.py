# tools/fear_greed.py
import requests
import logging
from pydantic import BaseModel
class BaseTool(BaseModel):
    """Basic tool base class for scrapers."""
    name: str = ""
    description: str = ""

from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class FearGreedTool(BaseTool):
    name: str = "FearGreedIndexTool" # Renamed for clarity
    description: str = "Fetches the latest Crypto Fear & Greed Index value from Alternative.me API."

    def _run(self, limit: int = 1) -> Optional[str]: # Return type is string value or None
        """
        Fetches the Fear & Greed Index.

        Args:
            limit: Number of results to fetch (default is 1 for the latest).

        Returns:
            The Fear & Greed Index value as a string if successful, otherwise None.
        """
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        try:
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)
            
            data: Dict[str, Any] = response.json()
            
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                latest_entry = data["data"][0]
                if "value" in latest_entry:
                    value = str(latest_entry["value"]) # Ensure it's a string
                    logger.info(f"Fear & Greed Index fetched: {value}")
                    return value
                else:
                    logger.warning("Fear & Greed API response 'data' entry missing 'value' field.")
            else:
                logger.warning("Fear & Greed API response missing 'data' list or data is empty.")
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching Fear & Greed Index from {url}.")
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Fear & Greed Index: {http_err.response.status_code} - {http_err.response.text[:100]}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error fetching Fear & Greed Index: {req_err}")
        except ValueError as json_err: # Includes json.JSONDecodeError
            logger.error(f"Error decoding JSON response from Fear & Greed API: {json_err}")
        except Exception as e:
            logger.error(f"Unexpected error fetching Fear & Greed Index: {e}", exc_info=True)
            
        return None # Return None on any error

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool = FearGreedTool()
    index_value = tool._run()
    if index_value:
        print(f"Current Fear & Greed Index: {index_value}")
    else:
        print("Failed to retrieve Fear & Greed Index.")

