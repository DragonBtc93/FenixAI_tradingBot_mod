# tools/reddit_scraper.py
import requests
from pydantic import BaseModel
class BaseTool(BaseModel):
    """Basic tool base class for scrapers."""
    name: str = ""
    description: str = ""

import logging
from typing import List, Dict, Optional
from src.config.mlx_models import get_agent_model

logger = logging.getLogger(__name__)

class RedditScraper(BaseTool):
    name: str = "RedditPostTitleScraper" # More descriptive name
    description: str = (
        "Extracts titles of the most recent posts from specified subreddits "
        "using Reddit's public JSON API. Handles common request errors."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mlx_model_name = get_agent_model('sentiment')

    def _run(
        self,
        subreddits: Optional[List[str]] = None,
        limit_per_subreddit: int = 25 # Reduced default for efficiency
    ) -> Dict[str, List[str]]:
        """
        Args:
            subreddits: List of subreddits (without 'r/' prefix). 
                        Defaults to a crypto-focused list if None.
            limit_per_subreddit: Maximum number of post titles to retrieve per subreddit.

        Returns:
            A dictionary where keys are subreddit names and values are lists of post titles.
            If a subreddit scrape fails, its list will be empty.
        """
        if subreddits is None:
            subreddits = [
                "CryptoCurrency", "bitcoin", "ethereum", "ethtrader",
                "CryptoMarkets", "altcoin", "solana", "binance" # Diversified list
            ]
        
        # Standard browser user-agent to avoid potential blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        all_titles_by_subreddit: Dict[str, List[str]] = {}
        request_timeout = 10 # seconds

        for sub in subreddits:
            subreddit_name_cleaned = sub.strip().replace('r/', '') # Clean name
            url = f"https://www.reddit.com/r/{subreddit_name_cleaned}/new.json?limit={limit_per_subreddit}"
            logger.info(f"[RedditScraper] Scraping /r/{subreddit_name_cleaned} for {limit_per_subreddit} new post titles.")
            
            try:
                response = requests.get(url, headers=headers, timeout=request_timeout)
                response.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)
                
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                titles = []
                for post in posts:
                    if "data" in post and "title" in post["data"]:
                        titles.append(str(post["data"]["title"])) # Ensure string
                
                all_titles_by_subreddit[subreddit_name_cleaned] = titles
                logger.debug(f"[RedditScraper] Found {len(titles)} titles for /r/{subreddit_name_cleaned}.")

            except requests.exceptions.Timeout:
                logger.warning(f"[RedditScraper] Timeout connecting to Reddit for /r/{subreddit_name_cleaned}.")
                all_titles_by_subreddit[subreddit_name_cleaned] = []
            except requests.exceptions.HTTPError as http_err:
                logger.warning(f"[RedditScraper] HTTP error {http_err.response.status_code} for /r/{subreddit_name_cleaned}: {http_err.response.text[:100]}")
                all_titles_by_subreddit[subreddit_name_cleaned] = []
            except requests.exceptions.RequestException as req_err:
                logger.warning(f"[RedditScraper] Request error for /r/{subreddit_name_cleaned}: {req_err}")
                all_titles_by_subreddit[subreddit_name_cleaned] = []
            except ValueError as json_err: # Includes json.JSONDecodeError
                logger.warning(f"[RedditScraper] Error decoding JSON from Reddit for /r/{subreddit_name_cleaned}: {json_err}")
                all_titles_by_subreddit[subreddit_name_cleaned] = []
            except Exception as e:
                logger.error(f"[RedditScraper] Unexpected error scraping /r/{subreddit_name_cleaned}: {e}", exc_info=True)
                all_titles_by_subreddit[subreddit_name_cleaned] = []
        
        total_titles_scraped = sum(len(t) for t in all_titles_by_subreddit.values())
        logger.info(f"[RedditScraper] Reddit scraping complete. Total titles fetched: {total_titles_scraped}")
        return all_titles_by_subreddit

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = RedditScraper()
    
    # Example usage:
    # specific_subs = ["wallstreetbets", "stocks"]
    # results = scraper._run(subreddits=specific_subs, limit_per_subreddit=5)
    
    results = scraper._run(limit_per_subreddit=3) # Test with default subreddits

    for subreddit, titles_list in results.items():
        if titles_list:
            print(f"\n--- Titles from r/{subreddit} ---")
            for i, title_text in enumerate(titles_list):
                print(f"{i+1}. {title_text}")
        else:
            print(f"\n--- No titles found for r/{subreddit} ---")

