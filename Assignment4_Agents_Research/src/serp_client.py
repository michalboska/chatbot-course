import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ScholarResult:
    """Data class representing a search result from Google Scholar."""
    title: str
    link: str
    snippet: str
    publication_info: str
    authors: List[str]
    year: Optional[int]
    cited_by: int

class SerpClient:
    """
    Client for interacting with the SERP API to perform Google Scholar searches.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the SERP API client for Google Scholar.
        
        Args:
            api_key (str): The SERP API key.
        """
        if not api_key:
            raise ValueError("SERP API key is required")
        
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
    
    def scholar_search(self, query: str, num_results: int = 5) -> List[ScholarResult]:
        """
        Search Google Scholar using the SERP API.
        
        Args:
            query (str): The search query for academic papers.
            num_results (int): The number of results to return. Default is 5.
            
        Returns:
            List[ScholarResult]: A list of academic paper search results.
        """
        params = {
            "api_key": self.api_key,
            "engine": "google_scholar",
            "q": query,
            "num": num_results,
            "hl": "en"   # Language
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        
        # Extract the organic results from Google Scholar
        results = []
        if "organic_results" in data:
            for result in data["organic_results"][:num_results]:
                # Extract authors and year
                authors = result.get("authors", [])
                year = result.get("year")
                if year and isinstance(year, str):
                    try:
                        year = int(year)
                    except ValueError:
                        year = None
                
                # Create a ScholarResult instance
                paper_result = ScholarResult(
                    title=result.get("title", ""),
                    link=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    publication_info=result.get("publication_info", {}).get("summary", ""),
                    authors=authors,
                    year=year,
                    cited_by=result.get("inline_links", {}).get("cited_by", {}).get("total", 0)
                )
                
                results.append(paper_result)
        
        return results
    
