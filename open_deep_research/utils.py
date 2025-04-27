import os
import asyncio
import requests
import random 
import concurrent
import aiohttp
import time
import logging
from typing import List, Optional, Dict, Any, Union
from urllib.parse import unquote

from bs4 import BeautifulSoup

from langsmith import traceable

from open_deep_research.state import Section
import os
#from serpapi import GoogleSearch
from typing import List, Optional, Dict, Any, Union, Callable

def get_config_value(value):
    """
    Helper function to handle both string and enum cases of configuration values
    """
    return value if isinstance(value, str) else value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],  # Tavily currently accepts no additional parameters
        #"perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        #"linkup": ["depth"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
            
    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

# -------------------------------
# Function for web scraping (synchronous)
# -------------------------------
def google_search_scraping(query: str, max_results: int, get_useragent: Callable[[], str]) -> List[dict]:
    """
    Performs Google search by scraping the web.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
        get_useragent (Callable): Function to generate a random user agent string.

    Returns:
        List[dict]: List of search results.
    """
    try:
        print(f"Scraping Google for '{query}'...")
        lang = "en"
        safe = "active"
        start = 0
        fetched_results = 0
        fetched_links = set()
        search_results = []
        
        while fetched_results < max_results:
            resp = requests.get(
                url="https://www.google.com/search",
                headers={
                    "User-Agent": get_useragent(),
                    "Accept": "*/*"
                },
                params={
                    "q": query,
                    "num": max_results + 2,
                    "hl": lang,
                    "start": start,
                    "safe": safe,
                },
                cookies={
                    'CONSENT': 'PENDING+987',  # Bypass the consent page
                    'SOCS': 'CAESHAgBEhIaAB',
                }
            )
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            result_block = soup.find_all("div", class_="ezO2md")
            new_results = 0
            
            for result in result_block:
                link_tag = result.find("a", href=True)
                title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                description_tag = result.find("span", class_="FrIlee")
                
                if link_tag and title_tag and description_tag:
                    link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                    
                    if link in fetched_links:
                        continue
                    
                    fetched_links.add(link)
                    title = title_tag.text
                    description = description_tag.text
                    
                    search_results.append({
                        "title": title,
                        "url": link,
                        "content": description,
                        "score": None,
                        "raw_content": description
                    })
                    
                    fetched_results += 1
                    new_results += 1
                    
                    if fetched_results >= max_results:
                        break
            
            if new_results == 0:
                break
            
            start += 10
            time.sleep(1)  # Delay between pages
        
        return search_results

    except Exception as e:
        print(f"Error in Google search for '{query}': {str(e)}")
        return []

# -------------------------------
# Asynchronous search function using Google API or fallback to scraping
# -------------------------------
@traceable
async def google_search_async(
    search_queries: Union[str, List[str]], 
    max_results: int = 2, 
    include_raw_content: bool = True,
    force_scraping: bool = False
):
    """
    Performs concurrent web searches using Google.
    Uses the Google Custom Search API if API credentials are set and force_scraping is False,
    otherwise falls back to the scraping function.

    Args:
        search_queries (Union[str, List[str]]): A single query or list of queries.
        max_results (int): Maximum number of results to return per query.
        include_raw_content (bool): Whether to fetch full page content.
        force_scraping (bool): If True, forces the use of the scraping function.

    Returns:
        List[dict]: List of search responses, each containing:
            - "query": the query string
            - "results": list of result dicts.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    # Use API only if credentials are available and we are not forcing scraping.
    use_api = bool(api_key and cx) and not force_scraping
    
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    def get_useragent():
        """Generates a random user agent string."""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    # Executor for synchronous operations when not using the API.
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    semaphore = asyncio.Semaphore(5 if use_api else 2)
    
    async def search_single_query(query):
        async with semaphore:
            try:
                results = []
                
                if use_api:
                    # Use API-based search
                    for start_index in range(1, max_results + 1, 10):
                        num = min(10, max_results - (start_index - 1))
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        print(f"Requesting {num} results for '{query}' from Google API...")
                        async with aiohttp.ClientSession() as session:
                            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break
                                data = await response.json()
                                
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        await asyncio.sleep(0.2)
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                else:
                    # Scraping branch via executor
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(
                        executor,
                        lambda: google_search_scraping(query, max_results, get_useragent)
                    )
                
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)
                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []
                        
                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }
                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            content_type = response.headers.get('Content-Type', '').lower()
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                result['raw_content'] = f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                            else:
                                                try:
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    result['raw_content'] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    result['raw_content'] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result['raw_content'] = f"[Error fetching content: {str(e)}]"
                                return result
                        
                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))
                        results = await asyncio.gather(*fetch_tasks)
                        print(f"Fetched full content for {len(results)} results")
                
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }
    
    try:
        search_tasks = [search_single_query(query) for query in search_queries]
        search_results = await asyncio.gather(*search_tasks)
        return search_results
    finally:
        if executor:
            executor.shutdown(wait=False)

# -------------------------------
# Asynchronous helper to obtain scraping results for each query using google_search_scraping
# -------------------------------
async def scraping_results_for_queries(query_list: list[str], max_results: int) -> List[dict]:
    """
    Runs the google_search_scraping function for each query asynchronously,
    wrapping the results in a dict with keys "query" and "results".
    """
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    def get_useragent():
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    tasks = [
        loop.run_in_executor(
            executor, 
            lambda q=query: {"query": q, "results": google_search_scraping(q, max_results, get_useragent)}
        )
        for query in query_list
    ]
    results = await asyncio.gather(*tasks)
    executor.shutdown(wait=False)
    return results

# -------------------------------
# Combined search selection function
# -------------------------------
async def select_and_execute_search(
    search_api: str, 
    query_list: list[str], 
    params_to_pass: dict
) -> str:
    """
    Select and execute the appropriate search API.
    
    It executes both the Google search API (or its scraping fallback) via google_search_async
    and the scraping function via google_search_scraping. The results from both methods are then
    combined, deduplicated, and formatted.
    
    Args:
        search_api: Name of the search API to use (currently supports "googlesearch")
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API (e.g., {"max_results": 5})
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "googlesearch":
        # Use google_search_async (which uses API or scraping depending on credentials)
        search_results = await google_search_async(query_list, **params_to_pass)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")
    
    # Now, get scraping results directly via the scraping function
    scraping_results = await scraping_results_for_queries(query_list, params_to_pass.get("max_results", 5))
    #print(f"Scraping results: {scraping_results}")
    
    # Combine the results from the API (or async search) and scraping
    combined_results = search_results + scraping_results
    #print(f"Combined results: {combined_results}")
    
    return deduplicate_and_format_sources(
        combined_results, 
        max_tokens_per_source=4000, 
        include_raw_content=False
    )