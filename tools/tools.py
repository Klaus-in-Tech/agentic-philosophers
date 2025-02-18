from langchain_community.tools.tavily_search import TavilySearchResults


def get_latest_info(query: str):
    """Searches for latest info about the users query."""
    search = TavilySearchResults()
    res = search.run(f"{query}")
    return res
