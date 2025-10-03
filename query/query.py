"""
Query processor placeholder.
Loads index files and supports simple query execution.
"""

from typing import List

def run_query(index_path: str, query: str) -> List[str]:
    print(f"[Query] Running query '{query}' against index in {index_path}")
    # TODO: implement BM25 and DAAT traversal
    return []
