from typing import List

from query.query import run_query
from shared.config import INDEX_DIR

def main() -> None:
    index_path: str = INDEX_DIR
    query: str = input("Enter query: ")
    results: List[str] = run_query(index_path, query)
    print("Results:", results)

if __name__ == "__main__":
    main()