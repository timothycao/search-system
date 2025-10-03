from indexer.indexer import run_indexer
from shared.config import POSTINGS_DIR, INDEX_DIR

def main() -> None:
    postings_path: str = POSTINGS_DIR
    index_path: str = INDEX_DIR
    run_indexer(postings_path, index_path)

if __name__ == "__main__":
    main()