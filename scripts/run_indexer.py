from search_system.indexer.indexer import run_indexer
from search_system.shared.config import POSTINGS_DIR, INDEX_DIR, BLOCK_SIZE

def main() -> None:
    input_dir: str = POSTINGS_DIR
    output_dir: str = INDEX_DIR
    block_size: int = BLOCK_SIZE
    run_indexer(input_dir, output_dir, block_size)

if __name__ == "__main__":
    main()