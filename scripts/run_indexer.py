from indexer.indexer import run_indexer
from shared.config import POSTINGS_DIR, INDEX_DIR

def main() -> None:
    input_dir: str = POSTINGS_DIR
    output_dir: str = INDEX_DIR
    run_indexer(input_dir, output_dir)

if __name__ == "__main__":
    main()