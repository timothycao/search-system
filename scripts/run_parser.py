from search_system.parser.parser import run_parser
from search_system.shared.config import RAW_DATA_PATH, POSTINGS_DIR, CHUNK_SIZE, MAX_DOCS

def main() -> None:
    dataset_path: str = RAW_DATA_PATH
    output_dir: str = POSTINGS_DIR
    chunk_size: int = CHUNK_SIZE
    max_docs: int | None = MAX_DOCS # optional (for testing)
    run_parser(dataset_path, output_dir, chunk_size, max_docs)

if __name__ == "__main__":
    main()