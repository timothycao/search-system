from parser.parser import run_parser
from shared.config import RAW_DATA_PATH, POSTINGS_DIR

def main() -> None:
    input_path: str = RAW_DATA_PATH
    output_dir: str = POSTINGS_DIR
    chunk_size: int = 100000
    max_docs: int = 100000
    run_parser(input_path, output_dir, chunk_size, max_docs)

if __name__ == "__main__":
    main()