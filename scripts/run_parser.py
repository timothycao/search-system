from parser.parser import run_parser
from shared.config import RAW_DATA_DIR, POSTINGS_DIR

def main() -> None:
    input_path: str = RAW_DATA_DIR
    output_path: str = POSTINGS_DIR
    run_parser(input_path, output_path)

if __name__ == "__main__":
    main()  