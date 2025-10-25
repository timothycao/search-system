# Parser configs
CHUNK_SIZE: int = 2000000  # number of lines to read into memory at once during parsing
MAX_DOCS: int | None = None # for testing

# Indexer configs
BLOCK_SIZE: int = 128

# Query processor configs
DEFAULT_TOPK: int = 20  # top k results to return

# Top level data directory
DATA_DIR: str = "data"

# Raw dataset file (MS MARCO collection.tsv)
RAW_DATA_PATH: str = f"{DATA_DIR}/raw/collection.tsv"

# Output directories
POSTINGS_DIR: str = f"{DATA_DIR}/postings"  # intermediate postings
INDEX_DIR: str = f"{DATA_DIR}/index"        # final index