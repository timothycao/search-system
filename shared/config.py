"""
Shared configuration constants.
"""

# Top K results to return
DEFAULT_TOPK: int = 20

# Parser configs
CHUNK_SIZE: int = 1000000
MAX_DOCS: int | None = None # for testing

# Top level data directory
DATA_DIR: str = "data"

# Raw dataset file (MS MARCO collection.tsv)
RAW_DATA_PATH: str = f"{DATA_DIR}/raw/collection.tsv"

# Output directories
POSTINGS_DIR: str = f"{DATA_DIR}/postings"  # intermediate postings
INDEX_DIR: str = f"{DATA_DIR}/index"        # final index