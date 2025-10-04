"""
Shared configuration constants.
"""

# Top level data directory
DATA_DIR: str = "data"

# Raw dataset file (MS MARCO collection.tsv)
RAW_DATA_PATH: str = f"{DATA_DIR}/raw/collection.tsv"

# Output directories
POSTINGS_DIR: str = f"{DATA_DIR}/postings"  # intermediate postings
INDEX_DIR: str = f"{DATA_DIR}/index"        # final index