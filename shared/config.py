"""
Shared configuration constants.
"""

# Top level data directory
DATA_DIR: str = "data"

# Subdirectories
RAW_DATA_DIR: str = f"{DATA_DIR}/raw"       # raw dataset input (e.g. MS MARCO)
POSTINGS_DIR: str = f"{DATA_DIR}/postings"  # intermediate postings
INDEX_DIR: str = f"{DATA_DIR}/index"        # final index