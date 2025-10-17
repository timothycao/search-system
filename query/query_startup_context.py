from os import path
from json import load
from typing import Dict


class QueryStartupContext:
    """Holds immutable index-wide data loaded once per session."""
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.lexicon_path = path.join(input_dir, "lexicon.json")
        self.page_table_path = path.join(input_dir, "page_table.json")
        self.index_path = path.join(input_dir, "inverted_index.bin")
        self.stats_path = path.join(input_dir, "collection_stats.json")

        # Load once
        self.lexicon = self.load_json(self.lexicon_path)
        self.page_table = self.load_json(self.page_table_path)       
        bm25_stats = self.load_json(self.stats_path)

        # BM25 parameters
        self.total_docs = bm25_stats.get("total_docs", 0)
        self.avg_len = bm25_stats.get("avg_len", 1.0)

    def load_json(self, file_path: str) -> Dict:
        """Load and return a JSON file as a dictionary."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = load(f)
        return data
