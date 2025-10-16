"""
inverted_list.py
----------------
Binary-compatible InvertedList reader for VarByte-compressed postings.
Fully compatible with DAAT and BM25 scoring logic in query.py.
"""

import math
from typing import Dict, List, Tuple

from shared.compression import varbyte_decode

INF_DOCID = 1 << 62

# ---------------------------------------------------------
#   Binary decoding utility
# ---------------------------------------------------------

def decode_postings(encoded_doc_ids: bytes, encoded_freqs: bytes) -> Tuple[List[int], List[int]]:
    """
    Decode VarByte-compressed docIDs and freqs from binary segments.
    - Converts gap-encoded docIDs back to absolute values.
    - Returns (decoded_doc_ids, decoded_freqs).
    """
    # Decode both sequences
    decoded_doc_ids = varbyte_decode(encoded_doc_ids)
    decoded_freqs = varbyte_decode(encoded_freqs)

    # Convert from gap form to absolute docIDs
    for i in range(1, len(decoded_doc_ids)):
        decoded_doc_ids[i] += decoded_doc_ids[i - 1]

    return decoded_doc_ids, decoded_freqs

# ---------------------------------------------------------
#   InvertedList class
# ---------------------------------------------------------

class InvertedList:
    """
    Represents a term's postings list stored in fixed-size blocks.
    Supports DAAT traversal and BM25 scoring.
    """

    def __init__(
        self,
        term: str,
        term_meta: Dict,
        index_path: str,
        page_table: Dict[str, Dict],
        N: int,
        avg_len: float,
        k1: float,
        b: float
    ) -> None:
        # Core metadata
        self.term = term
        self.term_meta = term_meta
        self.index_path = index_path
        self.page_table = page_table
        self.N = N
        self.avg_len = avg_len
        self.k1 = k1
        self.b = b

        # Block metadata
        self.blocks = term_meta.get("blocks", [])
        self.block_count = len(self.blocks)
        self.block_last_doc_ids = [block["last_doc_id"] for block in self.blocks]

        # Current traversal state
        self.curr_block_idx = 0
        self.curr_block_docIDs: List[int] = []
        self.curr_block_freqs: List[int] = []
        self.curr_idx = -1
        self.doc_id = INF_DOCID

        # Term-level stats
        self.df = term_meta.get("df", 0)
        self.idf = self.compute_idf()
        self.max_score = self.compute_max_score()   # maximum BM25 contribution

        # Initialize first block if available
        if self.block_count > 0:
            self.load_block(0)
            self.curr_idx = 0
            if self.curr_block_docIDs:
                self.doc_id = self.curr_block_docIDs[0]

    # ------------------- Block I/O -------------------

    def load_block(self, block_idx: int) -> None:
        """Load and decode a single block from disk into memory."""
        if block_idx < 0 or block_idx >= self.block_count:
            self.curr_block_docIDs, self.curr_block_freqs = [], []
            return

        block = self.blocks[block_idx]
        offset = block["offset"]
        bytes_doc_ids = block["bytes_doc_ids"]
        bytes_freqs = block["bytes_freqs"]

        with open(self.index_path, "rb") as index_file:
            # Jump to the block's byte offset
            index_file.seek(offset)

            # Read the exact number of bytes for each segment
            encoded_doc_ids = index_file.read(bytes_doc_ids)
            encoded_freqs = index_file.read(bytes_freqs)

        # Decode current block postings
        self.curr_block_docIDs, self.curr_block_freqs = decode_postings(encoded_doc_ids, encoded_freqs)
        
        # Update traversal state
        self.curr_block_idx = block_idx
        self.curr_idx = 0
        self.doc_id = self.curr_block_docIDs[0] if self.curr_block_docIDs else INF_DOCID

    # ------------------- BM25, scoring, cache -------------------

    def compute_idf(self) -> float:
        """Compute IDF for BM25."""
        numerator = self.N - self.df + 0.5
        denominator = self.df + 0.5
        return math.log((numerator / denominator) + 1.0)

    def getBM25(self, freq: int, doc_len: int) -> float:
        """Compute BM25 score contribution for this term in a document."""
        numerator = freq * (self.k1 + 1.0)
        denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_len))
        return self.idf * (numerator / denominator) if denominator != 0 else 0.0

    def compute_max_score(self) -> float:
        """Estimate maximum possible BM25 contribution across all blocks."""
        max_score = 0.0
        for block_idx in range(self.block_count):
            self.load_block(block_idx)
            for doc_id, freq in zip(self.curr_block_docIDs, self.curr_block_freqs):
                doc_meta = self.page_table.get(str(doc_id), {})
                doc_len = doc_meta.get("length", 1)
                score = self.getBM25(freq, doc_len)
                if score > max_score: max_score = score
        return max_score

    # ------------------- Traversal API -------------------

    def reset(self) -> None:
        """Reset traversal state to the first block."""
        self.curr_block_idx = 0
        if self.block_count > 0:
            self.load_block(0)
            self.curr_idx = 0
            self.doc_id = self.curr_block_docIDs[0]
        else:
            self.curr_idx = -1
            self.doc_id = INF_DOCID
    
    def nextGEQ(self, k: int) -> int:
        """
        Advance to the next docID ≥ k across blocks.
        Uses block skipping via last_doc_id metadata.
        """
        while True:
            # Skip entire blocks if their last docID < k
            while (
                self.curr_block_idx < self.block_count 
                and self.block_last_doc_ids[self.curr_block_idx] < k
            ):
                self.curr_block_idx += 1
                if self.curr_block_idx >= self.block_count:
                    self.doc_id = INF_DOCID
                    return self.doc_id
                self.load_block(self.curr_block_idx)

            # Within-block scan
            while (
                self.curr_idx < len(self.curr_block_docIDs)
                and self.curr_block_docIDs[self.curr_idx] < k
            ):
                self.curr_idx += 1

            # If we found docID ≥ k in current block
            if self.curr_idx < len(self.curr_block_docIDs):
                self.doc_id = self.curr_block_docIDs[self.curr_idx]
                return self.doc_id

            # Otherwise, move to next block and continue
            self.curr_block_idx += 1
            if self.curr_block_idx >= self.block_count:
                self.doc_id = INF_DOCID
                return self.doc_id
            self.load_block(self.curr_block_idx)

    def getScore(self, doc_id: int) -> float:
        """Return BM25 score contribution for the current docID."""
        if not self.curr_block_docIDs or not self.curr_block_freqs: return 0.0

        # Bounds check first
        if self.curr_idx < 0 or self.curr_idx >= len(self.curr_block_docIDs): return 0.0

        # Ensure current posting matches target doc_id
        if self.curr_block_docIDs[self.curr_idx] != doc_id: return 0.0

        # Safe to access
        freq = self.curr_block_freqs[self.curr_idx]
        doc_meta = self.page_table.get(str(doc_id), {})
        doc_len = doc_meta.get("length", 1)
        return self.getBM25(freq, doc_len)

    def closeList(self) -> None:
        """Close the list (no-op for binary file access, included for symmetry)."""
        return
