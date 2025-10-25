import math
from typing import Dict, List, Tuple
import bisect

from search_system.shared.compression import varbyte_decode

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
        self.term = term
        self.term_meta = term_meta
        self.index_path = index_path
        self.page_table = page_table
        self.N = N
        self.avg_len = avg_len
        self.k1 = k1
        self.b = b

        # Open file handle once (persistent reuse)
        self.file = open(index_path, "rb")

        # Block metadata
        self.blocks = term_meta.get("blocks", [])
        self.block_count = len(self.blocks)
        self.block_last_doc_ids = [block["last_doc_id"] for block in self.blocks]

        # Load precomputed block level BM25 upper bounds
        self.block_max_scores = [block.get("block_max_score", 0.0) for block in self.blocks]
        self.max_score = max(self.block_max_scores) if self.block_max_scores else 0.0

        # Current traversal state
        self.curr_block_idx = 0
        self.curr_block_docIDs: List[int] = []
        self.curr_block_freqs: List[int] = []
        self.curr_idx = -1
        self.doc_id = INF_DOCID

        # Term-level stats
        self.df = term_meta.get("df", 0)
        self.idf = self.compute_idf()

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

        # Jump to the block's byte offset
        self.file.seek(offset)

        # Read the exact number of bytes for each segment
        encoded_doc_ids = self.file.read(bytes_doc_ids)
        encoded_freqs = self.file.read(bytes_freqs)

        # Decode current block postings
        self.curr_block_docIDs, self.curr_block_freqs = decode_postings(encoded_doc_ids, encoded_freqs)
        
        # Update traversal state
        self.curr_block_idx = block_idx
        self.curr_idx = 0
        self.doc_id = self.curr_block_docIDs[0] if self.curr_block_docIDs else INF_DOCID

    # ------------------- BM25 -------------------

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

            if self.curr_block_docIDs and self.curr_block_docIDs[-1] >= k:
                self.curr_idx = self.galloping_search(self.curr_block_docIDs, k, self.curr_idx)
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
    
    def galloping_search(self, arr, k, start=0):
        """Find first index i where arr[i] >= k using exponential + binary search."""
        if not arr or arr[-1] < k:
            return len(arr)
        step = 1
        i = start
        while i < len(arr) and arr[i] < k:
            i += step
            step *= 2
        lo = i // 2
        hi = min(i, len(arr))
        return bisect.bisect_left(arr, k, lo, hi)

    def closeList(self) -> None:
        """Close the list (no-op for binary file access, included for symmetry)."""
        if hasattr(self, "file") and not self.file.closed: self.file.close()

    def curr_block_max(self) -> float:
        """Return the precomputed BM25 upper bound for the current block."""
        if 0 <= self.curr_block_idx < len(self.block_max_scores):
            return self.block_max_scores[self.curr_block_idx]
        return 0.0

    def advance_to_next_block(self) -> None:
        """Advance to the next block safely."""
        if self.curr_block_idx + 1 < self.block_count:
            self.load_block(self.curr_block_idx + 1)
        else:
            self.doc_id = INF_DOCID
