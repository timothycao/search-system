"""
inverted_list.py
----------------
Binary-compatible InvertedList reader for VarByte-compressed postings.
Fully compatible with DAAT and BM25 scoring logic in query2.py.
"""

from typing import Dict, List, Tuple, Optional
import math
import struct

from shared.compression import varbyte_decode

INF_DOCID = 1 << 62


# ---------------------------------------------------------
#   Binary decoding utilities
# ---------------------------------------------------------


def decode_postings(encoded_doc_ids: bytes, encoded_freqs: bytes) -> Tuple[List[int], List[int]]:
    """
    Decode VarByte-compressed docIDs and freqs from binary segments.
    - Converts gap-encoded docIDs back to absolute values.
    Returns (decoded_doc_ids, decoded_freqs).
    """
    # Decode both sequences
    decoded_doc_ids = varbyte_decode(encoded_doc_ids)
    decoded_freqs = varbyte_decode(encoded_freqs)

    # Convert from gap form to absolute docIDs
    for i in range(1, len(decoded_doc_ids)):
        decoded_doc_ids[i] += decoded_doc_ids[i - 1]

    return decoded_doc_ids, decoded_freqs


def read_postings(index_path: str, term_meta: Dict) -> Tuple[List[int], List[int]]:
    """
    Read and decode postings for a given term from the binary inverted index.
    The term_meta entry should include 'offset', 'bytes_doc_ids', and 'bytes_freqs'.
    """
    offset = term_meta["offset"]
    bytes_doc_ids = term_meta["bytes_doc_ids"]
    bytes_freqs = term_meta["bytes_freqs"]

    with open(index_path, "rb") as index_file:
        # Jump to the term’s byte offset
        index_file.seek(offset)

        # Read the exact number of bytes for each segment
        encoded_doc_ids = index_file.read(bytes_doc_ids)
        encoded_freqs = index_file.read(bytes_freqs)

    # Delegate decoding
    return decode_postings(encoded_doc_ids, encoded_freqs)


# ---------------------------------------------------------
#   InvertedList class
# ---------------------------------------------------------

class InvertedList:
    """
    Represents a single term’s postings list loaded from a binary index.
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

        # Read postings from binary file
        self.docIDs, self.freqs = read_postings(index_path, term_meta)
        self.df = term_meta.get("df", len(self.docIDs))
        self.curr_idx = 0
        self.doc_id = self.docIDs[0] if self.docIDs else INF_DOCID

        # Precompute IDF and maximum BM25 contribution
        self.idf = self.compute_idf()
        self.max_score = self.compute_max_score()

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
        """Compute the maximum BM25 score this term can contribute (for MaxScore)."""
        max_score = 0.0
        for doc_id, freq in zip(self.docIDs, self.freqs):
            doc_meta = self.page_table.get(str(doc_id), {})
            doc_len = doc_meta.get("length", 1)
            score = self.getBM25(freq, doc_len)
            if score > max_score:
                max_score = score
        return max_score
    
    def reset(self) -> None:
        """Reset traversal state to the beginning of the list."""
        self.curr_idx = 0
        self.doc_id = self.docIDs[0] if self.docIDs else INF_DOCID

    # ------------------- Traversal API -------------------

    def nextGEQ(self, k: int) -> int:
        """Advance to the next docID ≥ k, or INF_DOCID if exhausted."""
        while self.curr_idx < len(self.docIDs) and self.docIDs[self.curr_idx] < k:
            self.curr_idx += 1

        # If we've moved past the end of the list
        if self.curr_idx >= len(self.docIDs):
            self.doc_id = INF_DOCID
            self.curr_idx = len(self.docIDs)  # Clamp safely
        else:
            self.doc_id = self.docIDs[self.curr_idx]

        return self.doc_id


    def getScore(self, doc_id: int) -> float:
        """Return the BM25 contribution of this term for the current docID."""
        if not self.docIDs or not self.freqs:
            return 0.0

        # Bounds check first
        if self.curr_idx < 0 or self.curr_idx >= len(self.docIDs):
            return 0.0

        # Ensure current posting matches target doc_id
        if self.docIDs[self.curr_idx] != doc_id:
            return 0.0

        # Safe to access
        freq = self.freqs[self.curr_idx] if self.curr_idx < len(self.freqs) else 0
        doc_meta = self.page_table.get(str(doc_id), {})
        doc_len = doc_meta.get("length", 1)
        return self.getBM25(freq, doc_len)



    def closeList(self) -> None:
        """Close the list (no-op for binary file access, included for symmetry)."""
        return
