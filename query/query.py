"""
Query processor for MS MARCO dataset.
Loads index files, parses query, fetches postings, and returns matching docIDs.
"""

from os import path
from json import load
from typing import Dict, Set, List, Tuple

from shared.compression import varbyte_decode

def run_query(input_dir: str, query: str, conjunctive: bool = True) -> List[int]:
    """
    Run a simple query against the binary compressed inverted index.
    conjunctive=True => AND semantics
    conjunctive=False => OR semantics
    """
    lexicon: Dict[str, Dict] = load_json(path.join(input_dir, "lexicon.json"))
    page_table: Dict[str, Dict] = load_json(path.join(input_dir, "page_table.json"))
    index_path: str = path.join(input_dir, "inverted_index.bin")

    # TEMP: simple whitespace tokenizer (unify with shared tokenizer later)
    terms: List[str] = query.lower().split()

    # Retrieve lists of docIDs for all query terms
    doc_id_lists: List[List[int]] = []
    for term in terms:
        decoded_doc_ids, _ = read_postings(index_path, lexicon, term)
        if decoded_doc_ids: doc_id_lists.append(decoded_doc_ids)
    
    # Return empty list if no terms found in lexicon
    if not doc_id_lists: return []

    # Initialize with the first docID list
    doc_ids: Set[int] = set(doc_id_lists[0])
    remaining_lists: List[List[int]] = doc_id_lists[1:]
    
    # Apply AND/OR semantics across docID lists
    if conjunctive:
        # Intersect all docID sets (documents must contain every term)
        for doc_id_list in remaining_lists:
            doc_ids &= set(doc_id_list)
    else:
        # Union all docID sets (documents may contain any term)
        for doc_id_list in remaining_lists:
            doc_ids |= set(doc_id_list)

    return sorted(doc_ids)

def load_json(file_path: str) -> Dict[str, Dict]:
    """
    Load a JSON file (used for lexicon and page table).
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return load(file)

def read_postings(index_path: str, lexicon: Dict[str, Dict], term: str) -> Tuple[List[int], List[int]]:
    """
    Read and decode the postings list for a given term.
    - Seeks to the byte offset stored in the lexicon.
    - Reads exact byte lengths for encoded docIDs and freqs.
    - Delegates decompression to decode_postings().
    Returns (decoded_doc_ids, decoded_freqs).
    """
    if term not in lexicon: return [], []

    meta = lexicon[term]
    offset = meta["offset"]
    bytes_doc_ids = meta["bytes_doc_ids"]
    bytes_freqs = meta["bytes_freqs"]

    with open(index_path, "rb") as index_file:
        # Jump to the term’s byte offset
        index_file.seek(offset)

        # Read the exact number of bytes for each segment
        encoded_doc_ids = index_file.read(bytes_doc_ids)
        encoded_freqs = index_file.read(bytes_freqs)

    # Delegate decoding
    return decode_postings(encoded_doc_ids, encoded_freqs)

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