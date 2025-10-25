"""
Indexer for MS MARCO dataset.
Merges posting chunks and builds inverted index, lexicon, and page table.
"""

from os import makedirs, listdir, path
from json import dump
from math import log
from typing import Dict, List, Tuple, TextIO, Generator
from collections import defaultdict
from heapq import heappush, heappop

from tqdm import tqdm

from search_system.shared.compression import varbyte_encode

def run_indexer(input_dir: str, output_dir: str, block_size: int = 128) -> None:
    """
    Merge sorted posting chunks and build the final inverted index.
    Each posting: 'term docID freq'
    - Streams merged postings directly from chunk files
    - Builds inverted_index.bin, lexicon.json, and page_table.json
    - Writes postings in fixed-size blocks for efficient retrieval
    """
    makedirs(output_dir, exist_ok=True)
    inverted_index_path: str = path.join(output_dir, "inverted_index.bin")
    lexicon_path: str = path.join(output_dir, "lexicon.json")
    page_table_path: str = path.join(output_dir, "page_table.json")
    collection_stats_path: str = path.join(output_dir, "collection_stats.json")

    # Initialize index data structures
    lexicon: Dict[str, Dict] = {}
    page_table: Dict[str, Dict] = defaultdict(lambda: {"length": 0})

    # Initialize collection stats tracking
    total_len: int = 0
    total_docs: set = set()
    avg_len_estimate: float = 1.0  # will be updated after indexing

    # Initialize term tracking variables
    current_term: str = ""
    current_offset: int = 0
    doc_ids: List[int] = []
    freqs: List[int] = []

    # Create generator to stream merged postings
    generator = merge_postings(input_dir)

    # Count total number of postings across all chunks (for progress bar)
    total_postings: int = 0
    for chunk_fname in sorted(listdir(input_dir)):
        chunk_path = path.join(input_dir, chunk_fname)
        with open(chunk_path, "r", encoding="utf-8") as chunk_file:
            total_postings += sum(1 for _ in chunk_file)

    # Stream merged postings directly to index file
    with open(inverted_index_path, "wb") as inverted_index_file:
        with tqdm(total=total_postings, desc="Building index", unit="posting") as progress:
            for posting in generator:
                # Parse posting as (term, docID, freq)
                term, doc_id_str, freq_str = posting.strip().split()
                doc_id, freq = int(doc_id_str), int(freq_str)
                page_table[doc_id]["length"] += freq

                # Update collection stats
                total_len += freq
                total_docs.add(doc_id)

                # Update running average estimate
                if len(total_docs) > 0:
                    avg_len_estimate = total_len / len(total_docs)

                # Flush previous term when encountering a new one
                if current_term and term != current_term:
                    # Write previous term's postings in compressed fixed-size blocks
                    write_postings(inverted_index_file, lexicon, block_size, current_term, current_offset, doc_ids, freqs, page_table, avg_len_estimate)

                    # Advance byte offset by written length
                    current_offset += lexicon[current_term]["bytes"]
                    
                    # Reset buffers
                    doc_ids.clear()
                    freqs.clear()

                # Accumulate postings for the current term
                current_term = term
                doc_ids.append(doc_id)
                freqs.append(freq)
                progress.update(1)

            # Flush last term after merge completes
            if current_term:
                write_postings(inverted_index_file, lexicon, block_size, current_term, current_offset, doc_ids, freqs, page_table, avg_len_estimate)

    # Compute and record collection stats
    total_docs_count: int = len(total_docs)
    avg_len: float = total_len / total_docs_count if total_docs_count > 0 else 1.0
    collection_stats: Dict = {
        "total_docs": total_docs_count,
        "avg_len": avg_len
    }

    # Write lexicon to disk for lookup
    with open(lexicon_path, "w", encoding="utf-8") as lexicon_file:
        dump(lexicon, lexicon_file, indent=2)

    # Write page table to disk for lookup
    with open(page_table_path, "w", encoding="utf-8") as page_table_file:
        dump(page_table, page_table_file, indent=2)

    # Write collection stats to disk
    with open(collection_stats_path, "w", encoding="utf-8") as collection_stats_file:
        dump(collection_stats, collection_stats_file, indent=2)

    # print(f"[Indexer] Wrote inverted index, lexicon, and page table to {output_dir}")

def merge_postings(input_dir: str) -> Generator[str, None, None]:
    """
    Stream sorted postings from all chunk files using a heap (multi-way merge).
    Yields merged postings one at a time (term docID freq).
    """
    # Open each chunk file (sorted for deterministic order)
    chunk_files: List[TextIO] = []
    for chunk_fname in sorted(listdir(input_dir)):
        chunk_path: str = path.join(input_dir, chunk_fname)
        chunk_file: TextIO = open(chunk_path, "r", encoding="utf-8")
        chunk_files.append(chunk_file)

    # Initialize heap with first posting from each chunk
    min_heap: List[Tuple[str, int, str, int]] = []  # (term, docID, posting, file index)
    for i, chunk_file in enumerate(chunk_files):
        posting: str = chunk_file.readline().strip()    # read first line from each file
        term, doc_id, *_ = posting.split()
        heappush(min_heap, (term, int(doc_id), posting, i))

    # Yield postings in sorted order using heap
    while min_heap:
        # Repeatedly pop smallest tuple by (term, docID)
        term, doc_id, posting, file_idx = heappop(min_heap)
        yield posting   # stream one posting at a time

        # Push next posting from same chunk (readline automatically advances file pointer)
        next_posting: str = chunk_files[file_idx].readline().strip()
        if next_posting:
            next_term, next_doc_id, *_ = next_posting.split()
            heappush(min_heap, (next_term, int(next_doc_id), next_posting, file_idx))

    # Close all open chunk files
    for chunk_file in chunk_files:
        chunk_file.close()

def write_postings(
    inverted_index_file,
    lexicon: Dict[str, Dict],
    block_size: int,
    term: str,
    offset: int,
    doc_ids: List[int],
    freqs: List[int],
    page_table: Dict[str, Dict],
    avg_len: float
) -> None:
    """
    Write a term's postings in fixed-size blocks to the index file.
    Updates lexicon with block offsets and byte metadata.
    """
    if not doc_ids: return

    # BM25 parameters
    k1: float = 1.2
    b: float = 0.75 
    N: int = len(page_table)  # total number of documents in collection
    df: int = len(doc_ids)    # document frequency for term

    idf = compute_idf(df, N)

    # Initialize block tracking variables
    blocks_meta: List[Dict] = []
    current_offset: int = offset
    total_bytes: int = 0

    # Split and encode postings into fixed-size blocks
    for i in range(0, len(doc_ids), block_size):
        # Extract current block slice
        block_doc_ids = doc_ids[i : i + block_size]
        block_freqs = freqs[i : i + block_size]

        # Encode and write current block to index file
        encoded_doc_ids, encoded_freqs = encode_postings(block_doc_ids, block_freqs)
        inverted_index_file.write(encoded_doc_ids + encoded_freqs)

        # Compute byte sizes for current block
        bytes_doc_ids, bytes_freqs = len(encoded_doc_ids), len(encoded_freqs)
        bytes_block = bytes_doc_ids + bytes_freqs

        # Compute block level max BM25 score
        block_max_score: float = 0.0
        for doc_id, freq in zip(block_doc_ids, block_freqs):
            doc_meta = page_table.get(str(doc_id), {}) or page_table.get(doc_id, {})
            doc_len = doc_meta.get("length", 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / avg_len))
            score = idf * ((freq * (k1 + 1.0)) / denominator) if denominator != 0 else 0.0
            if score > block_max_score:
                block_max_score = score

        # TODO: block_max_score can be qunatized/rounded if needed for storage optimization
        #block_max_score = round(block_max_score, 3)

        # Record block metadata
        block_meta = {
            "offset": current_offset,           # byte offset
            "bytes_block": bytes_block,         # byte length of block
            "bytes_doc_ids": bytes_doc_ids,     # byte length of encoded docIDs
            "bytes_freqs": bytes_freqs,         # byte length of encoded freqs
            "last_doc_id": block_doc_ids[-1],   # last docID in block (for skipping)
            "block_max_score": block_max_score  # max BM25 score in block
        }
        blocks_meta.append(block_meta)

        # Advance byte offset and total
        current_offset += bytes_block
        total_bytes += bytes_block

    # Record term metadata in lexicon
    lexicon[term] = {
        "offset": offset,                   # byte offset
        "df": len(doc_ids),                 # document frequency
        "block_count": len(blocks_meta),    # number of blocks
        "blocks": blocks_meta,              # metadata for each block
        "bytes": total_bytes                # total bytes across blocks
    }

def encode_postings(doc_ids: List[int], freqs: List[int]) -> tuple[bytes, bytes]:
    """
    Compute gap-encoded docIDs and compress both docIDs and freqs using VarByte.
    Returns a tuple of (encoded_doc_ids, encoded_freqs) as separate byte streams.
    """
    if not doc_ids: return b"", b""
    
    # Convert docIDs to gap form for better compression
    gaps: List[int] = [doc_ids[0]]
    for i in range(1, len(doc_ids)):
        gaps.append(doc_ids[i] - doc_ids[i - 1])

    # Compress both sequences
    encoded_doc_ids = varbyte_encode(gaps)
    encoded_freqs = varbyte_encode(freqs)

    # Return encoded segments
    return encoded_doc_ids, encoded_freqs

def compute_idf(df: int, N: int) -> float:
    """
    Compute the Inverse Document Frequency (IDF) for a term.
    Uses the formula: log((N - df + 0.5) / (df + 0.5) + 1)
    """
    numerator = N - df + 0.5
    denominator = df + 0.5
    return log((numerator / denominator) + 1.0)