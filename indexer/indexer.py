"""
Indexer for MS MARCO dataset.
Merges posting chunks and builds inverted index, lexicon, and page table.
"""

from os import makedirs, listdir, path
from json import dump
from typing import Dict, List, Tuple, TextIO, Generator
from collections import defaultdict
from heapq import heappush, heappop

from tqdm import tqdm

def run_indexer(input_dir: str, output_dir: str) -> None:
    """
    Merge sorted posting chunks and build the final inverted index.
    Each posting: 'term docID freq'
    - Streams merged postings directly from chunk files
    - Builds inverted_index.bin, lexicon.json, and page_table.json
    """
    makedirs(output_dir, exist_ok=True)
    inverted_index_path: str = path.join(output_dir, "inverted_index.bin")
    lexicon_path: str = path.join(output_dir, "lexicon.json")
    page_table_path: str = path.join(output_dir, "page_table.json")

    # Initialize index data structures
    lexicon: Dict[str, Dict] = {}
    page_table: Dict[str, Dict] = defaultdict(lambda: {"length": 0})

    # Initialize term tracking variables
    current_term: str = ""
    current_offset: int = 0
    posting_list: List[str] = []

    # Create generator for merged postings
    generator = merge_postings(input_dir)

    # Count total number of postings across all chunks (for progress bar)
    total_postings: int = 0
    for chunk_fname in sorted(listdir(input_dir)):
        chunk_path = path.join(input_dir, chunk_fname)
        with open(chunk_path, "r", encoding="utf-8") as chunk_file:
            total_postings += sum(1 for _ in chunk_file)

    # Stream merged postings directly to index file
    with open(inverted_index_path, "w", encoding="utf-8") as inverted_index_file:
        # Sequentially process all streamed postings
        with tqdm(total=total_postings, desc="Building index", unit="posting") as progress:
            for posting in generator:
                # Parse posting as (term, docID, freq)
                term, doc_id, freq = posting.strip().split()
                page_table[doc_id]["length"] += int(freq)

                # When encountering a new term, flush previous one
                if current_term and term != current_term:
                    entry: str = " ".join(posting_list)

                    # TODO: replace with compressed binary encoding later
                    inverted_index_file.write(f"{current_term} {entry}\n")

                    # Record term metadata in lexicon
                    lexicon[current_term] = {
                        "offset": current_offset,   # byte offset
                        "df": len(posting_list)     # document freq
                    }

                    # Advance offset by written length (+1 for new line)
                    length: int = len(entry.encode("utf-8"))
                    current_offset += length + 1
                    posting_list.clear()

                # Accumulate postings for the current term
                current_term = term
                posting_list.append(f"{doc_id}:{freq}")
                progress.update(1)

            # Flush last term after generator completes
            if current_term:
                entry: str = " ".join(posting_list)
                inverted_index_file.write(f"{current_term} {entry}\n")
                lexicon[current_term] = {
                    "offset": current_offset,
                    "df": len(posting_list)
                }

    # Write lexicon to disk for lookup
    with open(lexicon_path, "w", encoding="utf-8") as lexicon_file:
        dump(lexicon, lexicon_file, indent=2)

    # Write page table to disk for lookup
    with open(page_table_path, "w", encoding="utf-8") as page_table_file:
        dump(page_table, page_table_file, indent=2)

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