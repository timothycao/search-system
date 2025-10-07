"""
Indexer for MS MARCO dataset.
Merges posting chunks and builds inverted index, lexicon, and page table.
"""

from os import makedirs, listdir, path
from json import dump
from typing import Dict, List, Tuple, TextIO
from collections import defaultdict
from heapq import heappush, heappop

from tqdm import tqdm

def run_indexer(input_dir: str, output_dir: str) -> None:
    """
    Entry point for indexer.
    """
    # print(f"[Indexer] Merging postings from {input_dir}, writing index to {output_dir}")
    postings: List[str] = merge_chunks(input_dir)
    build_index(postings, output_dir)

def merge_chunks(postings_dir: str) -> List[str]:
    """
    Merge sorted posting chunks using a heap (multi-way merge).
    Each chunk file is already sorted by (term, docID).
    """
    # Open each chunk file (sorted for deterministic order)
    chunk_files: List[TextIO] = []
    for chunk_fname in sorted(listdir(postings_dir)):
        chunk_path: str = path.join(postings_dir, chunk_fname)
        chunk_file: TextIO = open(chunk_path, "r", encoding="utf-8")
        chunk_files.append(chunk_file)
    
    # Count total number of postings across all chunks (for progress bar)
    total_postings: int = 0
    for chunk_file in chunk_files:
        total_postings += sum(1 for _ in chunk_file)
        chunk_file.seek(0)  # reset file pointer after counting

    # Initialize heap with first posting from each chunk
    min_heap: List[Tuple[str, int, str, int]] = []  # (term, docID, posting, file index)
    for i, chunk_file in enumerate(chunk_files):
        posting: str = chunk_file.readline().strip()    # read first line from each file
        term, doc_id, *_ = posting.split()
        heappush(min_heap, (term, int(doc_id), posting, i))

    # Merge postings in sorted order using heap
    merged_postings: List[str] = []
    with tqdm(total=total_postings, desc="Merging postings", unit="posting") as progress:
        while min_heap:
            # Repeatedly pop smallest tuple by (term, docID)
            term, doc_id, posting, file_idx = heappop(min_heap)
            merged_postings.append(posting)
            progress.update(1)

            # Push next posting from same chunk (readline automatically advances file pointer)
            next_posting: str = chunk_files[file_idx].readline().strip()
            if next_posting:
                next_term, next_doc_id, *_ = next_posting.split()
                heappush(min_heap, (next_term, int(next_doc_id), next_posting, file_idx))

    # Close all open chunk files
    for chunk_file in chunk_files:
        chunk_file.close()

    # TODO: stream merged postings directly to indexer for scalability
    return merged_postings

def build_index(postings: List[str], index_dir: str) -> None:
    """
    Build inverted index, lexicon, and page table from merged postings.
    """
    inverted_index_path: str = path.join(index_dir, "inverted_index.bin")
    lexicon_path: str = path.join(index_dir, "lexicon.json")
    page_table_path: str = path.join(index_dir, "page_table.json")

    makedirs(index_dir, exist_ok=True)

    lexicon: Dict[str, Dict] = {}
    term_to_postings: Dict[str, List[str]] = defaultdict(list)
    page_table: Dict[str, Dict] = defaultdict(lambda: {"length": 0})

    current_offset: int = 0
    with open(inverted_index_path, "w", encoding="utf-8") as inverted_index_file:
        # Build term postings and doc lengths in memory
        with tqdm(total=len(postings), desc="Preprocessing postings", unit="posting") as progress:
            for posting in postings:
                term, doc_id, freq = posting.split()
                term_to_postings[term].append(f"{doc_id}:{freq}")
                page_table[doc_id]["length"] += int(freq)
                progress.update(1)

        # Write postings list for each term
        with tqdm(total=len(term_to_postings), desc="Building index", unit="term") as progress:
            for term, posting_list in term_to_postings.items():
                entry: str = " ".join(posting_list)
                
                # TODO: replace with compressed binary encoding later
                inverted_index_file.write(f"{term} {entry}\n")

                # Record term metadata in lexicon
                lexicon[term] = {
                    "offset": current_offset,
                    "df": len(posting_list)
                }

                # Advance offset by written length (+1 for new line)
                length: int = len(entry.encode("utf-8"))
                current_offset += length + 1
                progress.update(1)

    # Write lexicon to JSON
    with open(lexicon_path, "w", encoding="utf-8") as lexicon_file:
        dump(lexicon, lexicon_file, indent=2)

    # Write page table to JSON
    with open(page_table_path, "w", encoding="utf-8") as page_table_file:
        dump(page_table, page_table_file, indent=2)

    # print(f"[Indexer] Wrote inverted index, lexicon, and page table to {index_dir}")