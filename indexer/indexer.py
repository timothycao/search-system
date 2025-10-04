"""
Indexer for MS MARCO dataset.
Merges posting chunks and builds inverted index, lexicon, and page table.
"""

from os import makedirs, listdir, path
from json import dump
from typing import Dict, List
from collections import defaultdict

def run_indexer(input_dir: str, output_dir: str) -> None:
    """
    Entry point for indexer.
    """
    # print(f"[Indexer] Merging postings from {input_dir}, writing index to {output_dir}")
    postings: List[str] = merge_chunks(input_dir)
    build_index(postings, output_dir)

def merge_chunks(postings_dir: str) -> List[str]:
    """
    Merge all chunk files into a single sorted posting list.
    """
    all_postings: List[str] = []
    for chunk_fname in listdir(postings_dir):
        # Read all lines from each chunk and extend into global postings
        chunk_path: str = path.join(postings_dir, chunk_fname)
        with open(chunk_path, "r", encoding="utf-8") as chunk_file:
            lines: List[str] = [line.strip() for line in chunk_file]
            all_postings.extend(lines)
    
    # TEMP: global sort in memory (replace later for scalability)
    all_postings.sort(key=lambda x: (x.split()[0], int(x.split()[1])))
    
    return all_postings

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
        for line in postings:
            term, doc_id, freq = line.split()
            term_to_postings[term].append(f"{doc_id}:{freq}")
            page_table[doc_id]["length"] += int(freq)

        # Write postings list for each term
        for term, postings in term_to_postings.items():
            entry: str = " ".join(postings)
            # TEMP: write as plain text (replace with compressed binary encoding later)
            inverted_index_file.write(f"{term} {entry}\n")

            # Record term metadata in lexicon
            lexicon[term] = {
                "offset": current_offset,
                "df": len(postings)
            }

            # Advance offset by written length (+1 for new line)
            length: int = len(entry.encode("utf-8"))
            current_offset += length + 1

    # Write lexicon to JSON
    with open(lexicon_path, "w", encoding="utf-8") as lexicon_file:
        dump(lexicon, lexicon_file, indent=2)

    # Write page table to JSON
    with open(page_table_path, "w", encoding="utf-8") as page_table_file:
        dump(page_table, page_table_file, indent=2)

    print(f"[Indexer] Wrote inverted index, lexicon, and page table to {index_dir}")