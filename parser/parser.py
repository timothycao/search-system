"""
Parser for MS MARCO dataset.
Reads collection.tsv and writes posting chunks.
"""

from os import makedirs, path
from typing import Dict, List

from shared.utils import tokenize

def run_parser(input_file: str, output_path: str, chunk_size: int = 100000, max_docs: int = None) -> None:
    """
    Parse MS MARCO collection.tsv and produce posting chunks.
    Each posting: 'term docID freq'
    - chunk_size: number of postings to buffer before writing a chunk
    - max_docs: optional limit for testing (stop after N docs)
    """
    # print(f"[Parser] Reading from {input_file}, writing postings to {output_path}")
    makedirs(output_path, exist_ok=True)

    postings: List[str] = []
    chunk_id: int = 0
    doc_count: int = 0

    file = open(input_file, "r", encoding="utf-8")
    
    for line in file:
        # Skip empty or whitespace-only lines
        if not line.strip(): continue
        
        # Split line into (docId, text)
        parts = line.strip().split("\t", 1)
        if len(parts) != 2: continue
        
        # Extract docID and text
        doc_id, text = parts
        doc_id = int(doc_id)

        # Build postings for this document
        frequencies = parse_document(text)
        for term, frequency in frequencies.items():
            postings.append(f"{term} {doc_id} {frequency}")

        # Increment counter and stop if it exceeds max_docs
        doc_count += 1
        if max_docs and doc_count >= max_docs: break

        # Flush buffer to disk if it exceeds chunk_size
        if len(postings) >= chunk_size:
            write_chunk(postings, output_path, chunk_id)
            postings.clear()
            chunk_id += 1
    
    file.close()

    # Flush any remaining postings
    if postings: write_chunk(postings, output_path, chunk_id)

    print(f"[Parser] Processed {doc_count} documents.")

def parse_document(text: str) -> Dict[str, int]:
    """
    Tokenize text and return a term frequency dictionary.
    """
    tokens: List[str] = tokenize(text)
    frequencies: Dict[str, int] = {}
    
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    
    return frequencies

def write_chunk(postings: List[str], output_path: str, chunk_id: int) -> None:
    """
    Sort postings and write them to a chunk file.
    """
    # Sort postings globally by (term, docID)
    postings.sort(key=lambda x: (x.split()[0], int(x.split()[1])))
    
    chunk_file = path.join(output_path, f"chunk{chunk_id}.txt")
    
    with open(chunk_file, "w", encoding="utf-8") as file:
        file.write("\n".join(postings))
    
    print(f"[Parser] Wrote {len(postings)} postings to {chunk_file}")