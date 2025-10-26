"""
Parser for MS MARCO dataset.
Reads collection.tsv and writes posting chunks.
"""

from os import makedirs, path
from typing import Dict, List, Optional
from collections import defaultdict

from tqdm import tqdm

from search_system.shared.utils import tokenize

def run_parser(dataset_path: str, output_dir: str, chunk_size: int = 1000000, max_docs: int = None, subset_ids_path: Optional[str] = None) -> None:
    """
    Parse MS MARCO collection.tsv and produce posting chunks.
    Each posting: 'term docID freq'
    - chunk_size: number of postings to buffer before writing a chunk
    - max_docs: optional limit for testing (stop after N docs)
    - subset_ids_path: optional .tsv file containing docIDs to include (first column only)
    """
    # print(f"[Parser] Reading from {dataset_path}, writing postings to {output_dir}")
    makedirs(output_dir, exist_ok=True)

    # Load subset IDs if provided
    subset_ids = set()
    if subset_ids_path:
        with open(subset_ids_path, "r", encoding="utf-8") as subset_ids_file:
            for line in subset_ids_file:
                if not line.strip(): continue
                subset_id = int(line.strip())
                subset_ids.add(subset_id)

    postings: List[str] = []
    chunk_id: int = 0
    doc_count: int = 0

    # Count total number of documents in the data set (for progress bar)
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        total_docs: int = max_docs or sum(1 for _ in dataset_file)

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        with tqdm(total=total_docs, desc="Parsing documents", unit="doc") as progress:
            for doc in dataset_file:
                # Skip empty or whitespace-only lines
                if not doc.strip(): continue
                
                # Split document into (docId, text)
                parts = doc.strip().split("\t", 1)
                if len(parts) != 2: continue

                # Extract docID and text
                doc_id, text = parts
                doc_id = int(doc_id)

                # Skip doc if not in subset (when subset_ids_path provided)
                if subset_ids and doc_id not in subset_ids:
                    progress.update(1)
                    continue

                # Build postings for this document
                freqs = parse_document(text)
                for term, freq in freqs.items():
                    postings.append(f"{term} {doc_id} {freq}")

                # Increment counter and stop if it exceeds max_docs
                doc_count += 1
                progress.update(1)
                if max_docs and doc_count >= max_docs: break

                # Flush buffer to disk if it exceeds chunk_size
                if len(postings) >= chunk_size:
                    write_chunk(postings, output_dir, chunk_id)
                    postings.clear()
                    chunk_id += 1

    # Flush any remaining postings
    if postings: write_chunk(postings, output_dir, chunk_id)

    # print(f"[Parser] Processed {doc_count} documents.")

def parse_document(text: str) -> Dict[str, int]:
    """
    Tokenize text and return a term frequency dictionary.
    """
    tokens: List[str] = tokenize(text)
    freqs: Dict[str, int] = defaultdict(int)
    
    for token in tokens:
        freqs[token] += 1
    
    return freqs

def write_chunk(postings: List[str], output_dir: str, chunk_id: int) -> None:
    """
    Sort postings and write them to a chunk file.
    """
    # Sort postings in memory by (term, docID)
    postings.sort(key=lambda x: (x.split()[0], int(x.split()[1])))
    
    chunk_path = path.join(output_dir, f"chunk{chunk_id}.txt")
    
    with open(chunk_path, "w", encoding="utf-8") as chunk_file:
        chunk_file.write("\n".join(postings))
    
    # print(f"[Parser] Wrote {len(postings)} postings to {chunk_path}")