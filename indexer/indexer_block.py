from typing import Dict, List, Tuple, Iterable, TextIO
import os
import json
from collections import defaultdict
from heapq import heappush, heappop
from tqdm import tqdm

# -----------------------------
# Tunables
# -----------------------------
BLOCK_POSTINGS = 128  # postings per block (fixed-size blocks)

# -----------------------------
# Utilities
# -----------------------------

def varbyte_encode(numbers: Iterable[int]) -> bytes:
    """
    VarByte encode non-negative ints.
    Encoding: MSB=1 for continuation bytes, MSB=0 for last byte (LSB-first chunks).
    """
    out = bytearray()
    for n in numbers:
        parts = []
        while True:
            parts.append(n & 0x7F)
            n >>= 7
            if n == 0:
                break
        for b in parts[:-1]:
            out.append(0x80 | b)   # continuation
        out.append(parts[-1])      # final
    return bytes(out)

# -----------------------------
# Indexer
# -----------------------------

class Indexer:
    """
    Merges sorted chunk files into a single blockwise-compressed inverted index.

    Outputs:
      - inverted_index.bin : VarByte compressed postings (per block: [docIDs bytes][freqs bytes])
      - lexicon.json       : per-term blocks [offset, sizes, last_doc_id], df
      - page_table.json    : per-doc {"len": document_length}
      - collection_stats.json : {"total_docs": N, "avg_len": avg_len}
    """

    def __init__(self, input_chunks_dir: str, output_dir: str):
        self.input_chunks_dir = input_chunks_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.lexicon: Dict[str, Dict] = {}
        self.page_table: Dict[str, Dict] = {}

        # Collection stats
        self.total_docs: int = 0
        self.total_len: int = 0

        self.postings_fp = None  # binary file handle

    # -------------------------
    # Public pipeline
    # -------------------------

    def run_indexer(self) -> None:
        postings_by_term, page_table = self.merge_chunks()

        # Stats
        self.page_table = page_table
        for _, meta in self.page_table.items():
            self.total_docs += 1
            self.total_len += int(meta.get("len", 0))
        avg_len = (self.total_len / self.total_docs) if self.total_docs else 0.0

        # Paths
        postings_path = os.path.join(self.output_dir, "inverted_index.bin")
        lexicon_path = os.path.join(self.output_dir, "lexicon.json")
        page_table_path = os.path.join(self.output_dir, "page_table.json")
        collection_stats_path = os.path.join(self.output_dir, "collection_stats.json")

        with open(postings_path, "wb") as fp:
            self.postings_fp = fp
            self.write_all_terms_blockwise(postings_by_term)

        with open(lexicon_path, "w", encoding="utf-8") as lexicon_file:
            json.dump(self.lexicon, lexicon_file, indent=2)

        with open(page_table_path, "w", encoding="utf-8") as page_table_file:
            json.dump(self.page_table, page_table_file, indent=2)

        with open(collection_stats_path, "w", encoding="utf-8") as collection_stats_file:
            json.dump(
                {"total_docs": self.total_docs, "avg_len": avg_len},
                collection_stats_file,
                indent=2
            )

        size_mb = os.path.getsize(postings_path) / (1024 * 1024)
        print(
            f"[Indexer] Wrote {len(self.lexicon):,} terms, "
            f"{self.total_docs:,} docs | avg_len={avg_len:.2f} | "
            f"index_size={size_mb:.2f} MB"
        )

    # -------------------------
    # Merge chunk postings (multi-way)
    # -------------------------

    def merge_chunks(self) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, Dict]]:
        """
        True multi-way merge over chunk files.
        Each chunk line: "term docID freq"
        Assumption (parser): each chunk is sorted by (term, docID).

        Returns:
            postings_by_term: {term: [(docID, tf), ...] sorted and de-duplicated}
            page_table: {docID_str: {"len": doc_len}}
        """
        postings_by_term: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        page_table: Dict[str, Dict] = defaultdict(lambda: {"len": 0})
        heap: List[Tuple[str, int, int, int, TextIO]] = []
        open_files: List[TextIO] = []

        for chunk_name in sorted(os.listdir(self.input_chunks_dir)):
            if not chunk_name.endswith(".txt"):
                continue
            f = open(os.path.join(self.input_chunks_dir, chunk_name), "r", encoding="utf-8")
            open_files.append(f)
            first = f.readline()
            if first:
                term, doc_id_str, freq_str = first.strip().split()
                heappush(heap, (term, int(doc_id_str), int(freq_str), len(open_files) - 1, f))

        print(f"[Indexer] Merging {len(heap)} chunk files...")

        with tqdm(desc="Building Index", unit="heap",total=len(postings_by_term)) as pbar:
            current_term = None
            current_postings: List[Tuple[int, int]] = []

            def flush_current_term():
                if not current_postings:
                    return
                current_postings.sort(key=lambda x: x[0])
                merged: List[Tuple[int, int]] = []
                for did, tf in current_postings:
                    if merged and merged[-1][0] == did:
                        merged[-1] = (did, merged[-1][1] + tf)
                    else:
                        merged.append((did, tf))
                postings_by_term[current_term] = merged

            while heap:
                term, doc_id, freq, idx, f = heappop(heap)

                # Track doc length (sum of term freqs)
                page_table[str(doc_id)]["len"] += freq

                if current_term is None:
                    current_term = term
                if term != current_term:
                    flush_current_term()
                    current_term = term
                    current_postings = []

                current_postings.append((doc_id, freq))
                pbar.update(1)

                nxt = f.readline()
                if nxt:
                    t2, did2, fr2 = nxt.strip().split()
                    heappush(heap, (t2, int(did2), int(fr2), idx, f))

            if current_term is not None:
                flush_current_term()

        for f in open_files:
            try:
                f.close()
            except Exception:
                pass

        print(f"[Indexer] Completed merging {len(postings_by_term):,} unique terms.")
        return postings_by_term, page_table

    # -------------------------
    # Blockwise writing
    # -------------------------

    def write_all_terms_blockwise(self, postings_by_term: Dict[str, List[Tuple[int, int]]]) -> None:
        for term, postings in postings_by_term.items():
            if not postings:
                continue
            blocks_meta = self.write_term_as_blocks(postings)
            self.lexicon[term] = {
                "df": len(postings),
                "blocks": blocks_meta,
            }

    def write_term_as_blocks(self, postings: List[Tuple[int, int]]) -> List[Dict]:
        """
        Per block we write VarByte(docIDs) then VarByte(freqs).
        DocIDs are gap-encoded within the block ONLY:
          first docID = ABSOLUTE (global docID),
          others = difference to previous in the same block.
        """
        blocks_meta: List[Dict] = []
        postings.sort(key=lambda x: x[0])  # safety

        for start in range(0, len(postings), BLOCK_POSTINGS):
            block = postings[start:start + BLOCK_POSTINGS]
            doc_ids = [d for d, _ in block]
            freqs   = [tf for _, tf in block]

            # gaps: first absolute, rest local gaps
            gaps = [doc_ids[0]]
            for i in range(1, len(doc_ids)):
                gaps.append(doc_ids[i] - doc_ids[i - 1])

            enc_doc_ids = varbyte_encode(gaps)
            enc_freqs   = varbyte_encode(freqs)

            offset = self.postings_fp.tell()
            self.postings_fp.write(enc_doc_ids)
            self.postings_fp.write(enc_freqs)

            blocks_meta.append({
                "offset": offset,
                "bytes_doc_ids": len(enc_doc_ids),
                "bytes_freqs": len(enc_freqs),
                "last_doc_id": doc_ids[-1],
            })

        return blocks_meta

# -----------------------------
# Script hook
# -----------------------------

def run_indexer(chunks_dir: str, output_dir: str) -> None:
    idx = Indexer(chunks_dir, output_dir)
    idx.run_indexer()
