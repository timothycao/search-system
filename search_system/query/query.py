from typing import Dict, List, Tuple, Optional
import heapq
import time

from search_system.query.inverted_list import InvertedList, INF_DOCID
from search_system.query.inverted_list_cache import InvertedListCache
from search_system.query.query_startup_context import QueryStartupContext

# Global in memory cache instance
LIST_CACHE = InvertedListCache()
QUERY_STARTUP_CONTEXT = None  

# ---------------------------------------------------------
#   Inverted List Management
# ---------------------------------------------------------

def openList(term: str,
             lexicon: Dict[str, Dict],
             index_path: str,
             page_table: Dict[str, Dict],
             N: int,
             avg_len: float,
             k1: float,
             b: float) -> Optional[InvertedList]:
    """
    Open an inverted list for a given term and return an InvertedList object.
    Returns None if term not found in lexicon.
    """
    cached = LIST_CACHE.get(term)
    if cached is not None:
        cached.reset()
        return cached

    if term not in lexicon:
        return None

    term_meta = lexicon[term]
    list_pointer = InvertedList(
        term=term,
        term_meta=term_meta,
        index_path=index_path,
        page_table=page_table,
        N=N,
        avg_len=avg_len,
        k1=k1,
        b=b
    )

    LIST_CACHE.put(term, list_pointer)
    return list_pointer


def closeList(lp: Optional[InvertedList]) -> None:
    """Close an inverted list pointer (no-op for binary index)."""
    if lp is None:
        return
    if lp.term in LIST_CACHE:
        return
    lp.closeList()

# ---------------------------------------------------------
#   DAAT Traversal and Ranking
# ---------------------------------------------------------

def check_push_topk(heap: List[Tuple[float, int]], doc_id: int, score: float, k: int) -> None:
    """
    Maintain a min-heap of size at most k with the highest scoring documents.
    """
    if len(heap) < k:
        heapq.heappush(heap, (score, doc_id))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, doc_id))


def min_score_in_heap(heap: List[Tuple[float, int]], k: int) -> float:
    """Return the smallest score in the current heap (or 0.0 if heap not full)."""
    if len(heap) < k or not heap:
        return 0.0
    return heap[0][0]


def daat_conjunctive(lists, k):
    if not lists:
        return []

    # Only perform pre-processing sort if multiple lists are present
    if len(lists) > 1:
        lists.sort(key=lambda lp: lp.df)

    heap = []

    while True:
        # Skip any exhausted lists
        active_lists = [lp for lp in lists if lp.doc_id < INF_DOCID]
        if not active_lists:
            break

        target = max(lp.doc_id for lp in active_lists)
        if target >= INF_DOCID:
            break

        for lp in active_lists:
            if lp.doc_id < target:
                lp.nextGEQ(target)

        if all(lp.doc_id == target for lp in active_lists):
            score = 0.0
            for lp in active_lists:
                # Only score if still within bounds
                if lp.doc_id < INF_DOCID:
                    score += lp.getScore(target)

            check_push_topk(heap, target, score, k)

            for lp in active_lists:
                lp.nextGEQ(target + 1)

    return sorted([(doc, sc) for sc, doc in heap], key=lambda x: (-x[1], x[0]))



def daat_disjunctive_maxscore(lists: List[InvertedList], k: int) -> List[Tuple[int, float]]:
    """
    Disjunctive (OR) query traversal using the MaxScore optimization.
    """
    if not lists:
        return []

    lists.sort(key=lambda lp: lp.max_score, reverse=True)
    heap: List[Tuple[float, int]] = []

    while True:
        current_doc = min(lp.doc_id for lp in lists)
        if current_doc >= INF_DOCID:
            break

        # Compute upper bound for this doc
        upper_bound = 0.0
        for lp in lists:
            if lp.doc_id <= current_doc:
                upper_bound += lp.max_score

        threshold = min_score_in_heap(heap, k)
        if upper_bound < threshold:
            for lp in lists:
                if lp.doc_id == current_doc:
                    lp.nextGEQ(current_doc + 1)
            continue

        score = 0.0
        for lp in lists:
            if lp.doc_id == current_doc:
                score += lp.getScore(current_doc)

        check_push_topk(heap, current_doc, score, k)

        for lp in lists:
            if lp.doc_id == current_doc:
                lp.nextGEQ(current_doc + 1)

    ranked = [(doc_id, score) for score, doc_id in heap]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked



def daat_disjunctive_blockmax_wand(lists: List[InvertedList], k: int) -> List[Tuple[int, float]]:
    """
    Disjunctive (OR) query traversal using Block-Max WAND optimization.
    Uses per-block BM25 upper bounds to skip blocks that cannot beat the current threshold.
    """
    if not lists:
        return []

    # Sort lists by descending max_score (impact ordering)
    lists.sort(key=lambda l: l.max_score, reverse=True)

    topk: List[Tuple[float, int]] = []   # (score, docID)
    threshold = 0.0

    while True:
        pivot_doc = min(l.doc_id for l in lists)
        if pivot_doc >= INF_DOCID:
            break

        # Compute block-level upper bound 
        ub = sum(l.curr_block_max() for l in lists)
        if ub < threshold:
            # Skip smallest-docID list’s current block
            smallest = min(lists, key=lambda l: l.doc_id)
            smallest.advance_to_next_block()
            continue
        

        # Score candidate doc
        score = 0.0
        for l in lists:
            if l.doc_id == pivot_doc:
                score += l.getScore(pivot_doc)

        if score > 0.0:
            if len(topk) < k:
                heapq.heappush(topk, (score, pivot_doc))
            elif score > topk[0][0]:
                heapq.heapreplace(topk, (score, pivot_doc))
            threshold = topk[0][0]

        # Advance lists that matched pivot
        for l in lists:
            if l.doc_id == pivot_doc:
                l.nextGEQ(pivot_doc + 1)

    return sorted([(doc_id, score) for score, doc_id in topk], key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------
#   Query Entry Point
# ---------------------------------------------------------

def run_query(startup_context: QueryStartupContext,
              query: str,
              mode: str = "and",
              top_k: int = 10,
              k1: float = 1.2,
              b: float = 0.75) -> List[Tuple[int, float]]:
    """
    Execute a ranked retrieval query using BM25.
    mode: "and" for conjunctive, "or" for disjunctive (MaxScore & Block Max WAND)
    """
    time0 = time.perf_counter()

    lexicon = startup_context.lexicon
    page_table = startup_context.page_table
    index_path = startup_context.index_path
    total_docs = startup_context.total_docs
    avg_doc_len = startup_context.avg_len

    # Tokenize query (simple whitespace-based)
    terms = [t.strip().lower() for t in query.split() if t.strip()]
    lists: List[InvertedList] = []

    time1 = time.perf_counter()
    gather_time = time1 - time0

    # Open lists for each query term
    for term in terms:
        lp = openList(term, lexicon, index_path, page_table, total_docs, avg_doc_len, k1, b)
        if lp is not None and lp.doc_id < INF_DOCID:
            lists.append(lp)

    if not lists:
        return []
    
    time2 = time.perf_counter()
    open_time = time2 - time1

    # Run DAAT traversal
    if mode == "and":
        results = daat_conjunctive(lists, top_k)
    elif mode == "or":
        results = daat_disjunctive_maxscore(lists, top_k)
    elif mode == "bwand-or":
        results = daat_disjunctive_blockmax_wand(lists, top_k)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    time3 = time.perf_counter()
    traversal_time = time3 - time2

    # Close all lists
    for lp in lists:
        closeList(lp)

    time4 = time.perf_counter()
    total_time = time4 - time0

    print(f"\n[Timing]")
    print(f"  Data gathering : {gather_time:.4f} s")
    print(f"  Opening lists  : {open_time:.4f} s")
    print(f"  Traversal      : {traversal_time:.4f} s")
    print(f"  Total          : {total_time:.4f} s\n")

    return results
