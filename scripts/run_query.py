from typing import List
from search_system.query.query import QueryStartupContext, run_query, LIST_CACHE, QUERY_STARTUP_CONTEXT
from search_system.shared.config import INDEX_DIR, DEFAULT_TOPK


def main() -> None:
    global QUERY_STARTUP_CONTEXT
    input_dir: str = INDEX_DIR
    topk = DEFAULT_TOPK
    QUERY_STARTUP_CONTEXT = QueryStartupContext(input_dir)

    print("Type your query below, or '+exit' to quit.\n")

    while True:
        query: str = input("Enter query: ").strip()
        if query.lower() in {"+exit"}:
            print("\nExiting search engine.")
            break

        query_mode = input("Conjunctive or Disjunctive [MaxScore: or, Block Max WAND: bwand-or]? [and/or/bwand-or]: ").strip().lower()
        if query_mode not in ("and", "or", "bwand-or"):
            print("Invalid choice. Please type 'and' or 'or'.\n")
            continue

        try:
            results: List[str] = run_query(QUERY_STARTUP_CONTEXT, query, query_mode, topk)

            if not results:
                print("\nNo results found.\n")
                continue

            print("\nResults:")
            for i, (doc_id, score) in enumerate(results, start=1):
                print(f"{i}) DocID: {doc_id}  Score: {score:.6f}")

            print(f"\n{LIST_CACHE.stats()}\n")

        except Exception as e:
            print(f"\nAn error occurred: {e}\n")


if __name__ == "__main__":
    main()