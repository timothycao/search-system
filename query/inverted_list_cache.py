"""
inverted_list_cache.py
----------------------
Implements a simple Least Recently Used (LRU) cache for InvertedList objects.
Used by query2.py to store frequently accessed term posting lists in memory.
"""

from collections import OrderedDict


class InvertedListCache:
    """LRU cache for storing InvertedList objects."""

    def __init__(self, capacity: int = 10):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, term: str):
        """Return cached InvertedList if present and mark it as recently used."""
        if term not in self.cache:
            self.misses += 1
            return None
        self.cache.move_to_end(term)
        self.hits += 1
        return self.cache[term]

    def put(self, term: str, inverted_list):
        """Insert a new InvertedList into cache, evicting least recently used if full."""
        if term in self.cache:
            self.cache.move_to_end(term)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # remove oldest (LRU)
            self.cache[term] = inverted_list

    def stats(self) -> str:
        """Return cache statistics for debugging."""
        return f"Cache: {len(self.cache)}/{self.capacity} | Hits: {self.hits} | Misses: {self.misses}"

    def __contains__(self, term: str):
        return term in self.cache

    def __len__(self):
        return len(self.cache)
