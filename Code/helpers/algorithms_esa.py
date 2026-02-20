from __future__ import annotations

from helpers.models import EnhancedSuffixArray, LCSResult


def _pick_separator(s: str, t: str) -> str: #NOTE: redundant. Might remove later
    """Pick a separator character that does not occur in either input string."""
    used = set(s) | set(t)
    for codepoint in range(1, 1024):
        candidate = chr(codepoint)
        if candidate not in used:
            return candidate
    raise ValueError("Could not find a separator character not present in input strings.")


def build_suffix_array(text: str) -> list[int]:
    """Build suffix array with prefix-doubling (O(n log n))."""
    n = len(text)
    if n == 0:
        return []

    suffix_array = list(range(n))
    rank = [ord(ch) for ch in text]
    k = 1

    while True:
        suffix_array.sort(key=lambda idx: (rank[idx], rank[idx + k] if idx + k < n else -1))

        new_rank = [0] * n
        for i in range(1, n):
            previous = suffix_array[i - 1]
            current = suffix_array[i]
            prev_key = (rank[previous], rank[previous + k] if previous + k < n else -1)
            curr_key = (rank[current], rank[current + k] if current + k < n else -1)
            new_rank[current] = new_rank[previous] + (curr_key != prev_key)

        rank = new_rank
        if rank[suffix_array[-1]] == n - 1:
            break
        k <<= 1

    return suffix_array


def build_lcp_array(text: str, suffix_array: list[int]) -> list[int]:
    """Build LCP array using Kasai's algorithm (O(n))."""
    n = len(text)
    if n == 0:
        return []

    rank = [0] * n
    for i, suffix_start in enumerate(suffix_array):
        rank[suffix_start] = i

    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            continue
        j = suffix_array[r - 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1

    return lcp


def build_enhanced_suffix_array(s: str, t: str) -> EnhancedSuffixArray:
    """Build Enhanced Suffix Array (SA + LCP) for s and t."""
    # separator = _pick_separator(s, t)
    separator = "\0"  # Using null character as a separator, assuming it doesn't appear in inputs
    joined = s + separator + t
    split_index = len(s)
    suffix_array = build_suffix_array(joined)
    lcp = build_lcp_array(joined, suffix_array)
    return EnhancedSuffixArray(
        joined=joined,
        split_index=split_index,
        suffix_array=suffix_array,
        lcp=lcp,
    )


def query_enhanced_suffix_array(esa: EnhancedSuffixArray) -> LCSResult:
    """All longest common substrings using a prebuilt Enhanced Suffix Array."""
    if not esa.joined:
        return LCSResult(0, set())

    best_length = 0
    best_substrings: set[str] = set()

    for i in range(1, len(esa.suffix_array)):
        left = esa.suffix_array[i - 1]
        right = esa.suffix_array[i]

        left_from_s = left < esa.split_index
        right_from_s = right < esa.split_index
        if left_from_s == right_from_s:
            continue

        candidate = esa.lcp[i]
        if candidate > best_length:
            best_length = candidate
            best_substrings = {esa.joined[right : right + best_length]}
        elif candidate == best_length and best_length > 0:
            best_substrings.add(esa.joined[right : right + best_length])

    if best_length == 0:
        return LCSResult(0, set())

    return LCSResult(best_length, best_substrings)
