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
    """Build suffix array using SA-IS (linear-time for integer alphabets).

    Notes:
        This implementation converts the input string to an integer alphabet and
        appends a unique sentinel (0) that is lexicographically smaller than any
        other symbol. The returned suffix array is for the original string (i.e.
        it does not include the sentinel suffix).
    """

    def _sais(s: list[int], alphabet_size: int) -> list[int]:
        n = len(s)
        if n == 1:
            return [0]

        # S-type: True, L-type: False
        is_s = [False] * n
        is_s[-1] = True  # sentinel
        for i in range(n - 2, -1, -1):
            if s[i] < s[i + 1]:
                is_s[i] = True
            elif s[i] > s[i + 1]:
                is_s[i] = False
            else:
                is_s[i] = is_s[i + 1]

        is_lms = [False] * n
        for i in range(1, n):
            if is_s[i] and not is_s[i - 1]:
                is_lms[i] = True

        lms_positions = [i for i in range(1, n) if is_lms[i]]

        def _bucket_sizes() -> list[int]:
            sizes = [0] * (alphabet_size + 1)
            for c in s:
                sizes[c] += 1
            return sizes

        def _bucket_heads(sizes: list[int]) -> list[int]:
            heads = [0] * (alphabet_size + 1)
            total = 0
            for c in range(alphabet_size + 1):
                heads[c] = total
                total += sizes[c]
            return heads

        def _bucket_tails(sizes: list[int]) -> list[int]:
            tails = [0] * (alphabet_size + 1)
            total = 0
            for c in range(alphabet_size + 1):
                total += sizes[c]
                tails[c] = total - 1
            return tails

        def _induced_sort(sorted_lms: list[int]) -> list[int]:
            sa = [-1] * n
            sizes = _bucket_sizes()

            tails = _bucket_tails(sizes)
            for pos in reversed(sorted_lms):
                c = s[pos]
                sa[tails[c]] = pos
                tails[c] -= 1

            heads = _bucket_heads(sizes)
            for i in range(n):
                j = sa[i] - 1
                if j >= 0 and not is_s[j]:
                    c = s[j]
                    sa[heads[c]] = j
                    heads[c] += 1

            tails = _bucket_tails(sizes)
            for i in range(n - 1, -1, -1):
                j = sa[i] - 1
                if j >= 0 and is_s[j]:
                    c = s[j]
                    sa[tails[c]] = j
                    tails[c] -= 1

            return sa

        # Initial induced sort using LMS positions as-is.
        sa = _induced_sort(lms_positions)

        def _lms_substrings_equal(a: int, b: int) -> bool:
            k = 0
            while True:
                if s[a + k] != s[b + k] or is_lms[a + k] != is_lms[b + k]:
                    return False
                # If either substring reaches the sentinel, they are equal only
                # if both do so at the same time.
                if a + k == n - 1 or b + k == n - 1:
                    return (a + k == n - 1) and (b + k == n - 1)
                if k > 0 and is_lms[a + k] and is_lms[b + k]:
                    return True
                k += 1

        # Name LMS substrings in SA order.
        lms_in_sa = [pos for pos in sa if pos != -1 and is_lms[pos]]
        name_at = [-1] * n
        current_name = 0
        prev = -1
        for pos in lms_in_sa:
            if prev == -1:
                name_at[pos] = current_name
                prev = pos
                continue
            if not _lms_substrings_equal(prev, pos):
                current_name += 1
            name_at[pos] = current_name
            prev = pos

        num_names = current_name + 1
        m = len(lms_positions)

        # Build reduced problem string in original LMS order.
        reduced = [name_at[pos] for pos in lms_positions]

        if num_names == m:
            # All LMS substrings are unique; build SA of reduced string directly.
            sa1 = [0] * m
            for i in range(m):
                sa1[reduced[i]] = i
        else:
            sa1 = _sais(reduced, num_names - 1)

        # Map reduced SA order back to LMS positions.
        sorted_lms = [lms_positions[i] for i in sa1]
        return _induced_sort(sorted_lms)

    n = len(text)
    if n == 0:
        return []

    # Map characters to a dense integer alphabet: 1..sigma; reserve 0 for sentinel.
    alphabet = sorted(set(text))
    to_int = {ch: i + 1 for i, ch in enumerate(alphabet)}
    s_int = [to_int[ch] for ch in text]
    s_int.append(0)  # sentinel

    sa = _sais(s_int, len(alphabet))
    sentinel_index = n
    return [i for i in sa if i != sentinel_index]


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
            pos = min(left, right) # bug fix
            best_substrings = {esa.joined[pos : pos + best_length]}
        elif candidate == best_length and best_length > 0:
            pos = min(left, right) # bug fix
            best_substrings.add(esa.joined[pos : pos + best_length])

    if best_length == 0:
        return LCSResult(0, set())

    return LCSResult(best_length, best_substrings)
