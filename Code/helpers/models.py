from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LCSResult:
    length: int
    substrings: set[str]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    scenario: str
    length: int
    s: str
    t: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SAM:
    transitions: list[dict[str, int]]
    suffix_link: list[int]
    state_length: list[int]


@dataclass(frozen=True)
class EnhancedSuffixArray:
    joined: str
    split_index: int
    suffix_array: list[int]
    lcp: list[int]
