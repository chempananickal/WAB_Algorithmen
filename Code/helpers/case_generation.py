from __future__ import annotations

import random

from helpers.models import BenchmarkCase


def random_string(rng: random.Random, length: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def mutate_string(rng: random.Random, source: str, mutation_rate: float, alphabet: str) -> str:
    if not source:
        return source
    chars = list(source)
    mutations = max(1, int(round(len(chars) * mutation_rate)))
    indices = rng.sample(range(len(chars)), k=min(mutations, len(chars)))
    for idx in indices:
        original = chars[idx]
        alternatives = [c for c in alphabet if c != original]
        chars[idx] = rng.choice(alternatives) if alternatives else original
    return "".join(chars)

def mutate_string_edges(
    rng: random.Random,
    source: str,
    mutation_rate: float,
    alphabet: str,
    prefer_edge: str | None = None,
) -> str:
    """Mutate random characters within a start/end edge window."""
    if not source:
        return source

    n = len(source)
    window_size = max(1, int(round(n * mutation_rate)))

    if prefer_edge is None:
        edge = rng.choice(("start", "end"))
    else:
        edge = prefer_edge

    if edge == "start":
        window = list(range(0, min(n, window_size)))
    else:
        window = list(range(max(0, n - window_size), n))

    # mutate only some positions in that window (not all)
    k = rng.randint(1, len(window))
    indices = rng.sample(window, k=k)

    chars = list(source)
    for idx in indices:
        original = chars[idx]
        alternatives = [c for c in alphabet if c != original]
        chars[idx] = rng.choice(alternatives) if alternatives else original
    return "".join(chars)

def insert_substring(rng: random.Random, host: str, fragment: str) -> str:
    position = rng.randint(0, len(host))
    return host[:position] + fragment + host[position:]


def generate_cases(lengths: list[int], cases_per_length: int, seed: int) -> list[BenchmarkCase]:
    """Generate reproducible test scenarios across all requested lengths."""
    rng = random.Random(seed)
    dna_alphabet = "ACGT" # Adenine, Cytosine, Guanine, Thymine. The DNA bases
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"
    disjoint_a = "abcdefghijklm"
    disjoint_b = "nopqrstuvwxyz"

    scenarios = [
        "random_uniform", # completely random strings over full alphabet
        "mutated_implant", # a common motif implanted in both strings with mutations, over the dna alphabet
        "repetitive_with_noise", # strings formed by repeating a pattern with some mutations, over the dna alphabet
        "near_identical", # strings that are nearly identical with minor mutations preferentially at edges over the full alphabet
        "disjoint_alphabet", # strings from completely disjoint alphabets to test worst-case LCS of length 0
    ]

    cases: list[BenchmarkCase] = []
    for length in lengths:
        for case_no in range(1, cases_per_length + 1):
            for scenario in scenarios:
                case_id = f"{scenario}_n{length}_c{case_no}"

                if scenario == "random_uniform": # might find small LCSs by pure coincidence
                    s = random_string(rng, length, full_alphabet)
                    t = random_string(rng, length, full_alphabet)
                    metadata = {}

                elif scenario == "mutated_implant": # will find a small-ish LCS (the mautated motif) within random noise
                    motif_length = max(8, length // 4)
                    motif = random_string(rng, motif_length, dna_alphabet)
                    mutated = mutate_string(rng, motif, mutation_rate=0.12, alphabet=dna_alphabet)

                    base_s = random_string(rng, length - motif_length, dna_alphabet)
                    base_t = random_string(rng, length - motif_length, dna_alphabet)
                    s = insert_substring(rng, base_s, motif)
                    t = insert_substring(rng, base_t, mutated)
                    metadata = {
                        "motif_length": motif_length,
                        "mutation_rate": 0.12,
                    }

                elif scenario == "repetitive_with_noise": # will generate a large number of small LCSs
                    pattern_length = max(3, length // 20)
                    pattern = random_string(rng, pattern_length, dna_alphabet)
                    repeats = max(1, length // pattern_length)
                    s_base = (pattern * repeats)[:length]
                    t_base = (pattern * repeats)[:length]
                    s = mutate_string(rng, s_base, mutation_rate=0.06, alphabet=dna_alphabet)
                    t = mutate_string(rng, t_base, mutation_rate=0.08, alphabet=dna_alphabet)
                    metadata = {
                        "pattern_length": pattern_length,
                        "mutation_rates": "0.06/0.08",
                    }

                elif scenario == "near_identical": # will generate a long LCS
                    s = random_string(rng, length, full_alphabet)
                    t = mutate_string_edges(rng, s, mutation_rate=0.02, alphabet=full_alphabet)
                    metadata = {"mutation_rate": 0.02}

                elif scenario == "disjoint_alphabet": # will generate LCS of length 0, testing worst-case behavior
                    s = random_string(rng, length, disjoint_a)
                    t = random_string(rng, length, disjoint_b)
                    metadata = {}

                else:
                    raise ValueError(f"Unknown scenario: {scenario}")

                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        scenario=scenario,
                        length=length,
                        s=s,
                        t=t,
                        metadata=metadata,
                    )
                )
    return cases
