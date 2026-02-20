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


def insert_substring(rng: random.Random, host: str, fragment: str) -> str:
    position = rng.randint(0, len(host))
    return host[:position] + fragment + host[position:]


def generate_cases(lengths: list[int], cases_per_length: int, seed: int) -> list[BenchmarkCase]:
    """Generate reproducible test scenarios across all requested lengths."""
    rng = random.Random(seed)
    dna_alphabet = "ACGT"
    full_alphabet = "abcdefghijklmnopqrstuvwxyz"
    disjoint_a = "abcdefghijklm"
    disjoint_b = "nopqrstuvwxyz"

    scenarios = [
        "random_uniform",
        "mutated_implant",
        "repetitive_with_noise",
        "near_identical",
        "disjoint_alphabet",
    ]

    cases: list[BenchmarkCase] = []
    for length in lengths:
        for case_no in range(1, cases_per_length + 1):
            for scenario in scenarios:
                case_id = f"{scenario}_n{length}_c{case_no}"

                if scenario == "random_uniform":
                    s = random_string(rng, length, full_alphabet)
                    t = random_string(rng, length, full_alphabet)
                    metadata = {}

                elif scenario == "mutated_implant":
                    motif_length = max(8, length // 4)
                    motif = random_string(rng, motif_length, dna_alphabet)
                    mutated = mutate_string(rng, motif, mutation_rate=0.12, alphabet=dna_alphabet)

                    base_s = random_string(rng, length, dna_alphabet)
                    base_t = random_string(rng, length, dna_alphabet)
                    s = insert_substring(rng, base_s, motif)
                    t = insert_substring(rng, base_t, mutated)
                    metadata = {
                        "motif_length": motif_length,
                        "mutation_rate": 0.12,
                    }

                elif scenario == "repetitive_with_noise":
                    pattern_length = max(3, length // 20)
                    pattern = random_string(rng, pattern_length, dna_alphabet)
                    repeats = max(1, length // pattern_length)
                    s_base = (pattern * repeats)[:length]
                    t_base = (pattern[::-1] * repeats)[:length]
                    s = mutate_string(rng, s_base, mutation_rate=0.06, alphabet=dna_alphabet)
                    t = mutate_string(rng, t_base, mutation_rate=0.08, alphabet=dna_alphabet)
                    metadata = {
                        "pattern_length": pattern_length,
                        "mutation_rates": "0.06/0.08",
                    }

                elif scenario == "near_identical":
                    s = random_string(rng, length, full_alphabet)
                    t = mutate_string(rng, s, mutation_rate=0.02, alphabet=full_alphabet)
                    metadata = {"mutation_rate": 0.02}

                elif scenario == "disjoint_alphabet":
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
