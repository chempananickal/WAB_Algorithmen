from __future__ import annotations

from helpers.models import LCSResult, SAM


def build_suffix_automaton(s: str) -> SAM:
    """Build a suffix automaton from s."""
    transitions: list[dict[str, int]] = [{}]
    suffix_link: list[int] = [-1]
    state_length: list[int] = [0]
    last = 0

    def extend(ch: str) -> None:
        nonlocal last
        cur = len(transitions)
        transitions.append({})
        suffix_link.append(0)
        state_length.append(state_length[last] + 1)

        p = last
        while p != -1 and ch not in transitions[p]:
            transitions[p][ch] = cur
            p = suffix_link[p]

        if p == -1:
            suffix_link[cur] = 0
        else:
            q = transitions[p][ch]
            if state_length[p] + 1 == state_length[q]:
                suffix_link[cur] = q
            else:
                clone = len(transitions)
                transitions.append(dict(transitions[q]))
                suffix_link.append(suffix_link[q])
                state_length.append(state_length[p] + 1)

                while p != -1 and transitions[p].get(ch) == q:
                    transitions[p][ch] = clone
                    p = suffix_link[p]

                suffix_link[q] = clone
                suffix_link[cur] = clone

        last = cur

    for char in s:
        extend(char)

    return SAM(transitions=transitions, suffix_link=suffix_link, state_length=state_length)


def query_suffix_automaton(automaton: SAM, t: str) -> LCSResult:
    """All longest common substrings using a prebuilt suffix automaton."""
    if not t:
        return LCSResult(0, set())

    transitions = automaton.transitions
    suffix_link = automaton.suffix_link
    state_length = automaton.state_length

    state = 0
    current_length = 0
    best_length = 0
    best_end_position = -1
    best_substrings: set[str] = set()

    for index, char in enumerate(t):
        if char in transitions[state]:
            state = transitions[state][char]
            current_length += 1
        else:
            while state != -1 and char not in transitions[state]:
                state = suffix_link[state]

            if state == -1:
                state = 0
                current_length = 0
                continue

            current_length = state_length[state] + 1
            state = transitions[state][char]

        if current_length > best_length:
            best_length = current_length
            best_end_position = index
            start = best_end_position - best_length + 1
            best_substrings = {t[start : best_end_position + 1]}
        elif current_length == best_length and best_length > 0:
            start = index - best_length + 1
            best_substrings.add(t[start : index + 1])

    if best_length == 0:
        return LCSResult(0, set())

    return LCSResult(best_length, best_substrings)
