"""문제 9.8. 따옴표 문자열 형식. 제약이 헐거우면 트라이가 지나는 노드가 는다."""
from dsai.constrained import DEAD, DFA, allowed_by_scan, allowed_by_trie, number_dfa
from dsai.trie import Trie


def string_dfa() -> DFA:
    """'"' 뒤에 따옴표 아닌 바이트들, 다시 '"'. 상태 0 시작, 1 안, 2 닫힘."""
    d = DFA(3, 0, {2})
    d.add(0, b'"', 1)
    d.add(1, bytes(b for b in range(256) if b != 34), 1)
    d.add(1, b'"', 2)
    return d


def visited_nodes(dfa: DFA, trie: Trie, state: int) -> int:
    n, stack = 0, [(0, state)]
    while stack:
        node, s = stack.pop()
        n += 1
        for b, child in trie.children_of(node):
            t = dfa.delta[s, b]
            if t != DEAD:
                stack.append((child, t))
    return n


def test_loose_grammar_visits_more():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"12", 258: b"1.5", 259: b'"a', 260: b'a"'})
    trie = Trie()
    for t, w in vocab.items():
        trie.insert(w, t)
    s, nd = string_dfa(), number_dfa()
    for st in range(3):
        assert allowed_by_scan(s, vocab, st) == allowed_by_trie(s, trie, st)
    assert len(allowed_by_trie(s, trie, 1)) > 250          # 안쪽 상태는 거의 다 허용
    assert visited_nodes(s, trie, 1) > 10 * visited_nodes(nd, trie, 1)
