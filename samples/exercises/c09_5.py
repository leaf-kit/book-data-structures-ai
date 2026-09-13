"""문제 9.5. 뿌리의 자식만 256 칸 배열로. 첫 걸음이 해시에서 첨자가 된다."""
from dsai.trie import NIL, Trie


class RootArrayTrie(Trie):
    def __init__(self):
        super().__init__()
        self.root_child = [NIL] * 256

    def insert(self, word: bytes, token_id: int) -> None:
        super().insert(word, token_id)
        self.root_child[word[0]] = self.child[(0, word[0])]

    def longest_match(self, text: bytes, start: int) -> tuple[int, int]:
        node = self.root_child[text[start]]             # 첫 걸음은 첨자
        if node == NIL:
            return NIL, 0
        best, best_len = (self.token[node], 1) if self.token[node] != NIL else (NIL, 0)
        for i in range(start + 1, len(text)):
            node = self.child.get((node, text[i]), NIL)
            if node == NIL:
                break
            if self.token[node] != NIL:
                best, best_len = self.token[node], i - start + 1
        return best, best_len


def test_root_array_matches_plain():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"abc", 258: b"bcd", 259: b" the"})
    a, b = Trie(), RootArrayTrie()
    for t, w in vocab.items():
        a.insert(w, t)
        b.insert(w, t)
    text = b"abcd the abc bcd x"
    assert a.tokenize(text) == b.tokenize(text)
    assert b"".join(vocab[t] for t in b.tokenize(text)) == text
