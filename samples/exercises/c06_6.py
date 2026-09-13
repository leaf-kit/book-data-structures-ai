"""문제 6.6. 첫 글자로 나눈 스물여섯 개의 작은 정렬 배열."""
import bisect

from dsai.vocab import SortedVocab


class BucketedVocab:
    def __init__(self, vocab: dict[str, int]):
        self.buckets: dict[str, tuple[list[str], list[int]]] = {}
        for k in sorted(vocab):
            keys, ids = self.buckets.setdefault(k[0], ([], []))
            keys.append(k)
            ids.append(vocab[k])

    def lookup(self, token: str) -> int:
        b = self.buckets.get(token[0])
        if b is None:
            return -1
        keys, ids = b
        i = bisect.bisect_left(keys, token)
        return ids[i] if i < len(keys) and keys[i] == token else -1


def test_bucketed_matches_sorted():
    words = ["apple", "avocado", "banana", "cherry", "date"]
    vocab = {w: i for i, w in enumerate(words)}
    a, b = SortedVocab(vocab), BucketedVocab(vocab)
    for w in list(vocab) + ["zebra", "app"]:
        assert a.lookup(w) == b.lookup(w)
