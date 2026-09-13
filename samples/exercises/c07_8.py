"""문제 7.8. 확인 단계를 조각 집합 대신 서명 비교로. 메모리가 서명뿐이 된다."""
import numpy as np

from dsai.dedup import DedupPipeline, exact_key, normalize
from dsai.minhash import estimate_jaccard, minhash, word_shingles


class SignatureOnlyPipeline(DedupPipeline):
    def __init__(self, expected: int, k_sig: int = 64, bands: int = 16,
                 threshold: float = 0.7):
        super().__init__(expected, k_sig, bands, threshold)
        self.sig_of: dict[int, np.ndarray] = {}

    def offer(self, doc_id: int, text: str) -> bool:
        self.stats.seen += 1
        key = np.array([exact_key(text)], dtype=np.int64)
        if self.bloom.contains(key)[0]:
            self.stats.exact_dups += 1
            return False
        self.bloom.add(key)
        sig = minhash(word_shingles(normalize(text)), self.k)
        cands = self.lsh.candidates(sig)
        self.stats.near_candidates += len(cands)
        for c in cands:
            if estimate_jaccard(sig, self.sig_of[c]) >= self.threshold:  # 서명 비교
                self.stats.near_dups += 1
                return False
        self.lsh.insert(doc_id, sig)
        self.sig_of[doc_id] = sig
        self.stats.kept.append(doc_id)
        return True


def test_signature_only_catches_near_dups():
    rng = np.random.default_rng(5)
    vocab = [f"w{i}" for i in range(2000)]
    base = [" ".join(rng.choice(vocab, 60)) for _ in range(300)]
    docs = list(base)
    for i in range(100):
        words = base[i].split()
        words[rng.integers(60)] = "changed"
        docs.append(" ".join(words))
    p = SignatureOnlyPipeline(expected=len(docs))
    for i, d in enumerate(docs):
        p.offer(i, d)
    assert p.stats.near_dups >= 95
    assert len(p.stats.kept) <= 305
