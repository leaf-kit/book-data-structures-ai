"""7장. 학습 데이터 중복 제거. 세 구조를 한 파이프라인으로.

정확히 같은 문서는 해시 하나로 거르고,
거의 같은 문서는 MinHash 서명과 LSH 로 후보를 모아 확인한다.
블룸 필터는 이미 본 정확 해시를 기억하는 데 쓴다.
문서 수십억 개의 해시를 표 하나에 다 넣을 수 없기 때문이다.
집합은 단어 두 개짜리 조각으로 만든다. 문자 조각은 짧은 문서끼리도 우연히 많이 겹친다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dsai.bloom import BloomFilter
from dsai.minhash import LSHIndex, jaccard, minhash, word_shingles


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def exact_key(text: str) -> int:
    return hash(normalize(text)) & 0xFFFFFFFFFFFF


@dataclass
class DedupStats:
    seen: int = 0
    exact_dups: int = 0
    near_candidates: int = 0
    near_dups: int = 0
    kept: list[int] = field(default_factory=list)


class DedupPipeline:
    """알고리즘 7.3. 정확 중복은 블룸 필터로, 근사 중복은 LSH 로."""

    def __init__(self, expected: int, k_sig: int = 64, bands: int = 16,
                 threshold: float = 0.7):
        bits = expected * 16
        self.bloom = BloomFilter(bits, 8)
        self.lsh = LSHIndex(bands, k_sig // bands)
        self.k = k_sig
        self.threshold = threshold
        self.shingle_of: dict[int, set[int]] = {}
        self.stats = DedupStats()

    def offer(self, doc_id: int, text: str) -> bool:
        """문서를 넣는다. 남기면 True, 중복이면 False."""
        self.stats.seen += 1
        key = np.array([exact_key(text)], dtype=np.int64)
        if self.bloom.contains(key)[0]:
            self.stats.exact_dups += 1
            return False
        self.bloom.add(key)

        sh = word_shingles(normalize(text))
        sig = minhash(sh, self.k)
        cands = self.lsh.candidates(sig)
        self.stats.near_candidates += len(cands)
        for c in cands:
            if jaccard(sh, self.shingle_of[c]) >= self.threshold:
                self.stats.near_dups += 1
                return False
        self.lsh.insert(doc_id, sig)
        self.shingle_of[doc_id] = sh
        self.stats.kept.append(doc_id)
        return True
