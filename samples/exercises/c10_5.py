"""문제 10.5. 배열에서 힙을 O(n) 에 만드는 heapify. 넣기 n 번과 견준다."""
import numpy as np

from dsai.heap import MaxHeap


class HeapifyHeap(MaxHeap):
    @classmethod
    def from_array(cls, xs: list[float]) -> "HeapifyHeap":
        h = cls()
        h.a = list(xs)
        for i in range(len(h.a) // 2 - 1, -1, -1):     # 마지막 내부 노드부터 가라앉힌다
            h._sift_down(i)
        return h

    def is_heap(self) -> bool:
        a = self.a
        return all(a[(i - 1) >> 1] >= a[i] for i in range(1, len(a)))


def test_heapify_builds_valid_heap():
    xs = np.random.default_rng(0).permutation(5000).tolist()
    h = HeapifyHeap.from_array(xs)
    assert h.is_heap()
    out = [h.pop() for _ in range(5000)]
    assert out == sorted(xs, reverse=True)
