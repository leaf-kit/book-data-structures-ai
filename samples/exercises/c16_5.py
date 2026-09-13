"""문제 16.5. 트리 all-reduce. 깊이 log p 의 reduce 와 broadcast.
링과 바이트를 견준다.
"""
import numpy as np

from dsai.collective import allreduce_ring, bytes_per_machine


def allreduce_tree(shards):
    """이진 트리로 모으고 (reduce) 다시 내려보낸다 (broadcast). (합,
    기계 하나의 최대 링크 바이트, 단계).
    """
    p = len(shards)
    buf = [s.copy() for s in shards]
    nbytes = shards[0].nbytes
    per_link, steps = 0, 0
    step = 1
    while step < p:                                  # reduce: 짝이 홀에게 보낸다
        for i in range(0, p - step, 2 * step):
            buf[i] += buf[i + step]
            per_link = max(per_link, nbytes)
        step *= 2
        steps += 1
    step //= 2
    while step >= 1:                                 # broadcast: 거꾸로
        for i in range(0, p - step, 2 * step):
            buf[i + step] = buf[i].copy()
        step //= 2
        steps += 1
    return buf[0], per_link * steps, steps


def test_tree_equals_ring_and_moves_more_than_ring():
    rng = np.random.default_rng(0)
    for p in (2, 4, 8, 16):
        shards = [rng.standard_normal(4096) for _ in range(p)]
        t_tree, b_tree, s_tree = allreduce_tree(shards)
        t_ring, b_ring, s_ring = allreduce_ring(shards)
        assert np.allclose(t_tree, sum(shards)) and np.allclose(t_ring, sum(shards))
        assert s_tree == 2 * int(np.log2(p))
        if p >= 4:
            assert b_tree > b_ring       # 트리는 단계마다 벡터 전부, 링은 조각
        assert b_ring == bytes_per_machine(4096, 8, p, "ring")
