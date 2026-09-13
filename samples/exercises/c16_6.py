"""문제 16.6. 내용 정의 청킹.
구르는 해시로 경계를 정하면 한 바이트 삽입에도 블록 대부분이 그대로다.
"""
import numpy as np

from dsai.merkle import ContentStore, block_hashes


def content_defined_chunks(data: bytes, window: int = 16, mask: int = 0xFFF,
                           min_len: int = 512):
    """구르는 합이 마스크에 걸리면 경계. 평균 블록이 mask+1 바이트 안팎이다."""
    bounds = [0]
    h, pw = 0, pow(31, window, 1 << 32)
    for i, byte in enumerate(data):
        h = (h * 31 + byte) & 0xFFFFFFFF                    # 다항식 구르는 해시
        if i >= window:
            h = (h - data[i - window] * pw) & 0xFFFFFFFF   # 창 밖 바이트를 뺀다
        if i - bounds[-1] >= min_len and (h & mask) == 0:
            bounds.append(i + 1)
    if bounds[-1] != len(data):
        bounds.append(len(data))
    return [data[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def store_chunks(store, chunks):
    added = 0
    for c in chunks:
        before = store.stored
        store.put(c, len(c))
        added += store.stored - before
    return added


def test_insert_one_byte_fixed_blocks_shift_but_content_defined_do_not():
    rng = np.random.default_rng(1)
    data = bytes(rng.integers(0, 256, 200_000, dtype=np.uint8))
    shifted = data[:1000] + b"X" + data[1000:]
    fixed = ContentStore()
    fixed.put(data, 4096)
    before = fixed.stored
    fixed.put(shifted, 4096)
    fixed_added = fixed.stored - before           # 삽입 뒤의 블록이 전부 밀린다
    cdc = ContentStore()
    store_chunks(cdc, content_defined_chunks(data))
    cdc_added = store_chunks(cdc, content_defined_chunks(shifted))
    assert fixed_added > len(data) * 0.9
    assert cdc_added < len(data) * 0.1
    assert len(block_hashes(data, 4096)) == 49
