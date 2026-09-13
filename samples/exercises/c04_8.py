"""문제 4.8. 압축이 돌려주는 재배치 표로 밖에서 든 칸 번호를 고친다."""
import numpy as np

from dsai.linked import IndexList


def compact_with_map(lst: IndexList) -> np.ndarray:
    """압축하고, 옛 칸 번호에서 새 칸 번호로 가는 표를 돌려준다. 없어진 칸은 -1."""
    order = lst.order()
    remap = np.full(len(lst.val), -1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    lst.compact()
    return remap


def test_remap_fixes_external_handles():
    lst = IndexList(16)
    a = lst.append(10.0)
    lst.append(20.0)
    b = lst.insert_after(a, 15.0)      # 순서 a, b, ...
    handles = np.array([a, b])
    remap = compact_with_map(lst)
    new = remap[handles]
    assert list(lst.val[new]) == [10.0, 15.0]
    assert np.array_equal(new, [0, 1])
