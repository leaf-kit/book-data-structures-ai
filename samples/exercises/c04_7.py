"""문제 4.7. 인덱스 링크 리스트에 삭제를 더한다."""
import numpy as np

from dsai.linked import IndexList


class DeletableList(IndexList):
    def delete_after(self, i: int) -> None:
        """칸 i 의 다음 노드를 빼서 자유 리스트로 되돌린다."""
        j = int(self.nxt[i])
        if j < 0:
            return
        self.nxt[i] = self.nxt[j]
        if self.tail == j:
            self.tail = i
        self.free.append(j)
        self.size -= 1


def test_delete_keeps_order_and_reuses_slot():
    lst = DeletableList(8)
    for v in (1.0, 2.0, 3.0, 4.0):
        lst.append(v)
    lst.delete_after(1)          # 3.0 을 뺀다
    assert [float(lst.val[i]) for i in lst.order()] == [1.0, 2.0, 4.0]
    k = lst.append(5.0)
    assert k == 2                # 빠진 칸이 다시 쓰인다
    assert lst.total() == 12.0
    lst.compact()
    assert np.array_equal(lst.order(), np.arange(4))
