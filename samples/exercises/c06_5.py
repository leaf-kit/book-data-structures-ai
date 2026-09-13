"""문제 6.5. 선형 탐사에 삭제를 더한다. 그냥 비우면 탐색이 깨지고, 묘비로 고친다."""
from dsai.hashing import LinearProbingTable


class DeletableTable(LinearProbingTable):
    TOMB = -2

    def delete(self, key: int) -> bool:
        i = self.slot(key)
        while self.table[i] != self.EMPTY:
            if self.table[i] == key:
                self.table[i] = self.TOMB      # 묘비. 탐색은 지나가고 삽입은 쓸 수 있다
                self.n -= 1
                return True
            i = (i + 1) % self.m
        return False

    def contains(self, key: int) -> bool:
        i = self.slot(key)
        while self.table[i] != self.EMPTY:
            if self.table[i] == key:
                return True
            i = (i + 1) % self.m
        return False


def test_tombstone_keeps_probe_chain():
    t = DeletableTable(8, seed=3)
    keys = [k for k in range(100) if t.slot(k) == 0][:3]   # 같은 자리에 오는 키 셋
    for k in keys:
        t.insert(k)
    assert t.delete(keys[0])
    assert t.contains(keys[2])          # 묘비 덕에 뒤의 키를 계속 찾는다
    # 묘비 없이 비우면 keys[2] 를 못 찾는다
    t.table[t.slot(keys[0])] = t.EMPTY
    assert not t.contains(keys[2])
