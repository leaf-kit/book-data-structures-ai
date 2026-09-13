"""문제 12.6. 스킵 리스트 삭제. 층마다 왼쪽 노드를 찾아 링크를 건너뛰게 한다."""
import numpy as np

from dsai.skiplist import NIL, SkipList


class DeletableSkipList(SkipList):
    def delete(self, k: int) -> bool:
        cur = self.head
        update = [self.head] * self.nxt.shape[0]
        for l in range(self.top, -1, -1):
            while self.nxt[l, cur] != NIL and self.key[self.nxt[l, cur]] < k:
                cur = self.nxt[l, cur]
            update[l] = cur
        target = self.nxt[0, cur]
        if target == NIL or self.key[target] != k:
            return False
        for l in range(int(self.level[target])):
            if self.nxt[l, update[l]] == target:
                self.nxt[l, update[l]] = self.nxt[l, target]
        while self.top > 0 and self.nxt[self.top, self.head] == NIL:
            self.top -= 1
        return True


def test_delete_keeps_search_correct():
    sl = DeletableSkipList(3000, seed=2)
    keys = np.random.default_rng(3).permutation(3000)
    for k in keys:
        sl.insert(int(k))
    gone = set(keys[:1000].tolist())
    for k in gone:
        assert sl.delete(int(k))
    for k in keys[1000:1300]:
        assert sl.search(int(k))[0]
    for k in list(gone)[:300]:
        assert not sl.search(int(k))[0]
