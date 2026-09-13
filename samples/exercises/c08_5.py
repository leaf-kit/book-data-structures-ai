"""문제 8.5. 트립 삭제. 지울 노드를 회전으로 내려 잎에서 떼어 낸다."""
import numpy as np

from dsai.bst import NIL, Treap


class DeletableTreap(Treap):
    def _delete_at(self, root: int, k: int) -> int:
        if root == NIL:
            return NIL
        if k < self.key[root]:
            self.left[root] = self._delete_at(self.left[root], k)
        elif k > self.key[root]:
            self.right[root] = self._delete_at(self.right[root], k)
        else:
            l, r = self.left[root], self.right[root]
            if l == NIL:
                return r
            if r == NIL:
                return l
            if self.prio[l] > self.prio[r]:    # 큰 자식을 올리고 한 층 내려간다
                root = self._rotate_right(root)
                self.right[root] = self._delete_at(self.right[root], k)
            else:
                root = self._rotate_left(root)
                self.left[root] = self._delete_at(self.left[root], k)
        return root

    def delete(self, k: int) -> None:
        self.root = self._delete_at(self.root, k)

    def inorder(self) -> list[int]:
        out, stack, cur = [], [], self.root
        while stack or cur != NIL:
            while cur != NIL:
                stack.append(cur)
                cur = self.left[cur]
            cur = stack.pop()
            out.append(int(self.key[cur]))
            cur = self.right[cur]
        return out

    def heap_ok(self) -> bool:
        stack = [self.root] if self.root != NIL else []
        while stack:
            i = stack.pop()
            for c in (self.left[i], self.right[i]):
                if c != NIL:
                    if self.prio[c] > self.prio[i]:
                        return False
                    stack.append(c)
        return True


def test_delete_keeps_both_invariants():
    rng = np.random.default_rng(0)
    keys = rng.permutation(2000)
    t = DeletableTreap(2000, seed=1)
    for k in keys:
        t.insert(int(k))
    gone = set(rng.choice(2000, 700, replace=False).tolist())
    for k in gone:
        t.delete(k)
    assert t.inorder() == sorted(set(range(2000)) - gone)   # 순서 불변식
    assert t.heap_ok()                                         # 힙 불변식
    assert t.height() < 4 * np.log2(2000)
