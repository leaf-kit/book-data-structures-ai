"""문제 9.7. 참조 수. 요청이 쓰는 동안의 경로는 떼지 않는다."""
from dsai.radix import RadixCache, RadixNode


class RefCountedCache(RadixCache):
    def acquire(self, tokens: list[int]) -> tuple[int, list[RadixNode]]:
        self.insert(tokens)
        matched, path = self.match_prefix(tokens)
        for n in path:
            n.refs += 1
        return matched, path

    def release(self, path: list[RadixNode]) -> None:
        for n in path:
            n.refs -= 1

    def _evict_one(self) -> None:
        best, best_parent = None, None
        stack = [(self.root, None)]
        while stack:
            node, parent = stack.pop()
            if not node.children and node is not self.root and node.refs == 0:
                if best is None or node.last_used < best.last_used:
                    best, best_parent = node, parent
            stack.extend((c, node) for c in node.children.values())
        if best is None:
            return
        del best_parent.children[best.label[0]]
        self.size -= len(best.label)


def test_refs_protect_active_request():
    plain = RadixCache(capacity_tokens=12)
    plain.insert([1, 2, 3, 4, 5, 6])
    matched, _ = plain.match_prefix([1, 2, 3, 4, 5, 6])
    plain.insert([7, 8, 9, 10, 11, 12, 13])           # 첫 요청이 쓰는 동안 밀려 나간다
    assert plain.match_prefix([1, 2, 3, 4, 5, 6])[0] < 6

    safe = RefCountedCache(capacity_tokens=12)
    matched, path = safe.acquire([1, 2, 3, 4, 5, 6])
    safe.insert([7, 8, 9, 10, 11, 12, 13])
    assert safe.match_prefix([1, 2, 3, 4, 5, 6])[0] == 6   # 참조 중이라 남는다
    safe.release(path)
    safe.insert([20, 21, 22, 23, 24, 25, 26])
    assert safe.match_prefix([1, 2, 3, 4, 5, 6])[0] < 6     # 풀고 나면 나간다
