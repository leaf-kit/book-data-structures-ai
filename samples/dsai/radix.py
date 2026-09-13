"""9장. 라디스 트리와 접두사 공유 캐시. 토큰 열의 공통 접두사가 KV 캐시의 공통 부분이다.

노드 하나가 토큰 열 한 조각 (간선 라벨) 을 갖고, 자식은 첫 토큰으로 찾는다.
경로가 하나뿐인 노드는 합쳐 둔다.
요청이 오면 가장 긴 공통 접두사를 찾아 그만큼의 KV 를 다시 쓰고, 나머지만 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RadixNode:
    label: tuple[int, ...]                   # 이 노드로 들어오는 간선의 토큰 열
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    last_used: int = 0
    refs: int = 0                            # 지금 이 노드를 쓰는 요청 수


class RadixCache:
    """정의 9.9. 토큰 열의 접두사를 공유하는 압축 트라이. 노드가 KV 블록의 단위다."""

    def __init__(self, capacity_tokens: int):
        self.root = RadixNode(())
        self.capacity = capacity_tokens
        self.size = 0
        self.clock = 0

    def match_prefix(self, tokens: list[int]) -> tuple[int, list[RadixNode]]:
        """알고리즘 9.3. 가장 긴 공통 접두사의 길이와 지나온 노드들."""
        node, i, path = self.root, 0, [self.root]
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            lab = child.label
            j = 0
            while j < len(lab) and i + j < len(tokens) and lab[j] == tokens[i + j]:
                j += 1
            if j < len(lab):                  # 간선 중간에서 갈린다. 노드를 쪼갠다
                child = self._split(node, child, j)
            path.append(child)
            i += len(child.label)
            node = child
        self.clock += 1
        for p in path:
            p.last_used = self.clock
        return i, path

    def _split(self, parent: RadixNode, node: RadixNode, at: int) -> RadixNode:
        """간선 라벨을 at 에서 잘라 위 노드를 끼운다. 토큰 수는 그대로다."""
        upper = RadixNode(node.label[:at], last_used=node.last_used)
        node.label = node.label[at:]
        upper.children[node.label[0]] = node
        parent.children[upper.label[0]] = upper
        return upper

    def insert(self, tokens: list[int]) -> int:
        """접두사 뒤의 나머지를 새 노드로 붙인다. 새로 들어간 토큰 수를 돌려준다."""
        matched, path = self.match_prefix(tokens)
        rest = tuple(tokens[matched:])
        if rest:
            leaf = RadixNode(rest, last_used=self.clock)
            path[-1].children[rest[0]] = leaf
            self.size += len(rest)
            while self.size > self.capacity:
                self._evict_one()
        return len(rest)

    def _evict_one(self) -> None:
        """명제 9.11. 가장 오래 안 쓴 잎을 뗀다. 잎만 떼야 접두사가 남는다."""
        best, best_parent = None, None
        stack = [(self.root, None)]
        while stack:
            node, parent = stack.pop()
            if not node.children and node is not self.root:
                if best is None or node.last_used < best.last_used:
                    best, best_parent = node, parent
            for c in node.children.values():
                stack.append((c, node))
        if best is None:
            return
        del best_parent.children[best.label[0]]
        self.size -= len(best.label)

    def node_count(self) -> int:
        n, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children.values())
        return n
