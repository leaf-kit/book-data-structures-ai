"""문제 5.5. 자식이 여럿인 트리에서 재귀와 같은 방문 순서를 내는 명시적 스택."""
from dsai.stack import depth_iterative


def preorder_recursive(children: list[list[int]], node: int = 0) -> list[int]:
    out = [node]
    for c in children[node]:
        out.extend(preorder_recursive(children, c))
    return out


def preorder_iterative(children: list[list[int]]) -> list[int]:
    """자식을 거꾸로 넣어야 첫 자식이 먼저 나온다."""
    out, stack = [], [0]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(reversed(children[v]))
    return out


def test_orders_agree():
    children = [[1, 2, 3], [4, 5], [], [6], [], [], []]
    assert preorder_iterative(children) == preorder_recursive(children)
    assert depth_iterative(children) == 3
