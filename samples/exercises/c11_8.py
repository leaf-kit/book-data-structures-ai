"""문제 11.8. 축소 연산을 사슬의 끝에 허용하는 융합 규칙."""
import numpy as np


def fusable_chains_with_reduce(op, indeg, outdeg, elementwise, reduce_ops):
    chains, cur = [], []

    def close():
        nonlocal cur
        if len(cur) > 1:
            chains.append(cur)
        cur = []
    for i in range(len(op)):
        if op[i] in reduce_ops and cur and outdeg[cur[-1]] == 1:
            cur.append(i)                 # 축소는 사슬의 끝에만 붙는다
            close()
        elif (op[i] in elementwise and indeg[i] <= 1
              and (not cur or outdeg[cur[-1]] == 1)):
            cur.append(i)
        else:
            close()
            if op[i] in elementwise:
                cur = [i]
    close()
    return chains


def test_reduce_ends_chain():
    op = np.array([0, 3, 3, 9, 3, 3, 9, 3])       # 3 원소별, 9 축소
    indeg = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    outdeg = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    chains = fusable_chains_with_reduce(op, indeg, outdeg, {3}, {9})
    assert chains == [[1, 2, 3], [4, 5, 6]]   # 축소 뒤의 7 은 혼자라 사슬이 아니다
