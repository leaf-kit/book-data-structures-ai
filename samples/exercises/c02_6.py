"""문제 2.6. 최적 맞춤 (best fit) 연속 할당자를 만들어 첫 맞춤과 단편화를 비교한다."""
from dsai.alloc import ContiguousArena


class BestFitArena(ContiguousArena):
    def alloc(self, key: int, length: int) -> bool:
        best = None
        for i, (start, size) in enumerate(self.free):
            if size >= length and (best is None or size < self.free[best][1]):
                best = i
        if best is None:
            return False
        start, size = self.free[best]
        self.used[key] = (start, length)
        if size == length:
            self.free.pop(best)
        else:
            self.free[best] = (start + length, size - length)
        return True


def test_best_fit_picks_tightest_hole():
    a = BestFitArena(100)
    a.alloc(1, 20)
    a.alloc(2, 50)
    a.alloc(3, 30)
    a.release(1)          # 빈 구간 20
    a.release(3)          # 빈 구간 30
    assert a.alloc(4, 25)
    assert a.used[4][0] == 70   # 30 짜리 구간 (시작 70) 에 들어간다
    f = ContiguousArena(100)
    f.alloc(1, 20); f.alloc(2, 50); f.alloc(3, 30)
    f.release(1); f.release(3)
    # 첫 맞춤도 여기서는 같다. 20 짜리 구간에는 안 들어가므로
    assert f.alloc(4, 25) and f.used[4][0] == 70
