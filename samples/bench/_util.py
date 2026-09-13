"""측정 스크립트가 함께 쓰는 것. 5.4.2절의 규칙을 여기서 강제한다.

- 난수 시드 고정
- perf_counter 로 다섯 번 이상 반복해 중앙값
- 출력 파일 첫 줄에 스크립트 이름, 입력 크기, 반복 횟수, 잰 날짜
"""
from __future__ import annotations

import datetime as _dt
import statistics
import time
from pathlib import Path
from typing import Callable

OUT = Path(__file__).resolve().parent / "out"
SEED = 20260909


def median_ms(fn: Callable[[], object], repeat: int = 7, warmup: int = 1) -> float:
    """fn 을 warmup 번 버리고 repeat 번 재서 중앙값 밀리초를 돌려준다."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def header(script: str, cond: str, repeat: int) -> str:
    today = _dt.date.today().isoformat()
    return f"# {script} | {cond} | 반복 {repeat}회 중앙값 | {today}"


def write_tsv(name: str, columns: list[str], rows: list[list],
              script: str, cond: str, repeat: int) -> Path:
    """tsv 로 쓴다. 헤더에 밑줄을 쓰지 않는다. 조판이 깨진다.

    첫 줄은 측정 조건 주석이 아니라 헤더다. pgfplots 가 바로 읽기 때문이다.
    조건은 같은 이름의 .txt 에 따로 적는다.
    """
    OUT.mkdir(exist_ok=True)
    for c in columns:
        assert "_" not in c, f"헤더에 밑줄: {c}"
    path = OUT / f"{name}.tsv"
    lines = ["\t".join(columns)]
    for r in rows:
        lines.append("\t".join(_fmt(v) for v in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cond_path = OUT / f"{name}.cond.txt"
    cond_path.write_text(header(script, cond, repeat) + "\n", encoding="utf-8")
    return path


def write_txt(name: str, body: str, script: str, cond: str, repeat: int = 1) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.txt"
    text = header(script, cond, repeat) + "\n" + body.rstrip() + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def write_tex(name: str, macros: dict[str, object]) -> Path:
    """루프라인의 피크와 대역폭처럼 스칼라 하나를 본문 매크로로 넘길 때 쓴다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.tex"
    lines = [f"\\def\\{k}{{{_fmt(v)}}}" for k, v in macros.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 0.1 or v == 0:
            return f"{v:.2f}"
        if abs(v) >= 1e-3:
            return f"{v:.4f}"
        return f"{v:.1e}"
    return str(v)
