"""9장 실험이 쓰는 말뭉치. 파이썬 표준 라이브러리의 소스 파일을 이름순으로 이어 붙인다.

같은 파이썬 버전이면 같은 바이트 열이 나온다. 조건 줄에 버전을 적는다.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

LIMIT = 1_000_000


def stdlib_text(limit: int = LIMIT) -> bytes:
    root = Path(os.__file__).parent
    out = bytearray()
    for p in sorted(root.glob("*.py")):
        out += p.read_bytes()
        if len(out) >= limit:
            break
    return bytes(out[:limit])


def word_counts(text: bytes, top: int = 8000) -> Counter:
    """공백으로 나눈 단어. 앞 공백을 단어에 붙여 둔다 (GPT 계열 토크나이저의 관례)."""
    words = Counter()
    for w in text.split(b" "):
        if w:
            words[b" " + w] += 1
    return Counter(dict(words.most_common(top)))


def version() -> str:
    return f"Python {sys.version_info.major}.{sys.version_info.minor}"
