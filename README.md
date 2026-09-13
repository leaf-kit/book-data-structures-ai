# AI 시대에 다시 쓰는 자료 구조

<a href="https://leaf-kit.github.io/book-data-structures-ai/"><img src="images/cover.png" alt="AI 시대에 다시 쓰는 자료 구조" width="320"></a>

전통 자료 구조를 디딤돌 삼아 읽는 현대 AI 시스템의 자료 구조

리프 메타 (leaf meta) 지음. 무료로 배포한다. 출처를 밝히면 복사, 재배포, 인쇄, 수업 자료로 쓸 수 있다.

## 이 책

전통 자료 구조 교과서는 네 가지를 말없이 전제한다. 메모리 접근 비용이 균일하다, 데이터의 단위가 스칼라다,
답은 정확해야 한다, 구조는 사람이 설계한다. 대규모 언어 모델의 추론 서버와 벡터 검색 시스템에서는 넷 다 흔들린다.
이 책은 절마다 전통 구조를 정의와 불변식으로 다시 세우고, 어느 가정이 깨지는지를 측정으로 보이고, 그 자리를 메운 현대 구조를 같은 밀도로 세운다.

| | |
|---|---|
| 쪽수 | 507 |
| 장 | 0장에서 17장 |
| 절 | 67 |
| 도식 | 112 |
| 실험 | 73 |
| 참고문헌 | 67편, 전부 확인 날짜가 있다 |

## 읽는 곳

브라우저에서 바로 읽는다. 설치도 내려받기도 필요 없다.

### https://leaf-kit.github.io/book-data-structures-ai/

내려받아 읽으려면 [릴리즈](https://github.com/leaf-kit/book-data-structures-ai/releases/latest)에 붙인 PDF 를 받는다.
본문은 저장소에 두지 않는다. 여기에는 장별 요약과 표본 코드와 측정 출력이 있다.

## 장별 요약

[chapters/README.md](chapters/README.md) 에 열여덟 장의 요약이 있다. 각 요약은 절 목록, 대표 실험 하나, 연습문제, 더 읽을 것이다.
정의와 증명과 분석은 책에 있다.

## 표본 코드

`samples/dsai/` 가 이 책의 자료 구조 구현이고, `samples/bench/` 가 실험, `samples/tests/` 가 시험, `samples/exercises/` 가 응용 문제의 답이다.
파이썬 3.10 과 NumPy 만 있으면 된다.

```
cd samples
python -m pytest tests exercises
python bench/c15_attention.py
```

`samples/bench/out/` 이 책의 모든 표가 나온 측정 출력이다. 기계가 바뀌면 절대값은 달라지고 비율은 대체로 유지된다.
잰 기계는 책의 부록 「측정 환경」에 있다.

## 배포 조건

책은 출처 표기 조건으로 자유롭게 배포한다. 표본 코드는 MIT 라이선스로 더 느슨하게 푼다. `LICENSE` 를 본다.
