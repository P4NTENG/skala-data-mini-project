# hERG 라벨 검증 (`herg_validation/`)

`data/train.csv`의 `label`(0 = 독성, 1 = 비독성)이 **hERG 칼륨 채널 차단 활성**을 의미하는지
공개 실험 데이터와 화합물별로 대조해 검증하는 재현 가능한 파이프라인입니다.

## 실행

```bash
# 1) 환경 (Python 3.12 — RDKit wheel 필요)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe pandas numpy scipy scikit-learn matplotlib seaborn rdkit requests

# 2) 전체 파이프라인 실행 (외부 데이터 다운로드 포함, 최초 1회 약 15분)
.venv/Scripts/python.exe herg_validation/validation.py --stage all

# 개별 단계 (catalog → fetch → match → stats → endpoints → alt → props → figures)
.venv/Scripts/python.exe herg_validation/validation.py --stage match
```

외부 데이터는 `herg_validation/cache/`에 캐시되며 재실행 시 재사용됩니다.

## 산출물

| 파일 | 내용 |
|---|---|
| `validation_report.md` | 방법·데이터·결과·한계·결론 |
| `DATA_SOURCES.md` | 외부 데이터 URL·조회일·라이선스·인용 |
| `output/matched_compounds.csv` | 구조 매칭 + 실험값 대조 (분자 × 레코드) |
| `output/compound_level.csv` | 화합물 단위 집계 (임계값별 · %inhibition 분리) |
| `output/compound_level_by_source.csv` | 출처별 화합물 단위 집계 |
| `output/label_conflicts.csv` | 불일치 사례 + 원인 분류 |
| `output/excluded_cases.csv` | 분석 제외 사례와 사유 |
| `output/metrics.csv` | 혼동행렬 · 민감도/특이도/정밀도/BA + 95% CI + 순열검정 |
| `output/labeling_rule_reverse.csv` | 라벨링 규칙 역추적 (집계 × 임계값) |
| `output/endpoint_scan.csv` | ChEMBL 다중 타깃 종말점 적합도 순위 |
| `output/alt_endpoint_scan.csv` | Tox21 12-endpoint · ClinTox 비교 |
| `figures/*.png` | 혼동행렬, 임계값 민감도, 출처별 일치율, potency 분포, 종말점 순위 |

## 설계 메모

- 라벨 정의(0 = 독성)는 지시문에 고정된 값이며 코드에서도 `LABEL_TOXIC = 0` 상수로만 사용합니다.
- `similarity_only`(ECFP Tanimoto ≥ 0.90) 매칭은 `matched_compounds.csv`에 남기되 **주 분석에서 제외**합니다.
- `percent inhibition` 단독 측정은 IC50로 환산하지 않고 별도 클래스(`class_pct_*`)로 보존합니다.
- 검열 측정값(`>`, `<`)은 임계값 대비 방향만 사용하고 `ambiguous_censored`로 표시합니다.
- 화합물 단위 집계에서 assay 간 결과가 상충하면 `ambiguous`로 유지하며 단일 값으로 축약하지 않습니다.
- 동일 화합물의 반복 assay는 행 단위/화합물 단위 통계를 분리해 보고합니다.

## 주의 (데이터 재배포)

`output/`·`cache/` 산출물에는 원본 데이터의 SMILES가 포함됩니다.
저장소 공개 범위 규칙(`README.md` §5)에 따라 커밋하지 않는 것을 권장합니다.
`cache/` 전체와, 원본 SMILES를 포함하는 `output/matched_compounds.csv`,
`compound_level*.csv`, `label_conflicts.csv`, `dataset_keys.csv`,
`dataset_with_properties.csv`는 `.gitignore`에 등록되어 있습니다.
(지표·종말점·규칙 역추적 CSV와 `figures/`, 보고서는 SMILES를 포함하지 않아 커밋 가능)
