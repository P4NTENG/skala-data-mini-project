# 데이터 품질 및 구조 검증 보고서

> 양성 클래스 = hERG blocker (`label = 0`), 음성 클래스 = non-blocker (`label = 1`)

이 문서는 `scripts/inspect_data.py`가 실제로 실행한 결과만 담는다.

## 1. 파일과 컬럼

- **rows**: 8,349
- **columns**: 3,078
- **SMILES column**: `SMILES`
- **label column**: `label`
- **fingerprint blocks**: `ecfp` 1,024 bits, `fcfp` 1,024 bits, `ptfp` 1,024 bits
- **shipped descriptors**: `MolWt`, `clogp`, `sa_score`, `qed`

## 2. 라벨 분포

- **label = 0 (blocker)**: 3,807
- **label = 1 (non-blocker)**: 4,542
- **blocker fraction**: 0.4560

라벨 결측치는 0개(전체 셀 기준)이며, 중복 SMILES는 0건, 고유 SMILES는 8,349건이다.

## 3. 구조 파싱

- **RDKit 파싱 성공**: 8,349 / 8,349
- **파싱 실패 예시**: 없음
- **고유 InChIKey**: 8,322
- **고유 connectivity key**: 8,025
- **고유 Bemis-Murcko scaffold**: 4,160
- **가장 큰 scaffold 그룹**: 92 개 화합물
- **염(다중 fragment) 화합물**: 262
- **입체화학 표기 화합물**: 3,951
- **tautomer 정규화 시 구조 변경**: not computed (tautomer enumeration is a separate sensitivity run)

## 4. 기존 fingerprint 재현성

기존 컬럼을 그대로 사용하되, SMILES로부터 동일한 값을 재생성할 수 있는지 확인했다. 재생성 결과가 일치하지 않는 블록은 재생성본으로 교체하지 않고 그대로 기록한다.

| block | generator | params | rows_checked | bit_agreement | row_exact_rate | all_rows_exact |
|---|---|---|---|---|---|---|
| ecfp | morgan | {"radius": 2, "fpSize": 1024} | 8,349 | 0.999988 | 0.999401 | no |
| fcfp | morgan_features | {"radius": 2, "fpSize": 1024} | 8,349 | 1.000000 | 1.000000 | yes |
| ptfp | pattern | {"fpSize": 1024} | 8,349 | 1.000000 | 1.000000 | yes |

shipped descriptor(`MolWt`, `clogp`, `qed`, `sa_score`)를 RDKit `Descriptors.MolWt`, `Crippen.MolLogP`, `QED.qed`, contrib `sascorer`로 재계산해 비교했다(수치는 §9). `MolWt`·`clogp`의 잔차는 컬럼이 float32로 저장된 데서 오는 반올림 수준이고, `qed` 1행과 `sa_score` 10행은 라이브러리 버전 차이로 보이는 불일치가 있다. 원본 컬럼은 재계산본으로 교체하지 않고 그대로 사용한다.

## 5. Fingerprint 블록 통계

| block | n_bits | constant_bits | on_fraction | mean_bits_per_molecule | duplicate_bit_columns |
|---|---|---|---|---|---|
| ecfp | 1,024 | 0 | 0.0548 | 56.1558 | 0 |
| fcfp | 1,024 | 0 | 0.0438 | 44.9020 | 0 |
| ptfp | 1,024 | 49 | 0.4235 | 433.7085 | 106 |

## 6. 중복 구조 / 라벨 충돌

| 기준 | 중복 그룹 수 | 라벨이 충돌하는 그룹 | 해당 멤버 행 수 |
|---|---|---|---|
| canonical_smiles | 1 | 0 | 2 |
| connectivity_key | 269 | 35 | 593 |
| normalized_key | 23 | 1 | 46 |
| scaffold | 1,264 | 456 | 5,453 |

중복 구조는 **제거하지 않는다**. 라벨이 충돌하는 그룹은 목록으로 남기고(§ 불일치 목록 파일), 학습 경로에서는 원본 행을 그대로 유지한다. scaffold 그룹은 누수 방지를 위해 분할 단위로만 사용한다.

## 7. 데이터 분할과 누수 검사

| split | n_rows | fraction | n_blocker | n_non_blocker | blocker_fraction | n_unique_scaffolds | strategy |
|---|---|---|---|---|---|---|---|
| train | 6,679 | 0.8000 | 3,080 | 3,599 | 0.4611 | 2,490 | scaffold |
| val | 835 | 0.1000 | 346 | 489 | 0.4144 | 835 | scaffold |
| test | 835 | 0.1000 | 381 | 454 | 0.4563 | 835 | scaffold |
| train | 6,679 | 0.8000 | 3,046 | 3,633 | 0.4561 | 3,539 | random |
| val | 835 | 0.1000 | 381 | 454 | 0.4563 | 685 | random |
| test | 835 | 0.1000 | 380 | 455 | 0.4551 | 686 | random |

- **scaffold 중복 (train∩val)**: 0
- **scaffold 중복 (train∩test)**: 0
- **scaffold 중복 (val∩test)**: 0
- **누수 여부**: 없음

## 8. 표준화 정책

염 제거 · 중성화 · tautomer 정규화는 **기본 학습 경로에서 적용하지 않는다**. 구조와 라벨의 의미가 달라질 수 있기 때문이며, 위 표의 중복 통계(normalized 기준)로만 민감도 정보를 제공한다.

## 9. 추가 메모

- `MolWt` 재계산 최대 절대오차 5.078e-05, 오차 >1e-3인 행 0개 (n=8349)
- `clogp` 재계산 최대 절대오차 4.706e-07, 오차 >1e-3인 행 0개 (n=8349)
- `sa_score` 재계산 최대 절대오차 2.438e-01, 오차 >1e-3인 행 10개 (n=8349)
- `qed` 재계산 최대 절대오차 2.777e-02, 오차 >1e-3인 행 1개 (n=8349)
- random split의 scaffold 중복: {"train__val": 363, "train__test": 373, "val__test": 156} -- 참고용 실험이 과대평가되는 이유
- acyclic 화합물은 generic Bemis-Murcko scaffold를 사용해 분할 그룹이 무의미하게 합쳐지거나(빈 scaffold) 개별 그룹으로 흩어지는 것을 막았다.
- `ecfp` 블록은 8,344/8,349 행에서 완전히 재현된다. 불일치 5행 중 5행이 중수소(`[2H]`) 표기를 포함한다. RDKit 2026.09.1의 Morgan generator로는 동위원소/H 제거 방식을 바꿔도 bit 단위로 재현되지 않아, 해당 행은 원본 컬럼을 그대로 사용하고 목록만 남긴다.
- validation split의 blocker 비율 0.4144, 전체 0.4560 -- scaffold 단위 분할이라 완전한 층화는 불가능하다.

## 10. 실행 환경

- **python**: 3.12.14
- **platform**: Windows-11-10.0.26200-SP0
- **rdkit**: 2026.9.1
- **scikit-learn**: 1.9.1
