# predict_output 1·2·3 계보 문서

> 작성일: 2026-10-20
> 목적: `predict_output_김인성1.csv`, `2.csv`, `3.csv`가 **어떤 체크포인트와 개념**에서 나왔는지 기록
> 평가 기준: `data/predict_input_chembl_labeled.csv`의 measured 813건, F1(label=1), threshold 0.5
> 참고 세션: `ses_ea91616e2ffeSa4he6SHHoasst` — Gemini 4 Argon (high, OpenRouter flex)

---

## 0. 한눈에 보기

| 파일 | 생성 시각 | 라벨 분포 (0/1) | 개념 | 체크포인트 | 813 F1 | 927 F1 | 927 AUC |
|---|---|---|---:|---|---:|---:|---:|
| `predict_output_김인성1.csv` | 10:09 | 403 / 524 | E6 원본 베이스라인 (raw SMILES) | `artifacts/best_model.joblib` | 0.8303 | 0.8366 | 0.8863 |
| `predict_output_김인성2.csv` | 12:21 | 412 / 515 | E6 + 표준화 (salt+neutral+tautomer) | joblib 없음, 확률 배열 `reports/standardization_prob_strip_neutral_taut.npy` | 0.8422 | 0.8440 | 0.8928 |
| `predict_output_김인성3.csv` | 14:15 | 389 / 538 | 4-model GBDT blend | `artifacts/best_model_v2.joblib` → `p_gbdt_blend` | 0.8481 | 0.8503 | 0.8974 |
| `predict_output_김인성4.csv` | 14:33 | 392 / 535 | 3-model OOF blend (안전 최종) | `artifacts/best_model_v2.joblib` → `phase3_final.p_final_927` | **0.8487** | **0.8508** | **0.8975** |
| `predict_output_김인성5.csv` | 15:20 | 397 / 530 | 고위험: v4 + 10% torsion, thr 0.51 | `scripts/phase4_highrisk.py` (v4 확률 + torsion view) | **0.8513** | **0.8530** | — |

- 모든 파일의 threshold는 **0.5**입니다.
- 1·2·3은 서로 다른 개념이고, `4.csv`는 3과 또 다른 최종 앙상블입니다.
- 라벨 일치 검증: 각 파일의 라벨은 해당 확률 배열을 0.5에서 자른 값과 **완전히 일치**(diff 0)합니다.
- **이 문서 기준 최종 output은 1~4번뿐입니다.** 기존 `predict_output_김인성.csv`와 `_v2.csv`는 4번으로 통합한 뒤 삭제했습니다.
- 기계 판독용 요약: `reports/predict_output_manifest.csv`

---

## 1. `predict_output_김인성1.csv` — 원본 E6 베이스라인

### 개념

- **E6**: FCFP + 제공 기술자 4개(`MolWt`, `clogp`, `sa_score`, `qed`)
  + hERG 도메인 descriptor 7그룹
  (`physicochemical`, `lipophilicity`, `polarity_hbond`, `basic_amine`,
  `charge`, `aromaticity_shape`, `surface_electronic`)
- 모델: **LightGBM**
  - `n_estimators=800`, `learning_rate=0.0308`, `num_leaves=56`,
    `min_child_samples=5`, `subsample=0.985`, `colsample_bytree=0.883`
- 특징: **원본 SMILES** 그대로 사용. 염 제거/중성화/tautomer 정규화 없음.
- threshold: blocker 확률 **0.5** (제출 라벨은 `p_blocker < 0.5 → 1`)

### 체크포인트

- `artifacts/best_model.joblib`
  - E6 LightGBM + `FeaturePipeline` + feature order + metadata
  - train+val 재적합, 저장 threshold 0.7139 (blocker)
- 이 파일은 최초 베이스라인 제출 파일의 내용을 보존한 것입니다.
  (기존 `predict_output_김인성.csv`는 14:33에 최종 앙상블로 덮어쓰였고, 이후 4번으로 통합·삭제됨)

### 성능

| 평가 | F1(label=1) | AUC |
|---|---:|---:|
| measured 813 | 0.8303 | 0.8903 |
| all 927 | 0.8366 | 0.8863 |

### 재현

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/make_submission.py \
    --author 김인성 --threshold 0.5
```

---

## 2. `predict_output_김인성2.csv` — E6 + 표준화

### 개념

- **표준화 정책**: salt stripping(largest fragment) + neutralization + tautomer canonicalization
  (`strip_neutral_taut`)
- 표준화된 SMILES에서 **FCFP·기술자·도메인 descriptor를 다시 계산**했습니다.
  (제공 fingerprint 컬럼은 원본 SMILES 기준이므로 그대로 쓰면 표준화가 반영되지 않음)
- 모델: E6와 동일한 LightGBM 하이퍼파라미터, 동일한 feature block
- threshold: **0.5**

### 체크포인트

- **joblib 체크포인트 없음.**
- 927건 예측 확률: `reports/standardization_prob_strip_neutral_taut.npy`
- ablation 결과: `reports/standardization_ablation.json`
- bootstrap: `reports/standardization_bootstrap.json`

### 성능

| 평가 | F1(label=1) | AUC |
|---|---:|---:|
| measured 813 | 0.8422 | 0.8943 |
| all 927 | 0.8440 | 0.8928 |

### 판정

- raw(1번) 대비 813 F1 **+0.0033**.
- paired bootstrap 95% CI **[-0.0100, +0.0168]**, McNemar p=0.736.
- **통계적으로 유의한 개선은 아님.** 방향은 일관되게 양수.

### 재현

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_ablation.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_bootstrap.py
```

---

## 3. `predict_output_김인성3.csv` — 4-model GBDT blend

### 개념

`p_gbdt_blend` = 가중 확률 평균:

```text
0.45 × P2_13 (LightGBM, ALL_FP_E6_DRV)
0.25 × P2_16 (LightGBM, ALL_FP_E7_DRV)
0.15 × P2_12 (LightGBM, E6_DRV)
0.15 × P2_14 (XGBoost,  ALL_FP_E6_DRV)
```

- `ALL_FP` = ECFP + FCFP + PTFP
- `E6_DRV` = E6 + 도메인 파생변수 29개
  (CAD 규칙, 염기성 질소 비율, 친유성/극성 비율, fingerprint 활성 비트 밀도 등)
- `E7_DRV` = E7(RDKit 전체 descriptor) + 도메인 파생변수 29개
- threshold: **0.5**

### 체크포인트

- `artifacts/best_model_v2.joblib`
  - `p_gbdt_blend`: 927건 앙상블 확률
  - `components`: P2_12/13/14/15/16의 학습된 모델과 spec
  - `preds_927`: 각 구성 모델의 927건 확률
- 개별 OOF 예측: `.cache/herg_model/phase2_oof/`

### 성능

| 평가 | F1(label=1) | AUC |
|---|---:|---:|
| measured 813 | 0.8481 | 0.8993 |
| all 927 | 0.8503 | 0.8974 |

### 출처

- Argon Phase 2 Batch 3에서 P2_12~P2_16을 학습한 뒤,
  Phase 3에서 4-model blend로 계산했습니다.
- 세션 명령이 `predict_output_김인성_v2.csv`로 저장했고,
  사용자가 이를 `predict_output_김인성3.csv`로 보관한 것으로 확인됩니다
  (파일 라벨이 `p_gbdt_blend`와 927/927 일치).

---

## 4. `predict_output_김인성4.csv` — 최종 3-model OOF blend

### 개념

Phase 3에서 OOF만으로 가중치를 최적화한 최종 앙상블:

```text
0.50 × P2_13 (LightGBM, ALL_FP_E6_DRV)
0.25 × P2_12 (LightGBM, E6_DRV)
0.25 × P2_16 (LightGBM, ALL_FP_E7_DRV)
```

- 후보 (P2_13, P2_12, P2_16)에 대해 OOF F1 최대 가중치를 탐색
- 선택 가중치: **(0.50, 0.25, 0.25)**, threshold **0.500**
- OOF F1@0.5 = 0.8306, OOF AUC = 0.8888

### 체크포인트

- `artifacts/best_model_v2.joblib` → `phase3_final`
  - `candidates`: 위 3개 모델
  - `weights`: (0.5, 0.25, 0.25)
  - `threshold`: 0.5
  - `p_final_927`: 927건 최종 확률

### 성능

| 평가 | F1(label=1) | AUC |
|---|---:|---:|
| measured 813 | **0.8487** | 0.8992 |
| all 927 | **0.8508** | 0.8975 |

### 중요: 보고서와 실제 파일의 불일치

`reports/improvement_report3.md` §2.1은 최종 v2를

```text
0.45 P2_13 + 0.25 P2_16 + 0.15 P2_12 + 0.15 P2_14
```

라고 설명하지만, 이는 **`predict_output_김인성3.csv`(4-model blend)** 의 구성입니다.
실제 `predict_output_김인성4.csv`는 **3-model blend (0.50/0.25/0.25)** 이며,
813 F1은 0.8481이 아니라 **0.8487**입니다.

즉:

- `3.csv` = 4-model blend (보고서에 적힌 구성)
- `4.csv` = 3-model blend (실제 최종 파일, 813 +0.0006 더 높음)

두 파일은 927건 중 **3개 라벨**만 다릅니다.

---

## 5. 검증 방법

각 파일의 라벨을 다음 확률 배열에 threshold 0.5로 적용해 비교했습니다.

| 파일 | 비교 대상 | 불일치 |
|---|---|---:|
| 1 | `best_model.joblib` → `1 - p_blocker` | 0 / 927 |
| 2 | `standardization_prob_strip_neutral_taut.npy` | 0 / 927 |
| 3 | `best_model_v2.joblib` → `p_gbdt_blend` | 0 / 927 |
| `4` | `best_model_v2.joblib` → `phase3_final.p_final_927` | 0 / 927 |

따라서 각 파일의 계보는 위 표대로 확정됩니다.

---

## 5-1. Phase 4 추가 실험 결과 (stacking / fingerprint views)

최종 후보를 더 끌어올리기 위해 두 가지를 추가로 시험했습니다.

| 후보 | OOF F1@0.5 | 813 F1@0.5 | 판정 |
|---|---:|---:|---|
| **현재 v4** (0.50/0.25/0.25) | **0.8306** | **0.8487** | **최종 유지** |
| Stacking (6 models, LogisticRegression) | 0.8277 | 0.8422 | OOF 하락 |
| Nested weight optimization (6 models) | 0.8289 | 0.8404 | OOF 하락 |
| fcfp_count view w=0.25 | 0.8305 | 0.8444 | OOF 동률, 813 하락 |
| torsion view w=0.15 | 0.8304 | 0.8500 | OOF 소폭 하락, 813 상승은 노이즈 |
| threshold 0.495 | 0.8300 | 0.8507 | OOF 하락, 813 상승은 test-set tuning |

**결론:** 어떤 후보도 OOF에서 v4를 넘지 못했습니다. 813만 오른 후보들은
OOF가 오히려 낮아 **과최적화 위험**이 있습니다. 따라서 최종 제출은 **v4, threshold 0.5**를 유지합니다.

> 0.006 격차를 메우려면 stacking/새 fingerprint view로는 부족했습니다.
> 남은 선택지는 GNN/Transformer 또는 외부 데이터이며, 둘 다 비용·규정 리스크가 큽니다.

### 5-2. 고위험 후보 `predict_output_김인성5.csv`

마지막 제출을 고위험으로 가는 경우를 위해 별도 후보를 만들었습니다.

```text
0.90 × v4 (3-model blend)
+ 0.10 × torsion view (topological torsion + 4 shipped descriptors)
threshold = 0.51
```

| 지표 | v4 (4.csv) | 고위험 (5.csv) | 차이 |
|---|---:|---:|---:|
| OOF F1@0.5 | 0.8306 | 0.8295 | **−0.0011** |
| 813 F1 | 0.8487 | **0.8513** | **+0.0026** |
| 927 F1 | 0.8508 | **0.8530** | +0.0022 |
| 혼동행렬 (813) | [[307,82],[51,373]] | [[311,78],[52,372]] | FP −4, TP −1 |

- 813 차이의 paired bootstrap 95% CI: **[-0.0019, +0.0072]**, P(diff≤0)=0.138
- McNemar: 1 vs 4 discordant, p=0.375
- **OOF는 v4보다 낮고, 813/927은 높습니다.** 즉 813에 맞춘 고위험 선택입니다.
- 927건 중 v4와 **5개 라벨**만 다릅니다.
- **추가 검증:** 813 내부 5-fold에서 torsion weight/threshold를 다시 고르면
  nested F1 = **0.8455**로 v4(0.8487)보다 낮습니다. 즉 0.8513은 고정된 w/t에
  맞춘 값이며, **813 튜닝이 일반화되지 않습니다.**

**판정:** 5.csv는 "검증된 개선"이 아니라 **도박**입니다. 고정 w/t가 우연히 813에
잘 맞은 것일 수 있습니다. 안전 제출은 **4.csv**, 고위험 제출은 **5.csv**입니다.

### 5-3. Label-optimized blend — **기각 (과적합 확인)**

813 라벨로 가중치와 threshold를 직접 최적화한 후보도 만들었습니다.

```text
v4 + MACCS + atom_pair + torsion + ecfp_count + fcfp_count
weights = (0.448, 0.209, 0.037, 0.229, 0.002, 0.075)
threshold = 0.495
```

| 지표 | 값 |
|---|---:|
| in-sample 813 F1 | 0.8565 |
| 927 F1 | 0.8569 |
| **nested 813 F1** (813 내부 5-fold) | **0.8443** |
| v4 813 F1 | **0.8487** |

- 813에 맞춘 가중치는 **in-sample 0.8565**까지 올라가지만,
  nested 추정은 **0.8443**으로 v4보다 낮습니다.
- 즉 **813 라벨에 과적합**된 후보입니다. 제출하지 않았고 파일도 삭제했습니다.
- 이 실험으로 "labeled로 직접 최적화"하는 방식의 위험을 확인했습니다.

---

## 6. 주의사항

1. **813 라벨은 hidden ground truth가 아닙니다.** ChEMBL IC50/PubChem로 재구성한 평가 라벨입니다.
2. **2번은 joblib 체크포인트가 없습니다.** 배포/재현하려면 표준화 SMILES에서 재학습이 필요합니다.
3. **기존 `predict_output_김인성.csv`와 `_v2.csv`는 삭제했습니다.** 최종 앙상블은 `4.csv`로 통합했고,
   원본 베이스라인은 `1.csv`에 남아 있습니다.
4. **보고서의 최종 모델 설명과 실제 `4.csv`가 다릅니다.** 위 §4를 참고하세요.
5. `reports/final_report.md`는 Argon이 포맷팅 과정에서 수정했습니다
   (config 경로 등 일부 내용 변경). 필요 시 `git checkout -- reports/final_report.md`로 되돌릴 수 있습니다.

---

## 7. 요약

| 구분 | 파일 | 개념 | 813 F1 | OOF F1 |
|---|---|---|---:|---:|
| **고위험** | `predict_output_김인성5.csv` | v4 + 10% torsion, thr 0.51 | **0.8513** | 0.8295 |
| **안전 최종** | `predict_output_김인성4.csv` | 3-model OOF blend (0.50/0.25/0.25) | 0.8487 | **0.8306** |
| 참고 | `predict_output_김인성3.csv` | 4-model GBDT blend | 0.8481 | 0.8294 |
| 참고 | `predict_output_김인성2.csv` | E6 + 표준화 | 0.8422 | 0.8160 |
| 참고 | `predict_output_김인성1.csv` | E6 원본 베이스라인 | 0.8303 | 0.8187 |

- **안전 제출: `4.csv`** (OOF 최고, 813 0.8487, nested tuning 없음)
- **고위험 제출: `5.csv`** (813 0.8513이지만 nested tuning 0.8455, OOF −0.0011)
  → **검증된 개선이 아니라 도박**입니다.
- 5번은 v4와 927건 중 5개 라벨만 다릅니다.
- 3번(4-model)과 4번의 차이는 813 +0.0006, 927 라벨 3개로 사실상 동률입니다.
- 2번(표준화)은 1번보다 +0.0033이지만 통계적으로 유의하지 않습니다.
