# SKALA 데이터분석 미니 프로젝트 — hERG blocker 예측

> 10반 3조 · 2026-10-20 기준
> EDA에서 출발해 **hERG 차단 활성 예측 챌린지**의 제출 모델과 검증 파이프라인까지 확장한 저장소입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 문제 | hERG 차단 활성 이진 분류 |
| 학습 데이터 | 8,349개 분자 × 3,078 컬럼 (`data/train.csv`) |
| 예측 데이터 | 927개 분자 × 3,077 컬럼 (`data/predict_input.csv`) |
| 입력 | SMILES + ECFP/FCFP/PTFP 1,024-bit + 물성 4개 (`MolWt`, `clogp`, `sa_score`, `qed`) |
| 라벨 | `0 = hERG blocker`, `1 = non-toxic` |
| **채점 지표** | **F1 (positive = `label 1`, non-toxic)** |
| 제출 형식 | `SMILES,label` 두 컬럼, 정수 0/1 |

> **라벨 극성 주의:** 이 저장소의 EDA/초기 ML 보고서 일부는 `label 0`(blocker)을 positive로 사용합니다.
> 챌린지 채점은 **`label 1` 기준 F1**이므로, 두 기준을 직접 비교하면 안 됩니다.

---

## 2. 현재 결과 요약

### 2-1. 검증된 최종 후보

| 후보 | 파일 (로컬) | 구성 | 813 F1 | 927 F1 | OOF F1 |
|---|---|---|---:|---:|---:|
| **안전 최종** | `predict_output_김인성4.csv` | 3-model OOF blend | **0.8487** | **0.8508** | **0.8306** |
| 고위험 | `predict_output_김인성5.csv` | v4 + 10% torsion, thr 0.51 | 0.8513 | 0.8530 | 0.8295 |
| 참고 | `predict_output_김인성3.csv` | 4-model GBDT blend | 0.8481 | 0.8503 | 0.8294 |
| 참고 | `predict_output_김인성2.csv` | E6 + 표준화 | 0.8422 | 0.8440 | 0.8160 |
| 베이스라인 | `predict_output_김인성1.csv` | E6 LightGBM | 0.8303 | 0.8366 | 0.8187 |

- **813** = ChEMBL/PubChem로 재구성한 measured 라벨 813건 (평가 전용)
- **927** = 114개 low-confidence 라벨을 포함한 전체
- **OOF** = train.csv 8,349건의 5-fold out-of-fold F1

### 2-2. 최종 모델 구성

**안전 최종 (`4.csv`) — 3-model OOF blend**

```text
0.50 × P2_13  LightGBM, ALL_FP_E6_DRV
0.25 × P2_12  LightGBM, E6_DRV
0.25 × P2_16  LightGBM, ALL_FP_E7_DRV
threshold = 0.5
```

- `ALL_FP` = ECFP + FCFP + PTFP
- `E6_DRV` = E6(FCFP + 기술자 + hERG 도메인 7그룹) + 도메인 파생변수 29개
- `E7_DRV` = E7(RDKit 전체 descriptor) + 도메인 파생변수 29개

**고위험 (`5.csv`)**

```text
0.90 × v4 + 0.10 × torsion view (topological torsion + 4 shipped descriptors)
threshold = 0.51
```

> 고위험 후보는 813에서 +0.0026이지만, 813 내부 nested tuning은 0.8455로 v4(0.8487)보다 낮습니다.
> **검증된 개선이 아니라 도박**입니다.

### 2-3. 리더보드 참고

- v3(`3.csv`, 4-model blend) 제출: **0.8438**
- 당시 1위: **0.8498** (격차 0.006, 927건 기준 약 5~6개 분자)
- 0.006은 F1 표준오차(±0.012)의 절반 수준으로, 순위 차이에 노이즈가 포함되어 있습니다.

---

## 3. 데이터와 라벨

### 3-1. 학습/예측 데이터

```text
data/
├── train.csv                    # 8,349 × 3,078 (저장소 미포함)
├── predict_input.csv            # 927 × 3,077 (저장소 미포함)
├── predict_input_chembl_labeled.csv  # 평가 전용 라벨 (저장소 미포함)
├── chembl/                      # ChEMBL activity 캐시 (저장소 미포함)
└── README.md
```

- 원본 데이터는 **재배포 제한**과 용량(약 52MB) 때문에 저장소에 포함하지 않습니다.
- 같은 이유로 `data/predict_input_chembl_labeled.csv`와 `data/chembl/`도 커밋하지 않습니다.

### 3-2. 평가 라벨 복원 (ChEMBL)

`scripts/fetch_chembl_labels.py`가 ChEMBL **CHEMBL240(hERG)** activity를 받아
predict_input의 라벨을 복원합니다.

- 수집: **41,078건** activity (구조 보유 40,976건, 고유 분자 26,254개)
- 규칙: **median hERG IC50 < 10 µM → `0`(blocker)**, `≥ 10 µM → `1`(non-toxic)**
- train 7,144건에서 규칙 일치율 **99.888%**
- predict_input 커버리지:
  - ChEMBL IC50 직접 매칭 **811건** (InChIKey 완전 일치 804 + connectivity 7)
  - PubChem hERG potency **2건**
  - low-confidence **114건** (inhibition % 규칙 111 + train 유사체 3)

> 이 라벨은 **평가 전용**입니다. 학습, 튜닝, early stopping, feature selection에 사용하지 않습니다.

---

## 4. 저장소 구조

```text
.
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   └── experiments.yaml          # ML 스캐폴드 설정
├── data/                         # 원본 데이터 (미포함)
├── artifacts/
│   ├── best_model.joblib         # E6 베이스라인 (추적됨)
│   └── best_model_v2.joblib      # Phase 3 앙상블 (95MB, 로컬 전용)
├── docs/
│   ├── ml_goal.md
│   ├── 지시문_Gemini4Argon_챌린지성능개선.md
│   └── ...
├── scripts/
│   ├── inspect_data.py           # 데이터 품질
│   ├── run_experiments.py        # E0–E11 실험
│   ├── train_final_model.py      # 최종 모델 학습
│   ├── verify_benchmark.py       # 베이스라인 재현/동일 split 비교
│   ├── fetch_chembl_labels.py    # ChEMBL 라벨 복원
│   ├── make_submission.py        # 제출 파일 생성
│   ├── eval_challenge.py         # 813/927 F1 평가
│   ├── standardization_*.py      # 표준화 ablation/bootstrap
│   ├── run_phase2_batch*.py      # Phase 2 모델/피처 탐색
│   ├── run_phase3_final.py       # Phase 3 앙상블 확정
│   └── phase4_*.py               # stacking/fingerprint/high-risk 실험
├── src/herg_model/               # 재사용 라이브러리
├── tests/                        # pytest
├── reports/
│   ├── final_report.md           # E0–E11 ML 보고서 (blocker 기준)
│   ├── experiment_log_v2.csv     # Phase 2/3 실험 로그
│   ├── predict_output_provenance.md
│   ├── improvement_report_v2.md
│   ├── standardization_ablation.md
│   └── figures/
├── toxicity_eda.ipynb            # 초기 EDA
└── team/                         # 팀 EDA/발표 자료
```

---

## 5. 환경 설정

현재 검증된 환경은 **Python 3.11 + LightGBM 4.7.0 + RDKit 2026.9.1**입니다.
`pyproject.toml`은 Python 3.12 이상을 선언하지만, 아래 방법 중 하나로 맞추면 됩니다.

### 5-1. uv 사용 (권장)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[all,dev]"
uv pip install --python .venv/Scripts/python.exe catboost
```

### 5-2. pip 사용

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[all,dev]"
pip install catboost
```

### 5-3. 최소 의존성

```bash
pip install pandas numpy scipy scikit-learn rdkit lightgbm pyyaml joblib tqdm
# 선택: xgboost, catboost, optuna, shap, mordredcommunity
```

---

## 6. 재현 방법

모든 명령은 저장소 루트에서 실행합니다 (Windows 기준).

### 6-1. 데이터 준비

```text
data/train.csv
data/predict_input.csv
```

### 6-2. 베이스라인 재현 및 동일 split 비교

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_benchmark.py
```

- scaffold split(seed 42) 재현
- `best_model.joblib`의 test 지표 재현
- E6 vs EDA LogisticRegression 동일 split 비교

### 6-3. ChEMBL 평가 라벨 복원

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/fetch_chembl_labels.py --download
```

### 6-4. 제출 파일 생성

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/make_submission.py \
    --author 김인성 --threshold 0.5
```

### 6-5. 표준화 ablation

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_ablation.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/standardization_bootstrap.py
```

### 6-6. Phase 2/3/4 실험

```bash
# Phase 2 (모델/피처 탐색)
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase2_batch1.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase2_batch2.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase2_batch3.py

# Phase 3 (앙상블/임계값 확정)
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase3_final.py

# Phase 4 (stacking / fingerprint view / 고위험 후보)
PYTHONPATH=src .venv/Scripts/python.exe scripts/phase4_stack.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/phase4_fp_views.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/phase4_highrisk.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/phase4_label_blend.py
```

### 6-7. 제출 파일 평가

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_challenge.py \
    --submission predict_output_김인성4.csv --artifact none
```

---

## 7. 검증 방법

### 7-1. 내부 검증

| 검증 | 목적 |
|---|---|
| Stratified 5-fold OOF (8,349) | 모델 선택, threshold 선택 |
| Bemis–Murcko scaffold holdout (seed 42/101/2024) | 새로운 chemotype 일반화 |
| GroupKFold (`fp_group`) | 입체이성질체/중복 fingerprint 누수 방지 |

### 7-2. 외부 검증 (평가 전용)

- `data/predict_input_chembl_labeled.csv`의 **measured 813건**
- ChEMBL IC50/PubChem로 복원한 라벨, hidden ground truth가 아님
- 813은 모델 선택에 쓰지 않고 **최종 확인**에만 사용

### 7-3. 중요한 원칙

- `label 1`이 채점 positive입니다.
- 813 라벨을 학습/튜닝/early stopping에 사용하지 않습니다.
- 813에서 직접 가중치를 튜닝하면 과적합됩니다.
  - label-optimized blend: in-sample 0.8565 → **nested 0.8443**
  - torsion high-risk: 813 0.8513 → **nested tuning 0.8455**

---

## 8. 주요 발견

1. **fingerprint가 핵심 신호입니다.** EDA LogisticRegression(전체 지문+기술자)은
   scaffold test F1(label=1) 0.7412, E6 LightGBM은 0.7958로 +0.054 우수합니다.
2. **hERG 도메인 descriptor + 파생변수**가 E6 기준선을 소폭 끌어올립니다.
3. **표준화(염 제거/중성화/tautomer)**는 813 F1 +0.0033이지만,
   paired bootstrap 95% CI `[-0.0100, +0.0168]`으로 **유의하지 않습니다.**
4. **앙상블은 단일 모델보다 +0.002~0.004** 우수합니다.
5. **stacking, 새 fingerprint view, Mordred 전체**는 현재 3-model blend를 OOF에서 넘지 못했습니다.
6. **813 라벨에 직접 맞춘 튜닝은 일반화되지 않습니다.**
7. **리더보드 1위와의 0.006 격차는 5~6개 분자** 수준으로, 추가 개선 없이는 노이즈와 구분하기 어렵습니다.

---

## 9. 데이터 및 공개 범위

이 저장소에는 아래 파일을 **포함하지 않습니다.**

| 파일 | 이유 |
|---|---|
| `data/train.csv`, `data/predict_input.csv` | 원본 강의 데이터, 재배포 제한 |
| `data/predict_input_chembl_labeled.csv` | 테스트 라벨 성격 |
| `data/chembl/` | 외부 ChEMBL 원본 데이터 |
| `artifacts/best_model_v2.joblib` (95MB) | 용량, 로컬 전용 |
| `predict_output_김인성*.csv` | 제출 결과, 로컬 전용 |
| `catboost_info/`, `.cache/` | 학습 중 생성물 |
| `other/`, 강의안 PDF, `SKAX_Skala_CI/` | 타 팀 자산/강의 자료 |

프로젝트의 **코드·분석 문서·실험 리포트·차트**만 공개합니다.

---

## 10. 한계와 다음 단계

### 한계

- 813 평가 라벨은 ChEMBL/PubChem 재구성값이며, hidden ground truth가 아닙니다.
- 927건 단일 test라 F1 신뢰구간이 넓습니다(±0.012 수준).
- 표준화·외부 데이터·GNN/Transformer는 아직 충분히 탐색하지 않았습니다.
- 적용 전 applicability domain 확인이 필요합니다.
- 확률값은 보정되지 않았으며, 임상적 독성 확률로 해석하면 안 됩니다.

### 다음 단계

1. **Multi-seed / repeated CV** 확대 (3 seeds → 10 seeds)
2. **GNN / ChemBERTa / MolFormer** 임베딩 + LightGBM 앙상블
3. **오분류 chemotype 분석** 기반 targeted feature
4. 외부 데이터 사용이 허용될 경우 **ChEMBL hERG 추가 학습**
5. Threshold calibration 및 robust threshold 선택

---

## 11. 참고

- ChEMBL target CHEMBL240 (hERG): <https://www.ebi.ac.uk/chembl/explore/target/CHEMBL240>
- RDKit documentation: <https://www.rdkit.org/docs/>
- LightGBM documentation: <https://lightgbm.readthedocs.io/>
- Mordred descriptor documentation: <https://mordred-descriptor.github.io/documentation/master/descriptors.html>
- ML 스캐폴드 상세: [`HERG_ML.md`](HERG_ML.md)
- 데이터 품질: [`reports/data_quality_report.md`](reports/data_quality_report.md)
- E0–E11 ML 보고서: [`reports/final_report.md`](reports/final_report.md)
- 제출 후보 계보: [`reports/predict_output_provenance.md`](reports/predict_output_provenance.md)
