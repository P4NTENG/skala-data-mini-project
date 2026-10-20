# 작업 지시문 — hERG 챌린지 성능 개선 (Gemini 4 Argon)

> 작성일: 2026-10-20
> 수신: Gemini 4 Argon
> 발신: 김인성
> 목적: 현재 **F1(label=1) 0.83**을 베이스라인으로, `predict_input.csv` 927건에 대한 **챌린지 제출 성능을 최대화**한다.

---

## 복사용 요약 프롬프트

```text
너는 Gemini 4 Argon이다. 이 저장소의 docs/지시문_Gemini4Argon_챌린지성능개선.md를 읽고
그대로 수행하라. 목표는 hERG 챌린지의 F1(label=1)을 현재 베이스라인 0.8303
(data/predict_input_chembl_labeled.csv의 measured 813건 기준)에서 최대화하는 것이다.

절대 규칙:
1. data/predict_input_chembl_labeled.csv의 라벨을 학습/튜닝/early stopping에 쓰지 마라.
   오직 최종 평가에만 사용한다.
2. 기존 파일(predict_output_김인성.csv, artifacts/best_model.joblib, reports/final_report.md)을
   덮어쓰지 마라. 새 산출물은 _v2 suffix를 붙인다.
3. 모델 선택은 train 내부 OOF(random 5-fold)와 scaffold holdout으로 하고,
   평가셋 813건은 마지막 확인에만 쓴다.
4. 최종 보고에는 813 measured F1, 95% bootstrap CI, 혼동행렬, 실패한 실험을 포함한다.
5. 라벨 극성을 뒤집지 마라. label=1이 채점 positive다.

먼저 scripts/eval_challenge.py와 random 5-fold OOF 하네스를 만들고,
베이스라인 0.8303을 재현한 뒤 **결과만 보고하고 즉시 종료하라.**
Phase 2(모델/피처 탐색)는 내가 새 메시지로 승인할 때까지 절대 시작하지 마라.
Phase 2 학습 명령을 추가로 실행하면 이번 작업은 실패로 간주한다.
```

---

## 0. 미션

hERG(human Ether-à-go-go-Related Gene) 차단 활성 이진 분류 챌린지에서
**F1-score (positive = `label 1`, 비독성)** 을 최대화한다.

- 학습 데이터: `data/train.csv` — 8,349행 × 3,078열
- 예측 대상: `data/predict_input.csv` — 927행 × 3,077열 (`SMILES` + 피처, `label` 없음)
- 제출 형식: `SMILES,label` 두 컬럼, `label`은 정수 0/1
- 산출물 규칙: 기존 `predict_output_김인성.csv`는 **보존**하고, 새 후보는 `predict_output_김인성_v2.csv`로 저장한다. 최종 파일 교체는 사용자 승인 후에만 한다.

**중요:** 채점은 F1(label=1)이다. 저장소의 기존 ML 보고서는 blocker(`label 0`)를 positive로
쓰고 있으므로, 그 보고서의 F1(0.6677 등)을 그대로 목표 지표로 착각하지 말 것.

---

## 1. 고정된 라벨 의미 (절대 뒤집지 말 것)

| 원본 `label` | 의미 | 챌린지 클래스 |
|---|---|---|
| `0` | hERG 차단 활성 있음 (blocker) | negative |
| `1` | hERG 차단 활성 낮음 (non-toxic) | **positive (채점 대상)** |

- 모델 내부에서 blocker를 positive로 학습해도 무방하지만, **최종 지표와 제출 라벨은 반드시 `label=1` 기준**으로 변환해 검증할 것.
- `y_blocker = (label == 0)` 변환 후 `p_blocker`를 예측하고, 제출은 `p_blocker < threshold → 1`로 만든다.

---

## 2. 현재 상태 (검증된 베이스라인)

### 2.1 데이터

| 항목 | 값 |
|---|---|
| train 행 | 8,349 |
| label 1 (non-toxic) | 4,542 (54.4%) |
| label 0 (blocker) | 3,807 (45.6%) |
| predict_input 행 | 927 |
| 결측 | 0 |
| 중복 SMILES | 0 |

### 2.2 현재 제출 모델

| 항목 | 값 |
|---|---|
| 아티팩트 | `artifacts/best_model.joblib` |
| 구성 | E6: FCFP + 기존 기술자 4개 + hERG 도메인 descriptor 7그룹 |
| 모델 | LightGBM (`n_estimators=800`, `learning_rate=0.0308`, `num_leaves=56`, `min_child_samples=5`) |
| 피처 수 | 1,224 |
| 임계값 | blocker 확률 0.5 (제출 기준) |
| 제출 파일 | `predict_output_김인성.csv` (927행, 0: 403 / 1: 524) |

### 2.3 베이스라인 성능

ChEMBL에서 재구성한 평가 라벨(`data/predict_input_chembl_labeled.csv`) 대비:

| 평가 대상 | Accuracy | F1(label=1) | F1(label=0) | AUC |
|---|---:|---:|---:|---:|
| 전체 927건 | 0.8188 | **0.8366** | 0.7966 | 0.8884 |
| Measured 813건 | 0.8180 | **0.8303** | 0.8037 | 0.8903 |

- **이 0.8303(measured 813)이 공식 베이스라인**이다.
- 전체 927건 수치는 low-confidence 라벨 114건이 섞여 있으므로 참고용이다.

### 2.4 학습/검증 참고 수치

| 실험 | F1(label=1) | AUC |
|---|---:|---:|
| E6 train-only, scaffold test 835행 | 0.7958 | 0.8418 |
| E6 train-only, random holdout 835행 | 0.8278 | 0.8730 |
| EDA LogisticRegression, scaffold test | 0.7412 | 0.7737 |

- scaffold split(seed 42): train 6,679 / val 835 / test 835, blocker 3,080 / 346 / 381
- random split도 같은 seed로 재현 가능

---

## 3. 평가용 라벨 (절대 학습에 쓰지 말 것)

`data/predict_input_chembl_labeled.csv`는 **평가 전용**이다.

- ChEMBL CHEMBL240 activity 41,078건을 InChIKey로 대조해 복원
- 규칙: **median hERG IC50 < 10 µM → label 0**, `>= 10 µM → label 1`
- train 7,144건에서 규칙 일치율 **99.888%**
- 커버리지:
  - ChEMBL IC50 직접 매칭 811건 (804 full InChIKey + 7 connectivity)
  - PubChem hERG potency 2건
  - low-confidence 114건 (inhibition % 규칙 111 + train 유사체 3)

### 절대 금지

1. **`predict_input_chembl_labeled.csv`의 라벨을 학습 데이터로 사용하지 말 것.**
   - feature engineering, pseudo-labeling, early stopping, threshold tuning에 사용 금지.
   - 오직 **최종 성능 보고**와 **모델 간 비교**에만 사용한다.
2. **low-confidence 114건을 ground truth로 쓰지 말 것.**
   - Primary metric은 `label_chembl.notna() | source == 'pubchem_herg_potency'`인 **813건**에서 계산한다.
3. **모델 선택을 813건 평가 점수로만 하지 말 것.** 과적합된다.
   - 모델 선택은 train 내부 OOF/CV로 하고, 813건은 마지막 확인용으로만 쓴다.
4. **기존 파일 덮어쓰기 금지.**
   - `artifacts/best_model.joblib`, `predict_output_김인성.csv`, `reports/final_report.md`는 보존.
   - 새 산출물은 `_v2`, `_v3` suffix를 붙인다.
5. **라벨 극성을 뒤집지 말 것.**
6. **단일 split 결과만으로 "개선됐다"고 보고하지 말 것.** CI 또는 반복 CV를 함께 제시한다.

> 참고: `predict_input`에는 train과 완전히 동일한 분자가 8건 있고, connectivity(InChIKey 앞
> 14자)가 같은 분자는 총 116건(그중 8건이 완전 동일)이다. 이들을 이용해 train 라벨을 복사하는
> 방식은 **제출 성능 개선으로 간주하지 않는다.** (평가 라벨 누설과 같은 문제다.)

---

## 4. 평가 프로토콜 (반드시 이 순서로)

### 4.1 Primary metric

```
F1(label=1) on 813 measured predict rows
```

- 라벨: `data/predict_input_chembl_labeled.csv`
- subset: `(label_chembl.notna()) | (source == 'pubchem_herg_potency')`
- bootstrap 1,000회로 95% CI를 함께 보고

### 4.2 Secondary metrics

| 지표 | 용도 |
|---|---|
| F1(label=1) on 927 all | 참고 (low-confidence 포함) |
| ROC-AUC / PR-AUC(label=1) | 순위 능력 비교 |
| F1(label=0) / blocker recall | 안전성 관점 |
| Accuracy | 보조 |

### 4.3 모델 선택용 내부 검증

평가셋(813)을 보기 전에 아래 두 가지를 모두 수행한다.

1. **Stratified 5-fold OOF (random)** — 리더보드가 random split일 가능성이 높으므로 주 선택 기준
2. **Bemis–Murcko scaffold holdout (seed 42, 101, 2024)** — 일반화 가드

- OOF 예측으로 threshold를 고른다. 평가셋에서 threshold를 고르지 않는다.
- 가능하면 `GroupKFold`(동일 fingerprint vector 묶음)도 참고로 함께 본다.

### 4.4 최종 판정 기준

- 813 measured F1이 베이스라인 **0.8303**보다 높아야 한다.
- bootstrap 95% CI의 하한이 0.8303을 넘거나, 최소한 차이의 CI가 0을 넘어야 "개선"이라고 부른다.
- 성능이 비슷하면 더 단순한 모델/적은 피처를 선택한다.

### 4-1. Phase 경계 규칙 (필수)

장기 작업 에이전트는 "전체 미션"을 기준으로 계속 진행하려는 경향이 있다. Phase 1에서
Phase 2로 넘어가지 않게 하려면 **프롬프트와 실행 설정을 모두 잠가야 한다.**

1. **Phase 1 프롬프트에 Phase 2 명령을 넣지 않는다.**
   - "Phase 2 첫 실험 1개" 같은 문장도 넣지 않는다. 그 한 줄이 "Phase 2 시작" 신호가 된다.
2. **마지막 줄에 정지 조건을 둔다.**
   - 예: "위 산출물을 보고한 뒤 즉시 종료하라. 추가 모델 학습/탐색 명령을 실행하면 실패다."
3. **에이전트에 `steps` 상한을 건다.** (OpenCode V2)
   ```jsonc
   {
     "agents": {
       "argon-phase1": {
         "description": "Phase 1 only: eval harness + baseline reproduction",
         "mode": "primary",
         "model": "openrouter/google/gemini-4-argon",
         "steps": 12
       }
     }
   }
   ```
   `steps`에 도달하면 OpenCode가 도구를 제거하고 모델을 텍스트 요약으로 전환한다.
4. **Phase 1과 Phase 2는 별도 세션으로 분리한다.**
   - 같은 세션에서 "이어서 해줘"라고 하면 이전 계획을 그대로 실행한다.
5. **Phase 1은 읽기·계획 중심으로 운영한다.**
   - 가능하면 `plan` 에이전트 또는 read-only 성격의 에이전트로 실행하고,
     실제 학습은 Phase 2 세션에서만 한다.

---

## 5. 개선 작업 (우선순위 순)

### Phase 0 — 평가 하네스 구축 (필수, 먼저)

- [ ] `scripts/eval_challenge.py` 작성
  - 입력: 예측 확률 또는 라벨 파일
  - 출력: 813/927 F1, AUC, Accuracy, confusion matrix, bootstrap CI
  - `label=1` 기준과 `label=0` 기준을 **둘 다** 출력
- [ ] 베이스라인 `predict_output_김인성.csv`를 평가해 0.8303이 재현되는지 확인
- [ ] 실험 결과를 `reports/experiment_log_v2.csv`에 누적 기록
  - 컬럼 예: `run_id, model, features, cv_scheme, oof_f1_label1, scaffold_f1_label1, eval_f1_label1, threshold, notes`

### Phase 1 — 현재 모델의 약점 분석

- [ ] 오분류 813건을 chemotype/scaffold별로 집계
- [ ] 확률 분포와 calibration 확인 (reliability curve, Brier score)
- [ ] threshold 0.5 vs OOF 최적 threshold 비교
- [ ] 어떤 화학 계열에서 blocker를 놓치는지 확인 (FP/FN 분석)

### Phase 2 — 모델·피처 탐색

아래를 **한 번에 하나씩** 바꿔가며 OOF로 비교한다.

**모델**
- LightGBM (현재)
- XGBoost
- CatBoost
- HistGradientBoosting
- ExtraTrees
- RandomForest
- LogisticRegression (L2/L1, `class_weight='balanced'`)
- SVM (RBF, subsample)

**피처 세트**
- FCFP only + 기존 기술자 (E3)
- ECFP + FCFP + PTFP + 기존 기술자 (EDA baseline)
- E6 (현재): FCFP + 기존 + hERG 도메인 7그룹
- E7: FCFP + 전체 RDKit descriptor
- Mordred 2D (전체 및 train-only selection)
- 도메인 파생변수:
  - CAD 규칙 (`clogp > 3` and basic amine)
  - basic amine / amide / sulfonamide / aromatic N 개수
  - TPSA, MolLogP, HBD/HBA, RotatableBonds, aromatic ring count
  - 분자 크기·표면적·전하 관련 descriptor
- 상수/중복 fingerprint bit 제거 (train 기준)

**하이퍼파라미터**
- Optuna 30~50 trials (가능하면 100)
- 목적함수: OOF F1(label=1) 또는 OOF AUC + threshold tuning
- 조기 종료 사용 시 validation fold만 사용
- seed 3개 이상 반복해 평균±표준편차 기록

### Phase 3 — 앙상블

- 상위 3~5개 모델의 확률을 **rank average** 또는 가중 평균
- 가중치는 OOF에서만 최적화 (평가셋 사용 금지)
- 앙상블이 단일 모델보다 OOF F1이 높을 때만 채택
- Calibration (isotonic/Platt) 후 threshold를 다시 선택

### Phase 4 — Threshold 최적화

- OOF 예측에서 F1(label=1) 최대 threshold를 탐색
- 813 평가셋에서는 threshold를 고정한 채 최종 성능만 측정
- threshold 민감도 표를 함께 보고 (예: 0.40~0.70)

### Phase 5 — 최종 검증 및 제출

- [ ] 최종 모델을 train 전체(또는 train+val)로 재학습
- [ ] `predict_output_김인성_v2.csv` 생성 (927행, `SMILES,label`, 입력 순서 유지)
- [ ] 813 measured F1, 927 all F1, CI, confusion matrix 보고
- [ ] 베이스라인 대비 개선폭과 함께 **실패한 실험도** 기록
- [ ] 최종 보고서: `reports/improvement_report_v2.md`

---

## 6. 실험 설계 권장 사항

### 6.1 데이터 분할

- **Random 5-fold OOF**: 리더보드 근사. 주 모델 선택 기준.
- **Scaffold holdout**: seed 42/101/2024. 일반화 가드.
- **GroupKFold (`fp_group`)**: 동일 fingerprint(입체이성질체) 누수 방지. 참고용.

세 결과가 크게 다르면 random만 믿지 말고, scaffold 성능도 함께 보고한다.

### 6.2 절대 하지 말 것

- 평가셋 813건으로 early stopping / feature selection / threshold tuning
- `predict_input`의 8개 완전 중복 분자 라벨을 복사해 제출
- low-confidence 114건을 정답처럼 사용
- 테스트 점수만 보고 모델을 고르는 행위
- 기존 제출 파일 덮어쓰기

### 6.3 클래스 불균형

- 54:46으로 경미하다. `class_weight='balanced'`는 선택적으로 실험하되, F1(label=1) 기준으로만 채택.
- SMOTE/리샘플링은 fingerprint 이진 벡터에서 권장하지 않는다.

### 6.4 시간 예산

| 단계 | 예상 |
|---|---|
| Phase 0–1 | 1시간 |
| Phase 2 (모델 5종 × 피처 5세트 × 30 trials) | 3~5시간 |
| Phase 3–4 | 1~2시간 |
| Phase 5 | 1시간 |

LightGBM 1회 학습은 수십 초, RDKit descriptor 계산은 전체 8,349행 기준 1~2분이다.
캐시(`.cache/herg_model`, `data/chembl`)를 활용한다.

---

## 7. 환경

- OS: Windows 11
- Python: `.venv/Scripts/python.exe` (3.11)
- 설치됨: numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, rdkit 2026.9.1, lightgbm 4.7.0, joblib 1.6.0
- 추가 설치 필요:
  ```bash
  uv pip install --python .venv/Scripts/python.exe optuna xgboost catboost shap mordredcommunity
  ```
- `pyproject.toml`은 Python >=3.12를 요구하지만, 현재 `.venv`는 3.11에서 동작한다.
  `uv sync`로 환경을 재생성하지 말고, 기존 `.venv`에 패키지만 추가한다.

### 주요 경로

```
data/train.csv                          # 학습 데이터
data/predict_input.csv                  # 예측 대상 927행
data/predict_input_chembl_labeled.csv   # 평가용 라벨 (학습 금지)
data/predict_output_chembl.csv          # ChEMBL 라벨 참고용
artifacts/best_model.joblib             # 현재 모델
predict_output_김인성.csv                # 현재 제출 (보존)
reports/verification_benchmark.json     # 베이스라인 검증
reports/chembl_label_reconstruction.json# 라벨 복원 요약
scripts/verify_benchmark.py             # 베이스라인 재현
scripts/make_submission.py              # 제출 생성
scripts/fetch_chembl_labels.py          # 라벨 복원 재현
src/herg_model/                         # 기존 ML 스캐폴드
```

### 현재 모델 로드 예시

```python
import sys
sys.path.insert(0, "src")
from herg_model.config import load_config
from herg_model.predict import load_artifact, compute_blocks_from_smiles
from herg_model.models import positive_scores

cfg = load_config()
artifact = load_artifact("artifacts/best_model.joblib")
names, X = compute_blocks_from_smiles(smiles_list, artifact["blocks"], cfg)
X = artifact["pipeline"].transform(X)
p_blocker = positive_scores(artifact["model"], X)
```

---

## 8. 산출물 (필수)

| 파일 | 내용 |
|---|---|
| `scripts/eval_challenge.py` | 813/927 평가 + bootstrap CI |
| `reports/experiment_log_v2.csv` | 모든 실험의 OOF/scaffold/eval 지표 |
| `reports/improvement_report_v2.md` | 베이스라인 대비 개선 결과, 실패 사례, 한계 |
| `predict_output_김인성_v2.csv` | 최종 제출 후보 (927행) |
| 필요 시 `artifacts/best_model_v2.joblib` | 새 최종 모델 |

### 최종 보고에 반드시 포함할 것

1. 813 measured F1(label=1) — baseline 0.8303 대비 증감, 95% CI
2. 927 all F1(label=1) — 참고
3. OOF F1 / scaffold F1 — 모델 선택 근거
4. 사용한 피처·모델·threshold·seed
5. 평가셋을 학습에 쓰지 않았다는 명시
6. 실패한 실험과 그 이유
7. 재현 명령

---

## 9. 성공 기준 (Definition of Done)

- [ ] `predict_output_김인성_v2.csv`가 927행, `SMILES,label`, 정수 0/1, 입력 순서 유지
- [ ] 813 measured F1(label=1)이 **0.8303 초과**
- [ ] 최소 2개 이상의 독립적인 검증(random OOF + scaffold)에서 개선이 재현
- [ ] bootstrap CI와 혼동행렬 포함
- [ ] 평가 라벨을 학습에 사용하지 않았음을 코드/로그로 입증
- [ ] 기존 파일을 덮어쓰지 않음
- [ ] 한국어 보고서 작성

성능이 0.8303을 넘지 못하면, **그 사실을 그대로 보고**하고 가장 안정적인 모델을 제출 후보로 남긴다.
과장된 단일 split 수치로 "개선"이라고 보고하지 않는다.

---

## 10. 첫 30분 체크리스트

1. `data/train.csv`, `data/predict_input.csv` 로드 및 shape/라벨 분포 확인
2. `scripts/verify_benchmark.py` 실행해 베이스라인 수치 재현
3. `predict_output_김인성.csv`를 `data/predict_input_chembl_labeled.csv`로 평가해 0.8303 확인
4. `scripts/eval_challenge.py` 뼈대 작성
5. random 5-fold OOF 하네스 구축
6. 베이스라인 모델을 OOF로 재학습해 OOF F1 기록
7. 결과를 보고하고 **즉시 종료** (Phase 2는 승인 후 별도 세션에서 시작)

---

## 11. 참고: 이미 확인된 사실 (반복 실험 방지)

- EDA LogisticRegression(전체 지문+기술자)은 scaffold test F1(1) 0.7412로 현재 모델보다 낮다.
- 현재 모델(E6)은 동일 split에서 EDA 베이스라인보다 F1(1) +0.054 우수하다 (CI [+0.031, +0.080]).
- 기존 보고서의 F1 0.6677은 **blocker 기준**이므로 챌린지 점수와 무관하다.
- E6의 저장 threshold 0.7139(blocker)와 0.5는 813 평가셋에서 F1 차이가 0.006 수준이다.
- RDKit descriptor 전체 추가(E7)는 E6와 거의 동일했다. Mordred 전체(E8)는 오히려 낮았다.
- 팀 노트북 기준 random 5-fold F1은 0.816, 구조 클러스터 holdout은 0.784였다.
  (현재 모델의 0.83은 이보다 높은 수치이므로, 단순히 "팀 방식이 더 좋다"고 가정하지 말 것.)

---

*이 문서는 Gemini 4 Argon에게 그대로 전달하는 것을 전제로 작성되었다. 경로와 수치는 2026-10-20 저장소 상태 기준이다.*
