# hERG blocker 예측 — 재현 가능한 ML 스캐폴드

`docs/ml_goal.md`의 요구사항(§1–§14)을 구현한 실험 스캐폴드입니다.
목표는 **높은 점수 하나**가 아니라, **fingerprint 기준선 → 도메인 descriptor → Mordred**로
이어지는 표현(representation)의 기여를 **동일한 scaffold split · 동일한 예산**에서 통제 비교하고,
그 결과를 재현 가능하게 남기는 것입니다.

---

## 요약 — 이 저장소에 실제로 커밋된 실행

| 항목 | 값 |
|---|---|
| 예산 모드 | `quick` (Optuna 5 trials/모델, 모델 3종) |
| 학습된 (실험, 모델) 조합 | 24개 설정 × 모델 = 50행 |
| 선택된 fingerprint | **FCFP** (E0/E1/E2 validation BA 비교) |
| 최종 모델 | **E6 / LightGBM** (최고 validation BA 0.7799) |
| Validation (임계값 0.5) | BA 0.7799 · ROC-AUC 0.8467 · PR-AUC 0.8133 |
| **Test (1회, 임계값 0.7139)** | **BA 0.7320 · ROC-AUC 0.8475 · PR-AUC 0.8257 · MCC 0.5023** |
| Test (임계값 0.5) | BA 0.7614 · ROC-AUC 0.8475 |
| Seed 변동성 (3 seed) | BA 표준편차 0.004 ~ 0.010 |
| Random split 참고값 (E3) | BA 0.8146 vs scaffold 0.7653 → **+0.049** 낙관 편향 |

핵심 결론:

1. **fingerprint 계열:** FCFP > ECFP > PTFP (validation BA 0.7653 / 0.7397 / 0.7236, lightgbm).
   fingerprint 단독(E0b/E1b/E2b)은 descriptor를 더한 쪽보다 낮다.
2. **descriptor 확장:** fingerprint 기준선(E3, 0.7653) 위에 hERG 도메인 그룹을 더한 **E6가 0.7799**로
   가장 높았고, RDKit 전체(E7, 0.7778)와 거의 같으면서 feature는 더 적다(1,223 vs 1,253).
   Mordred 전체(E8, 0.7685)·선별(E8b, 0.7644)·RDKit+Mordred 결합(E9, 0.7625)은 기준선을 넘지 못했다.
3. **그룹별 ablation:** 단일 그룹을 더한 효과는 모두 |ΔBA| < 0.01로 작다(최대 `aromaticity_shape`
   +0.0037). 특정 descriptor 한 그룹이 결정적이라기보다 여러 그룹을 함께 넣었을 때 소폭 개선된다.
4. **임계값 전이:** validation에서 고른 임계값(0.7139)의 test BA가 기본 0.5보다 0.0294 낮다.
   순위 지표(ROC-AUC/PR-AUC)는 validation과 test에서 거의 같으므로(0.8467 → 0.8475),
   **모델의 순위 능력은 안정적이지만 임계값 선택은 잡음에 민감**하다.
5. **일반화 격차:** 같은 E3 설정을 random split에서 돌리면 BA가 +0.049 올라간다. scaffold split이
   더 어려운 평가이며, random split 수치를 성능으로 보고하면 안 된다.

실행하지 않은 항목은 §9에 명시했습니다(full 모드, 나머지 4개 모델 계열).

---

## 0. 고정된 라벨 의미 (절대 뒤집지 않음)

| 원본 `label` | 의미 | 이 프로젝트의 클래스 |
|---|---|---|
| `0` | hERG 차단 활성 있음 | **blocker = 양성 클래스(positive)** |
| `1` | hERG 차단 활성 낮음 | non-blocker = 음성 클래스 |

코드 전체에서 `y_blocker = (label == 0).astype(int)` 로 변환한 뒤 학습·평가합니다.
따라서 **ROC-AUC / PR-AUC / Precision / Recall / F1 / MCC 는 모두 blocker 기준**입니다.

> ⚠️ 저장소의 기존 EDA(`README.md`, `toxicity_eda.ipynb`)는 `label = 1`을 F1의 positive로
> 사용했습니다. **두 관점은 "독성 = label 0"이라는 사실은 같지만 positive 클래스 정의가 반대**이므로,
> 기존 EDA의 F1과 이 보고서의 F1을 직접 비교하면 안 됩니다.

예측 종말점은 **hERG 차단 활성**이며, 임상적 QT 연장·부정맥을 직접 예측한다고 해석하지 않습니다.

---

## 1. 구조

```text
.
├── configs/experiments.yaml          # 모든 실험 설정의 단일 출처
├── src/herg_model/
│   ├── config.py                     # 설정 로딩·검증, 환경 스냅샷, 캐시 키
│   ├── data.py                       # train.csv 로딩, 라벨 매핑, 데이터 품질 통계
│   ├── chemistry.py                  # 파싱, InChIKey, Bemis–Murcko scaffold, 표준화(민감도용)
│   ├── fingerprints.py               # ECFP/FCFP/PTFP 재생성 + 원본 컬럼 일치 검증
│   ├── descriptors.py                # RDKit 251개 2D descriptor + hERG 도메인 그룹 + Mordred
│   ├── splitting.py                  # scaffold split, random split, 누수 검사
│   ├── preprocessing.py              # train-only 상수/중복 제거, 결측 대체, 스케일링
│   ├── feature_selection.py          # train-only feature selection
│   ├── models.py                     # 모델 zoo, 기본값, Optuna 탐색 공간
│   ├── metrics.py                    # positive=blocker 지표, threshold 선택, cluster bootstrap
│   ├── experiments.py                # feature block, E0–E11 정의, 실험 실행기
│   ├── explainability.py             # permutation importance, SHAP, 오분류·scaffold 분석
│   ├── figures.py                    # ROC/PR, ablation, 혼동행렬, 중요도, scaffold 분포
│   ├── reporting.py                  # markdown 보고서 생성
│   └── predict.py                    # 아티팩트 저장/로드, 새 SMILES 추론
├── scripts/
│   ├── inspect_data.py               # 단계 A
│   ├── calculate_descriptors.py      # 단계 C
│   ├── run_experiments.py            # 단계 B/D/E
│   ├── train_final_model.py          # 단계 F (test는 여기서 1회만)
│   └── analyze_results.py            # 단계 E/F (분석·그림·최종 보고서)
├── tests/                            # pytest 26개
├── artifacts/                        # 실험 manifest, 최종 모델, 메타데이터
├── reports/                          # 보고서·지표 CSV·그림
└── data/train.csv                    # 원본 (저장소에 포함하지 않음)
```

---

## 2. 설치

이 저장소는 `uv`로 환경을 관리합니다.

```bash
uv venv .venv --python 3.12
# Windows
uv pip install --python .venv/Scripts/python.exe -e ".[all,dev]"
# macOS / Linux
uv pip install --python .venv/bin/python -e ".[all,dev]"
```

`[all]`은 선택 확장입니다: `lightgbm`, `xgboost`, `mordredcommunity`, `optuna`, `shap`.
하나라도 설치되지 않아도 파이프라인은 **해당 단계만 건너뛰고 사유를 기록**합니다
(`unavailable_models()`, `compute_mordred_descriptors()`의 `status`).

### 검증된 실행 환경

| 항목 | 버전 |
|---|---|
| Python | 3.12.14 |
| OS | Windows 11 (10.0.26200) |
| RDKit | 2026.9.1 |
| scikit-learn | 1.9.1 |
| numpy / pandas / scipy | 2.5.4 / 3.0.6 / 1.18.1 |
| LightGBM / XGBoost | 4.7.0 / 3.4.1 |
| Mordred (`mordredcommunity`) | 2.0.7 |
| Optuna / SHAP | 5.0.0 / 0.54.1 |

---

## 3. 실행

`data/train.csv`를 `data/`에 둔 뒤, 아래 순서로 실행합니다.

```bash
# 1) 데이터·구조·fingerprint 재현성 검사
python scripts/inspect_data.py

# 2) descriptor 계산 (RDKit + Mordred, 캐시됨)
python scripts/calculate_descriptors.py

# 3) 통제 실험 E0–E11 + ablation + seed 안정성
python scripts/run_experiments.py

# 4) validation에서 고른 설정을 동결하고 test를 1회 평가 (+ 아티팩트 저장)
python scripts/train_final_model.py

# 5) ablation·중요도·오분류·그림·최종 보고서 생성
python scripts/analyze_results.py
```

선택 단계:

```bash
# fingerprint 비교(E0–E2)를 마친 뒤, 같은 실험을 random split에서 한 번 더 돌려
# "scaffold split과의 격차"를 참고용으로 남긴다 (결과: reports/split_comparison_results.csv)
python scripts/run_experiments.py --random-reference --only E3
```

- 예산 모드: `--mode quick`(기본) / `--mode full`
  - `quick`: 모델 3종, Optuna **5 trials**/모델 → 실제로 커밋된 실행
  - `full`: 모델 6종, Optuna **40 trials**/모델
- 부분 실행: `python scripts/run_experiments.py --only E0 E1 E2`
- 테스트: `python -m pytest tests -q`

> **실행 순서 주의:** `analyze_results.py`는 `train_final_model.py`가 남긴 test 결과를 읽어
> 최종 보고서에 포함합니다. test를 아직 평가하지 않았다면 그 사실을 보고서에 명시합니다.

---

## 4. 설정 (`configs/experiments.yaml`)

| 키 | 의미 |
|---|---|
| `data.label_meanings` | **고정**. 로딩 시 `0=blocker`가 아니면 즉시 `ValueError` |
| `data.fingerprints.*` | 블록별 generator/params (원본 컬럼과 1:1 재현) |
| `descriptors.groups` | ablation 단위가 되는 도메인 그룹 |
| `descriptors.mordred.max_missing_rate` | 학습 split 결측률이 이 값을 넘으면 제거 |
| `split.*` | scaffold split 비율·seed 목록·클래스 균형 |
| `optimization.n_trials` | `{quick, full}` 예산 |
| `optimization.models` | 모드별 학습할 모델 목록 |
| `evaluation.bootstrap` | scaffold 단위 cluster bootstrap 횟수 |
| `experiment.feature_selection` | E10/E8b/E11의 선택 방식·개수 |
| `runtime.n_jobs`, `runtime.cache` | 병렬도·캐시 사용 |

---

## 5. 실험 정의

| ID | 입력 피처 | 목적 |
|---|---|---|
| E0 | `ecfp` + 기존 descriptor | 기준선 |
| E1 | `fcfp` + 기존 descriptor | fingerprint 계열 비교 |
| E2 | `ptfp` + 기존 descriptor | fingerprint 계열 비교 |
| E0b/E1b/E2b | 각 fingerprint 단독 | fingerprint 자체의 정보량 |
| E3 | validation에서 고른 최적 fingerprint + 기존 descriptor | 이후 실험의 기준선 |
| E3c | ECFP+FCFP+PTFP 결합 | 결합 표현의 효과 |
| E4 | RDKit descriptor만 | descriptor 단독 정보량 |
| E5 | 최적 fp + `physicochemical` | 일반 물성 추가 효과 |
| E6 | 최적 fp + hERG 도메인 6개 그룹 | 도메인 지식의 기여 |
| E7 | 최적 fp + RDKit 전체 | 전체 descriptor 확장 |
| E8 | 최적 fp + Mordred 전체 | Mordred 확장 |
| E8b | 최적 fp + Mordred + selection | 선별 Mordred의 효과 |
| E9 | 최적 fp + RDKit + Mordred | 결합 표현 |
| E10 | E7 + train-only feature selection | 피처 선택 효과 |
| E11 | E10과 동일 피처 + 전체 모델 | 모델 구조와 피처 효과 분리 |
| `A_<group>` | E3 + 단일 descriptor 그룹 | 그룹별 ablation |

모든 실험은 **같은 `Splits` 객체**, **같은 탐색 예산**, **같은 전처리 규칙**을 사용합니다.

---

## 6. 누수 방지 프로토콜

1. **분할 단위 = Bemis–Murcko scaffold.** 동일 scaffold가 train/val/test에 걸치면
   `assert_no_leakage()`가 즉시 실패합니다(실제 비율과 클래스 분포는 그대로 보고).
   무환식 분자는 generic scaffold를 사용해 개별 그룹으로 흩어지지 않게 합니다.
2. **전처리·피처 선택·하이퍼파라미터 탐색은 train 내부에서만** 적합합니다.
   `FeaturePipeline`은 train에서만 `keep_idx`/`fill_values`/`scaler`를 학습합니다.
3. **임계값은 validation에서만** 선택합니다.
4. **test는 `train_final_model.py`에서 1회만** 평가하며, 그 전까지 어떤 선택에도 쓰지 않습니다.
5. **random split은 참고용**으로만 별도 보고합니다(scaffold 중복이 실제로 발생함을 수치로 제시).
6. **fingerprint 유사도를 동일성 증거로 쓰지 않습니다.** descriptor/fingerprint 재현성은
   bit 단위 일치로만 판정합니다.

---

## 7. 산출물

| 파일 | 내용 | git 포함 |
|---|---|---|
| `reports/data_quality_report.md` | 컬럼·라벨·파싱·fingerprint 재현성·중복·분할·누수 | ✅ |
| `reports/descriptor_report.md` | descriptor 정의·성공률·결측·중복·상관 | ✅ |
| `reports/final_report.md` | E0–E11 결과, ablation, 중요도, 오분류, 최종 test, 한계 | ✅ |
| `reports/experiment_results.csv` | (실험 × 모델) validation 지표 + CI | ✅ |
| `reports/ablation_results.csv` | descriptor 그룹별 추가 효과 | ✅ |
| `reports/feature_importance.csv` | permutation importance | ✅ |
| `reports/error_analysis.csv` | 오분류 사례 (SMILES 포함) | ❌ (로컬) |
| `reports/scaffold_performance.csv` | scaffold별 성능 (SMILES 포함) | ❌ (로컬) |
| `reports/figures/*.png` | ROC/PR, ablation, 혼동행렬, 중요도, scaffold 분포 | ✅ |
| `artifacts/best_model.joblib` | 최종 모델 + 전처리 파이프라인 (5 MB) | ✅ |
| `artifacts/model_metadata.json` | 라벨 의미, 피처 순서, 설정, 라이브러리 버전, 임계값 | ✅ |
| `artifacts/experiment_manifest.json` | 실험별 설정·seed·예산·실행 상태 | ✅ |

> **재배포 제한:** `train.csv`와 SMILES를 포함한 파생 파일은 저장소에 올리지 않습니다
> (`.gitignore`). 보고서·지표·그림·코드·모델 아티팩트만 공개됩니다. `best_model.joblib`은 학습된
> 모델과 전처리 파이프라인, feature 이름만 담고 원본 SMILES는 담지 않습니다.

---

## 8. 추론

```python
import sys; sys.path.insert(0, "src")
from herg_model.config import load_config
from herg_model.predict import load_artifact, predict_smiles

cfg = load_config()
artifact = load_artifact("artifacts/best_model.joblib")
print(predict_smiles(["CN1CCC[C@H]1c2cccnc2", "CC(=O)Oc1ccccc1C(=O)O"], artifact, cfg))
```

반환 컬럼:

| 컬럼 | 의미 |
|---|---|
| `blocker_probability` | 모델이 산출한 blocker 확률 (보정되지 않음) |
| `predicted_label` | **원본 라벨 체계** (0 = blocker) |
| `predicted_class` | `blocker` / `non_blocker` |
| `threshold` | validation에서 선택된 임계값 |

확률은 해당 모델의 출력이며, **보정(calibration)과 외부 검증 없이 임상적 독성 확률로 해석하면 안 됩니다.**

---

## 9. 알려진 한계

- 라벨이 대략 `IC50 < 10 µM` 규칙으로 생성되었을 가능성이 있으나, 모든 행이 단일 assay·단일
  규칙이라고 단정할 수 없습니다. 라벨 노이즈가 남아 있습니다.
- scaffold split 성능은 "학습에 없는 chemotype"에 대한 성능이므로 random split과 직접 비교할 수 없습니다.
- ECFP 블록은 8,344/8,349 행에서 bit 단위로 재현됩니다. 나머지 5행은 모두 중수소(`[2H]`) 표기를
  포함하며, 현재 RDKit으로는 재현되지 않아 원본 컬럼을 그대로 사용합니다(보고서에 목록 기록).
- Mordred 2D descriptor 1,613개 중 168개는 학습 split 결측률 10%를 넘어 제거됩니다.
- feature importance는 상관 수준의 설명이며 작용 기전의 증거가 아닙니다.

### 실행 범위 (실행하지 않은 것)

- **`full` 모드는 실행하지 않았습니다.** 커밋된 결과는 `quick`(5 trials, 3개 모델)입니다.
  `full`(40 trials, 6개 모델)은 `--mode full`로 같은 명령을 다시 실행하면 됩니다.
- 모델 계열 중 `extra_trees`, `hist_gradient_boosting`, `xgboost`, `svm`은 실행하지 않았습니다.
- **SMOTE 등 오버샘플링은 사용하지 않았습니다.** 클래스 불균형은 class weight와 임계값 조정만
  비교했고(`reports/class_imbalance_results.csv`), 합성 샘플은 fingerprint 공간의 화학적 타당성이
  보장되지 않아 근거 없이 도입하지 않았습니다.
- tautomer/염 정규화 민감도 분석은 데이터 품질 보고서의 중복 통계로만 제시하고, 학습 경로에는
  넣지 않았습니다(구조·라벨 의미 변경 위험).
- 외부 데이터셋에 대한 검증은 하지 않았습니다. `herg_validation/`의 라벨 검증 결과는 별개 작업입니다.
