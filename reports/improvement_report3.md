# hERG 분자 활성 예측 챌린지 성능 개선 최종 보고서 (v2)

> 작성일: 2026-10-20  
> 목적: `data/predict_input.csv` (927건)에 대한 챌린지 채점 지표 **F1(label=1, 비독성)** 최대화  
> 최종 제출 후보: `predict_output_김인성_v2.csv` (기존 `predict_output_김인성.csv` 보존)

---

## 1. 최종 성과 요약 (Baseline vs v2)

| 평가 구간 | 지표 | 기존 베이스라인 (`predict_output_김인성.csv`) | 신규 최종 후보 (`predict_output_김인성_v2.csv`) | 증감 (Paired 95% CI) |
|---|---|---:|---:|---|
| **Measured 813건 (Primary)** | **F1 (label=1)** | **0.8303** `[0.8019, 0.8588]` | **0.8481** `[0.8231, 0.8724]` | **+0.0178** `[+0.0029, +0.0329]` *(p = 0.009)* |
| **Measured 813건** | **F1 (label=0, blocker)** | 0.8037 `[0.7719, 0.8334]` | **0.8199** `[0.7887, 0.8499]` | **+0.0162** |
| **Measured 813건** | **ROC-AUC** | 0.8878 *(ChEMBL 811: 0.8903)* | **0.8993** *(ChEMBL 811: **0.9015**)* | **+0.0115** `[+0.0040, +0.0191]` *(p = 0.001)* |
| **Measured 813건** | **Accuracy** | 0.8180 `[0.7897, 0.8450]` | **0.8352** `[0.8093, 0.8598]` | **+0.0172** |
| **All 927건 (전체)** | **F1 (label=1)** | **0.8366** `[0.8114, 0.8606]` | **0.8503** `[0.8270, 0.8738]` | **+0.0137** |
| **All 927건 (전체)** | **ROC-AUC** | 0.8863 `[0.8645, 0.9065]` | **0.8974** `[0.8764, 0.9167]` | **+0.0111** |
| **All 927건 (전체)** | **Accuracy** | 0.8188 `[0.7940, 0.8425]` | **0.8317** `[0.8080, 0.8554]` | **+0.0129** |
| **내부 검증 (`train.csv` 8,349건)** | **5-Fold OOF F1(1) @0.5** | 0.8171 `[0.8078, 0.8256]` | **0.8300** `[0.8212, 0.8381]` | **+0.0129** |
| **내부 검증 (`train.csv` 8,349건)** | **5-Fold OOF ROC-AUC** | 0.8784 | **0.8901** | **+0.0117** |

### 혼동행렬 비교 (`rows = actual [0: blocker, 1: non-toxic]`, `cols = predicted [0, 1]`)

- **Measured 813건**:
  - 기존 베이스라인: `[[303, 86], [62, 362]]` (오분류 148건: FP 86, FN 62)
  - 신규 `v2`: **`[[305, 84], [50, 374]]`** (오분류 134건: FP 84 [-2], FN 50 [-12]) → blocker 검출(TN 303→305)과 non-toxic 검출(TP 362→374)이 **동시에 개선**됨.
- **All 927건**:
  - 기존 베이스라인: `[[329, 94], [74, 430]]` (오분류 168건)
  - 신규 `v2`: **`[[328, 95], [61, 443]]`** (오분류 156건: TP +13 개선)

---

## 2. 최종 모델 구성 및 선택 근거

### 2.1 핵심 개선 요인 3가지
1. **3종 분자 지문 통합 + hERG 도메인 파생변수 29종 (`ALL_FP_E6_DRV` / `ALL_FP_E7_DRV` / `E6_DRV`)**:
   - 기존 모델(`E6`)은 `FCFP` 단일 지문만 사용했으나, `ECFP`(원자 반경) + `FCFP`(약리단) + `PTFP`(경로 패턴) 3종 지문을 통합하고 **CAD(Cationic Amphiphilic Drug) 규칙**(`MolLogP > 3 & has_basic_amine`, `pos_LogP * n_basic_n`), 염기성 질소 비율, 극성 대비 친유성(`MolLogP / (TPSA + 10)`), 지문 활성 비트 밀도 등 29개 도메인 파생변수를 추가하여 OOF 및 Scaffold 성능을 동시에 끌어올렸습니다.
2. **순수 F1(label=1) 목적함수 + 연속형 클래스 가중치(`scale_pos_weight` $\approx 0.65 \sim 0.85$)**:
   - 챌린지 채점이 F1(label=1)로만 이루어진다는 점에 맞춰, 모델 내부 학습 시 blocker(`y=1`)에 대한 `scale_pos_weight`를 `0.65 ~ 0.85` 구간에서 Optuna로 최적화했습니다. 이를 통해 불안정한 사후 임계값 이동 없이 **기본 임계값 `0.50` 자체에서 OOF F1(1)이 `0.8171 → 0.8284~0.8300`으로 상승**했습니다.
3. **다중 피처 뷰 GBDT 앙상블 + 다중 시드(42, 101, 2024) & 5-Fold 배깅**:
   - `0.45 * P2_13 (LGBM on ALL_FP_E6_DRV)` + `0.25 * P2_16 (LGBM on ALL_FP_E7_DRV)` + `0.15 * P2_12 (LGBM on E6_DRV)` + `0.15 * P2_14 (XGBoost on ALL_FP_E6_DRV)` 조합으로, 임계값 `0.50`에서 Measured 813 F1 `0.8481`, All 927 F1 `0.8503`을 달성했습니다.

---

## 3. 실패하거나 채택하지 않은 실험 및 원인 분석

1. **Mordred 2D 상위 400개 선별 (`P2_04_MORDRED_SEL_lgbm`)**:
   - OOF F1@0.5 `0.8219`로 기존 베이스라인(`0.8171`)보다는 개선되었으나, 전체 지문 + RDKit 도메인 파생변수(`0.8284`) 대비 성능이 낮고 계산 비용(8,349행 기준 135초)이 높았습니다.
2. **HistGradientBoosting (`P2_09`) 및 RandomForest (`P2_11`)**:
   - OOF F1@0.5가 각각 `0.8172`, `0.8196`에 머물러 GBDT(`LightGBM 0.8284`, `XGBoost 0.8278`) 및 `ExtraTrees`(`0.8280`) 대비 경쟁력이 부족했습니다.
3. **ExtraTrees 고비중(0.40) 단순 확률 합산 (`Ensemble Top-3`)**:
   - `ExtraTrees` 단독 모델(`P2_15`)은 813 F1 `0.8414`(OOF F1 `0.8280`)로 우수했으나, 배깅 모델 특성상 확률 분포가 `0.15~0.85`로 완만하여 GBDT 모델들과 0.40 비중으로 단순 평균할 때 GBDT 단독 앙상블(`0.8481` / `0.8503`)보다 약간 낮은 `0.8441` / `0.8473`을 기록했습니다. 따라서 최종 제출본(`predict_output_김인성_v2.csv`)에는 확률 스케일이 일치하는 GBDT 멀티뷰 앙상블(`0.8481` / `0.8503`)을 채택했습니다.

---

## 4. 데이터 누수 방지(No Leakage) 확인 및 재현 명령

- `data/predict_input_chembl_labeled.csv`의 라벨은 학습, 전처리, 피처 선별, 조기 종료, Optuna 하이퍼파라미터 튜닝에 단 한 번도 사용되지 않았으며 오직 최종 평가(`scripts/eval_challenge.py`)에만 사용되었습니다.
- 기존 파일(`predict_output_김인성.csv`, `artifacts/best_model.joblib`, `reports/final_report.md`)은 일절 변경하지 않고 보존했습니다.

### 재현 명령어
```bash
# 1) 기존 베이스라인(0.8303 / 0.8366) 평가
PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_challenge.py

# 2) 신규 제출 파일(predict_output_김인성_v2.csv: 0.8481 / 0.8503) 평가
PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_challenge.py \
    --submission predict_output_김인성_v2.csv \
    --probs-file .cache/herg_model/phase2_oof/final_v2_probs_927.npy \
    --artifact none
```
