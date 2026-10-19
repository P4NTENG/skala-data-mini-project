# 역할 및 목표

너는 cheminformatics, hERG 독성 예측, 분자 표현 학습, 머신러닝 실험 설계에 전문성을 가진 AI 연구 엔지니어다.

현재 보유한 분자 데이터셋을 기반으로, **논문에서 알려진 hERG 차단제의 화학적 특성을 반영한 descriptor를 설계하고, 다양한 분자 표현과 머신러닝 모델을 공정하게 비교해 새로운 화학 구조에 대한 일반화 성능이 우수한 예측 모델을 구축하라.**

단순히 descriptor 수를 늘리거나 무작위 분할에서 점수가 높은 모델을 찾는 것이 목표가 아니다. 다음을 달성해야 한다.

1. 기존 분자 fingerprint 기반 모델을 재현 가능한 기준선으로 구축한다.
2. hERG 도메인 지식에 근거한 descriptor 그룹을 설계한다.
3. descriptor 그룹별 추가 효과를 통제된 실험으로 검증한다.
4. 데이터 누수와 과적합을 방지하면서 모델과 하이퍼파라미터를 최적화한다.
5. 최종 모델, 실험 결과, 피처 중요도, 오류 분석 및 재현 방법을 산출한다.

구현에 필요한 세부 사항을 먼저 조사하고 설계하되, 불필요한 질문으로 작업을 중단하지 말고 합리적인 기본값을 사용하라. 단, 실제 데이터나 실행 환경을 확인하지 않은 결과를 만들어 내거나 실험을 수행한 것처럼 보고하지 마라.

---

# 1. 데이터셋과 라벨 정의

기본 입력 데이터는 프로젝트의 `data/train.csv`다. 실제 파일 위치와 프로젝트 구조를 먼저 확인하고, 경로가 다르면 확인된 실제 경로를 사용하라.

예상 데이터 구조는 다음과 같다.

- `SMILES`: 분자 구조를 나타내는 문자열
- `ecfp_0` ~ `ecfp_1023`: ECFP 계열 fingerprint
- `fcfp_0` ~ `fcfp_1023`: FCFP 계열 fingerprint
- `ptfp_0` ~ `ptfp_1023`: Pattern fingerprint
- `MolWt`, `clogp`, `sa_score`, `qed`: 기존 분자 descriptor
- `label`: 이진 분류 타깃

데이터의 실제 컬럼, 자료형, 행 수, 결측치, 중복, fingerprint 값의 분포와 기존 descriptor의 계산 방식을 코드로 검증하라. 위의 컬럼 목록은 실제 파일을 확인하기 위한 사전 정보이지, 파일을 검증하지 않고 그대로 가정하라는 의미가 아니다.

## 고정된 라벨 의미

- `label = 0`: hERG 차단 활성, 즉 이 프로젝트에서 독성으로 정의한 클래스
- `label = 1`: hERG 차단 활성이 낮은 클래스, 즉 이 프로젝트에서 비독성으로 정의한 클래스

이 정의를 코드, 평가 지표, 시각화, 보고서 전체에서 일관되게 유지하라. 라벨 의미를 다시 질문하거나 반대로 해석하지 마라.

예측 종말점은 hERG 차단 활성이다. 임상적 QT 연장, 부정맥 또는 일반적인 임상 독성을 직접 예측한다고 표현하지 마라.

라벨이 대략적인 `IC50 < 10 µM` 기준으로 생성되었을 가능성을 고려하되, 모든 라벨이 동일한 assay에서 동일한 규칙으로 생성되었다고 단정하지 마라.

---

# 2. 실험 스캐폴드의 설계 원칙

다음 원칙을 반드시 지켜라.

1. 기존 fingerprint와 descriptor를 이용한 기준선부터 구축한다.
2. descriptor는 도메인 지식에 따라 그룹화하고, 그룹별 추가 효과를 독립적으로 측정한다.
3. 모든 피처 계산은 SMILES에서 재현 가능하도록 구현한다.
4. 모든 전처리, 결측치 처리, 피처 선택 및 하이퍼파라미터 최적화는 학습 데이터 내부에서만 적합한다.
5. 동일하거나 매우 유사한 분자 구조가 학습 세트와 평가 세트에 분산되어 발생하는 성능 과대평가를 방지한다.
6. 무작위 분할 결과는 참고용으로만 사용하고, 최종 모델 선정은 scaffold 기반 일반화 성능을 중심으로 수행한다.
7. 모델의 복잡도, 추론 비용, 성능 변동성도 함께 기록한다.
8. 모든 실험은 설정과 random seed를 기록하고 재실행할 수 있어야 한다.

최종 목표는 검증 데이터에서 우수한 성능을 보이면서, 학습에 등장하지 않은 Bemis–Murcko scaffold에서도 합리적인 성능을 유지하는 모델을 찾는 것이다.

---

# 3. 분자 descriptor 설계

RDKit을 기본 descriptor 엔진으로 사용하고, Mordred는 추가적인 분자 표현을 탐색하는 선택적 확장으로 구현하라.

모든 descriptor는 각 화합물의 SMILES로부터 계산한다. 기존 `MolWt`, `clogp`, `sa_score`, `qed`와 동일한 개념의 descriptor가 중복될 수 있으므로, 중복 여부와 계산 정의를 확인하라.

## 3.1 기본 물리화학적 특성

다음 항목을 우선 검토하라.

- 분자량: `MolWt`, `ExactMolWt`
- 지용성: `MolLogP`
- 극성: `TPSA`
- 수소 결합: `NumHDonors`, `NumHAcceptors`
- 구조 크기: `HeavyAtomCount`
- 결합 회전성: `NumRotatableBonds`
- 탄소 혼성화와 입체적 특성: `FractionCSP3`
- 고리 구조: `RingCount`, `NumAromaticRings`, `NumAliphaticRings`
- 원자 및 결합 구성: 탄소·질소·산소·황·할로젠 원자 수
- 분자 부피 및 표면 특성을 반영하는 RDKit descriptor
- `MolMR`, `LabuteASA` 등 분자 크기·표면 관련 지표

이미 보유한 descriptor와 중복되는 항목은 무조건 추가하지 말고, 기존 계산값과의 일치 여부를 검증한 뒤 실험 설계에 반영하라.

## 3.2 hERG 도메인 기반 descriptor

다음 그룹은 논문에서 보고된 경향을 검증하기 위한 후보군으로 설계하라.

### A. 지용성 및 소수성

- `MolLogP`
- RDKit의 `SlogP_VSA` 계열
- 소수성 표면과 관련된 분자 표면 분포 지표

검증 가설: 높은 지용성이 hERG 차단 활성과 연관되는가?

### B. 극성 및 수소 결합 특성

- `TPSA`
- HBD 및 HBA
- `PEOE_VSA` 계열
- 극성 표면과 관련된 descriptor

검증 가설: 상대적으로 낮은 극성 또는 특정 극성 표면 분포가 hERG 차단 활성 예측에 기여하는가?

### C. 염기성 아민 및 질소 원자 환경

- 총 질소 원자 수
- 아민 계열 SMARTS 패턴에 기반한 작용기 개수
- 1차·2차·3차 아민 관련 지표
- 고리형 아민과 방향족 질소의 구분
- 양전하 및 음전하 원자 수
- 분자 전체의 형식 전하
- 가능한 경우 pKa 예측값

중요: 단순히 질소 원자가 있다는 이유만으로 염기성 아민으로 분류하지 마라. 아마이드, 설폰아마이드, 피롤형 질소, 피리딘형 질소 등은 구분하라. SMARTS 패턴은 화학적 정의를 명시하고, 중첩 매칭으로 개수가 과대 계산되는 문제를 확인하라.

pKa 예측은 신뢰할 수 있는 라이브러리가 실제 환경에서 설치되고 재현 가능한 경우에만 사용하라. pKa를 임의의 SMARTS 규칙만으로 정확한 수치처럼 생성하지 마라.

검증 가설: 특정 염기성 아민의 존재와 환경이 hERG 차단 활성 예측에 추가 정보를 제공하는가?

### D. 방향족성, 강직성 및 분자 형상

- 방향족 고리 수
- 융합 고리 및 전체 고리 수
- 회전 가능한 결합 수
- `Kappa` 및 `Chi` 계열
- `BalabanJ`
- `BertzCT`
- 분자 표면 및 형상 관련 descriptor

검증 가설: 방향족성, 분자 강직성 및 구조 복잡도가 fingerprint만으로 포착하기 어려운 추가 정보를 제공하는가?

### E. 전하 및 원자 환경

- 형식 전하
- 부분 전하 분포
- 원자 환경 및 표면 전하 관련 descriptor
- `PEOE_VSA` 계열

전하 descriptor의 계산 정의와 RDKit 버전을 기록하라.

## 3.3 RDKit descriptor 전체 세트

RDKit에서 제공하는 적절한 2D descriptor를 가능한 범위에서 계산하라. 단, 구조 계산 실패, 상수 피처, 비정상값, 결측치가 많은 피처는 별도로 기록한다.

전체 descriptor를 추가한 실험과 도메인 기반 부분집합을 추가한 실험을 구분하라.

## 3.4 Mordred descriptor

Mordred의 2D descriptor를 선택적 확장 실험으로 구현하라.

- 기본 설정은 3D 좌표에 의존하지 않는 2D descriptor를 대상으로 한다.
- 계산 성공률과 descriptor별 결측률을 기록한다.
- 무한대, NaN, 상수 피처, 중복 피처를 처리한다.
- 결측률 기준은 설정 파일에서 조정 가능하게 한다.
- 결측치 대체 및 스케일링은 학습 데이터에만 적합한다.
- 계산 비용이 크면 캐시를 사용한다.
- 설치 및 계산 실패 시 RDKit 실험을 계속 수행하고 실패 원인을 기록한다.

Mordred 전체 피처를 무조건 추가하는 실험과, 학습 데이터 내부에서 선별한 Mordred 피처를 추가하는 실험을 구분하라.

## 3.5 피처 그룹 관리

각 descriptor는 명시적인 그룹으로 관리하라. 예를 들어 다음과 같은 구조를 사용한다.

- `physicochemical`
- `lipophilicity`
- `polarity_hbond`
- `basic_amine`
- `charge`
- `aromaticity_shape`
- `surface_electronic`
- `rdkit_all`
- `mordred_2d`

피처 계산 코드와 실험 설정을 분리해 그룹별 ablation 실험이 쉽게 가능하도록 설계하라.

---

# 4. 데이터 무결성과 피처 검증

모델 학습 전에 다음 검증을 구현하라.

1. SMILES 파싱 성공률
2. 분자별 descriptor 계산 성공률
3. 결측치·무한대·상수 피처
4. 동일한 descriptor의 중복 계산 여부
5. 높은 상관관계를 갖는 descriptor 쌍
6. 기존 fingerprint 컬럼과 RDKit에서 재계산한 fingerprint의 일치율
7. 라벨 분포 및 분자 구조 분포
8. 동일한 Bemis–Murcko scaffold에 속하는 화합물의 분포
9. 분자 표준화 전후의 구조 중복 가능성
10. 염, 혼합물, 입체화학 및 tautomer 처리 현황

기존 fingerprint를 무조건 재생성하여 대체하지 마라. 먼저 기존 피처의 생성 방식과 재현성을 확인하고, 불일치 사례를 보고하라.

염 제거, 중성화, tautomer 정규화는 구조와 라벨의 의미를 바꿀 수 있으므로 기본 학습 경로에서 자동으로 적용하지 마라. 필요하다면 별도의 민감도 분석으로 비교하라.

중복 제거 시 서로 다른 라벨을 가진 동일 구조가 발견되면 임의로 하나를 삭제하지 말고, 충돌 목록을 생성하고 처리 정책을 명시하라.

---

# 5. 데이터 분할과 누수 방지

## 5.1 기본 분할

Bemis–Murcko scaffold를 계산하고, 동일 scaffold가 서로 다른 데이터 분할에 들어가지 않도록 한다.

초기 목표 비율은 다음과 같다.

- 학습 세트: 약 80%
- 검증 세트: 약 10%
- 최종 테스트 세트: 약 10%

scaffold 크기와 클래스 비율 때문에 정확한 비율을 맞추기 어려울 수 있다. 이 경우 비율을 억지로 맞추지 말고 실제 분할 비율과 클래스 분포를 보고하라.

사용 가능한 경우 scaffold 단위 그룹 분할과 클래스 분포의 균형을 함께 고려한다. 분할 알고리즘과 random seed를 기록하고, 각 세트의 scaffold 중복 여부를 자동 검사하라.

scaffold가 없거나 분할이 불안정한 경우에는 적절한 대체 그룹 분할을 사용하되, 한계와 선택 이유를 설명하라.

## 5.2 추가 검증

다음 두 결과를 분리해서 보고하라.

- Random split: 기존 연구와 비교하기 위한 참고 실험
- Scaffold split: 새로운 화학 구조에 대한 일반화 성능을 평가하는 핵심 실험

가능하다면 여러 scaffold split seed를 반복하거나 scaffold 기반 교차검증을 추가하여 성능 변동성을 측정하라.

최종 테스트 세트는 모델이나 피처 선택에 사용하지 마라. 테스트 결과를 확인한 뒤 반복적으로 설정을 변경하는 방식도 금지한다.

---

# 6. 기준선과 비교 실험

각 실험은 같은 데이터 분할을 사용해 공정하게 비교하라.

다음 실험을 최소한 구현하라.

| ID | 입력 피처 | 목적 |
|---|---|---|
| E0 | 기존 `ecfp` + 기존 descriptor | 기준선 |
| E1 | 기존 `fcfp` + 기존 descriptor | fingerprint 계열 비교 |
| E2 | 기존 `ptfp` + 기존 descriptor | fingerprint 계열 비교 |
| E3 | 기존 fingerprint 중 E0~E2에서 검증된 최적 구성 | fingerprint 기준 모델 |
| E4 | RDKit descriptor만 사용 | descriptor 단독 정보량 평가 |
| E5 | 기존 최적 fingerprint + 기본 물리화학 descriptor | 일반적인 물성 추가 효과 |
| E6 | 기존 최적 fingerprint + hERG 도메인 descriptor | 도메인 지식의 기여 평가 |
| E7 | 기존 최적 fingerprint + RDKit 전체 descriptor | 전체 descriptor 확장 효과 |
| E8 | 기존 최적 fingerprint + Mordred 2D descriptor | Mordred 확장 효과 |
| E9 | 기존 최적 fingerprint + RDKit + Mordred | 결합 표현의 효과 |
| E10 | 기존 최적 fingerprint + 선택된 descriptor 그룹 | 피처 선택의 효과 |
| E11 | E10의 입력 피처 + 다른 모델 계열 | 모델 구조와 피처 효과 분리 |

가능하다면 ECFP, FCFP, PTFP를 각각 단독으로 사용하는 실험과 이들을 결합한 실험도 비교하라.

기존 CSV의 `ecfp`, `fcfp`, `ptfp`가 실제로 서로 다른 정보를 제공하는지 실험으로 확인하라. 단순히 피처 수가 많다는 이유로 결합 모델을 우선하지 마라.

도메인 descriptor의 기여를 주장하려면 반드시 동일한 분할, 동일한 평가 기준, 동일한 최적화 예산에서 해당 descriptor가 없는 모델과 비교하라.

---

# 7. 머신러닝 모델과 최적화

데이터 규모와 피처 특성에 맞는 모델을 비교하라.

최소 모델 후보:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting 계열 모델
- XGBoost 또는 LightGBM: 실제 설치 가능 여부를 확인한 경우
- SVM: 피처 규모와 계산 비용이 허용되는 경우

모든 모델을 무조건 실행할 필요는 없다. 계산 비용이 과도하거나 환경에서 지원되지 않는 모델은 제외 사유를 기록하라.

모델별로 적절한 하이퍼파라미터 탐색 공간을 정의한다. 가능하면 Optuna 또는 재현 가능한 동등한 최적화 방법을 사용한다.

- 최적화는 학습 세트 내부의 교차검증 또는 학습/검증 세트에서 수행한다.
- 피처 선택과 하이퍼파라미터 최적화를 테스트 세트에 맞추지 않는다.
- 모델별 탐색 횟수와 시간 예산을 기록한다.
- 최적화 결과뿐 아니라 기본 설정 대비 개선량을 기록한다.
- 모든 모델에 동일한 탐색 횟수를 강제하기보다는 계산량과 탐색 복잡도를 고려하되, 비교의 공정성을 설명한다.

트리 기반 모델에서는 불필요한 스케일링을 강제하지 마라. 선형 모델과 SVM에서는 적절한 스케일링을 적용하라.

피처 선택은 상수·중복 피처 제거, 상관관계 기반 제거, 모델 기반 중요도, permutation importance 등을 비교할 수 있다. 데이터 전체에서 타깃을 이용해 사전 선별한 피처를 테스트 평가에 사용하는 누수를 금지한다.

---

# 8. 평가 지표와 양성 클래스

이 프로젝트에서 관심 있는 양성 클래스는 hERG blocker인 `label=0`이다.

scikit-learn에 라벨을 그대로 전달할 경우 기본 양성 클래스가 `1`이 되어 해석이 잘못될 수 있으므로, 평가 전 다음처럼 명시적으로 변환하라.

`y_blocker = (label == 0).astype(int)`

모델 내부에서 원래 라벨을 학습할 경우에도 예측 확률과 지표 계산 시 클래스 순서를 확인하라.

필수 평가 지표:

- ROC-AUC
- PR-AUC (양성 클래스 = hERG blocker)
- Balanced Accuracy
- Precision
- Recall
- F1-score
- MCC
- Confusion Matrix

Accuracy는 보조 지표로 보고한다.

모델의 분류 임계값은 검증 데이터에서 선택할 수 있지만, 테스트 데이터의 성능을 본 뒤 임계값을 변경하지 마라. 기본 임계값 결과와 검증 세트에서 선택한 임계값 결과를 구분하라.

실제 연구 목적에 따라 민감도와 정밀도의 균형이 중요할 수 있으므로, 단일 지표만으로 모델의 우수성을 주장하지 마라.

최종 결과에는 주요 지표의 반복 실험 평균과 표준편차 또는 신뢰구간을 포함하라. 가능한 경우 scaffold 단위 부트스트랩을 사용하되, 표본 독립성 가정을 명시하라.

---

# 9. 도메인 지식의 실제 기여 분석

단순히 feature importance 상위에 특정 descriptor가 등장했다는 이유만으로 해당 화학적 특성이 hERG 차단의 원인이라고 결론 내리지 마라.

다음 분석을 수행하라.

1. 기존 fingerprint 기준선 대비 각 descriptor 그룹의 성능 변화
2. descriptor 그룹별 ablation 결과
3. 모델별 permutation importance
4. 상관된 descriptor의 중복 기여 여부
5. 주요 descriptor와 hERG blocker 라벨의 분포 비교
6. 오분류된 화합물의 구조 및 물성 분석
7. scaffold별 성능 차이
8. 가능하면 SHAP 등 해석 도구를 이용한 개별 예측 설명

각 결과는 상관관계와 예측 기여를 설명하는 수준으로 해석하라. 모델 중요도를 인과관계나 분자 수준의 확정적 작용 기전으로 표현하지 마라.

도메인 가설과 데이터 결과가 일치하지 않는 경우에도 그 결과를 숨기지 말고 보고하라.

---

# 10. 클래스 불균형과 데이터 품질

클래스 분포를 먼저 확인하고, 불균형이 성능에 미치는 영향을 평가하라.

필요하면 다음을 비교하라.

- 기본 학습
- class weighting
- 적절한 임계값 조정

SMOTE 등 합성 오버샘플링을 사용할 경우, 반드시 교차검증 내부의 학습 폴드에서만 적용하라. 분할 전에 오버샘플링하여 검증·테스트 데이터와 정보를 공유하지 마라.

분자 fingerprint는 고차원 희소 표현이므로, 일반적인 연속형 피처에 적용하는 오버샘플링이 적절한지 검토하라. 근거 없이 SMOTE를 기본값으로 사용하지 마라.

---

# 11. 프로젝트 구조

기존 프로젝트 구조를 먼저 확인한 뒤, 기존 코드를 무조건 덮어쓰지 말고 재사용 가능한 부분을 보존하라.

가능하면 다음과 같은 구조로 구성하라.

```text
project/
├── data/
│   ├── train.csv
│   └── processed/
├── configs/
│   └── experiments.yaml
├── src/
│   └── herg_model/
│       ├── data.py
│       ├── chemistry.py
│       ├── descriptors.py
│       ├── fingerprints.py
│       ├── splitting.py
│       ├── preprocessing.py
│       ├── feature_selection.py
│       ├── models.py
│       ├── metrics.py
│       ├── experiments.py
│       ├── explainability.py
│       └── predict.py
├── scripts/
│   ├── inspect_data.py
│   ├── calculate_descriptors.py
│   ├── run_experiments.py
│   ├── analyze_results.py
│   └── train_final_model.py
├── tests/
├── artifacts/
├── reports/
├── pyproject.toml
├── README.md
└── .gitignore
```

이는 권장 구조다. 실제 프로젝트의 기존 구조, 패키지 관리자 및 운영체제에 맞게 조정하라.

`uv`를 사용할 수 있다면 의존성을 `pyproject.toml`에 관리하고 lockfile을 생성하라. RDKit 및 Mordred의 버전 호환성을 확인하고, 실제 설치에 성공한 환경을 기록하라.

분자 descriptor 계산 결과는 SMILES와 구조 키를 이용해 캐시할 수 있다. 캐시를 재사용할 때도 SMILES, descriptor 설정, 라이브러리 버전이 일치하는지 검증하라.

원본 데이터는 수정하지 말고, 파생 데이터와 실험 산출물을 분리하라.

---

# 12. 실행 인터페이스와 산출물

설정 파일에서 다음을 조정할 수 있도록 구현하라.

- 입력 데이터 경로
- 라벨 매핑
- fingerprint 종류
- descriptor 그룹
- 분할 방식과 random seed
- 모델 목록
- 하이퍼파라미터 탐색 예산
- 평가 지표
- 출력 경로
- 계산 병렬도 및 캐시 사용 여부

최소한 다음 명령으로 데이터 검사, descriptor 생성, 실험 실행, 결과 분석, 최종 모델 학습을 각각 수행할 수 있어야 한다. 실제 CLI 구현에 맞게 명령을 작성하라.

```bash
python scripts/inspect_data.py
python scripts/calculate_descriptors.py
python scripts/run_experiments.py
python scripts/analyze_results.py
python scripts/train_final_model.py
```

필수 산출물:

1. `reports/data_quality_report.md`: 데이터 품질과 구조 검증
2. `reports/descriptor_report.md`: descriptor 정의, 계산 성공률, 결측치 및 중복 분석
3. `reports/experiment_results.csv`: 모든 실험의 설정과 평가 지표
4. `reports/ablation_results.csv`: descriptor 그룹별 추가 효과
5. `reports/error_analysis.csv`: 오분류 사례와 관련 구조 정보
6. `reports/feature_importance.csv`: 주요 피처 중요도
7. `reports/final_report.md`: 실험 방법, 핵심 결과, 한계 및 권고
8. `artifacts/best_model.joblib`: 최종 학습된 모델 또는 전체 추론 파이프라인
9. `artifacts/model_metadata.json`: 라벨 의미, 피처 순서, descriptor 설정, 라이브러리 버전, 학습 데이터 정보, 분할 설정 및 선택된 임계값
10. `artifacts/experiment_manifest.json`: 실험별 설정, random seed, 실행 상태 및 결과 파일 참조

시각화도 생성하라.

- 모델별 ROC 및 PR 곡선
- 실험별 핵심 지표 비교
- descriptor 그룹별 ablation 결과
- 혼동행렬
- 주요 피처 중요도
- scaffold별 성능 분포

모든 그림과 표에는 양성 클래스가 `label=0`인 hERG blocker임을 명확히 표시하라.

최종 추론 코드는 학습에 사용한 피처 생성 순서와 전처리를 동일하게 적용해야 한다. 새로운 SMILES 입력에 대해 예측 클래스, blocker 확률 및 원래 라벨 체계를 반환하도록 구현하라.

확률은 해당 모델이 산출한 예측값이며, 별도의 보정과 검증 없이 실제 임상적 독성 확률로 해석하지 마라.

---

# 13. 구현 및 검증 순서

다음 순서를 지켜 작업하라.

**단계 A — 프로젝트 및 데이터 확인**

- 파일과 기존 코드를 확인한다.
- 라벨 분포, 피처 컬럼, 구조 파싱 및 fingerprint 일치율을 검사한다.
- 기존 기준선 구현에 필요한 입력을 확정한다.

**단계 B — 기준선 구현**

- 기존 fingerprint 기반 모델을 구현한다.
- scaffold split을 생성하고 누수 검사를 수행한다.
- 기준선 성능과 실행 시간을 기록한다.

**단계 C — descriptor 파이프라인 구현**

- RDKit 기본 descriptor와 도메인 그룹을 구현한다.
- Mordred 2D descriptor를 추가한다.
- 계산 결과와 실패 사례를 캐시하고 보고한다.

**단계 D — 통제된 비교 실험**

- 실험 E0~E11을 실행한다.
- 동일한 분할에서 피처 그룹과 모델의 영향을 비교한다.
- 탐색 예산과 실패한 실험을 기록한다.

**단계 E — 최적화 및 오류 분석**

- 학습 데이터 내부에서 하이퍼파라미터를 최적화한다.
- descriptor 그룹별 기여와 scaffold별 오류를 분석한다.
- 성능이 낮은 실험도 결과에 포함한다.

**단계 F — 최종 모델과 재현성 검증**

- 검증 데이터 기준으로 모델과 임계값을 선택한다.
- 선택이 완료된 후 최종 테스트 세트를 평가한다.
- 최종 파이프라인을 저장하고, 새로운 SMILES에 대한 추론 테스트를 수행한다.
- 설치 및 실행 방법을 README에 기록한다.

각 단계는 실제로 실행 가능한 상태인지 확인하고 진행하라. 전체 실험을 수행할 계산 자원이 부족하다면, 우선순위가 높은 실험부터 실행하고 미실행 항목을 명확히 표시하라.

---

# 14. 최종 완료 기준

작업을 완료했다고 보고하기 전에 다음을 확인하라.

- [ ] `label=0`을 hERG blocker로 일관되게 처리했다.
- [ ] 기존 fingerprint 기반 기준선이 실행된다.
- [ ] descriptor 계산 성공률과 결측치 처리가 검증되었다.
- [ ] 도메인 기반 descriptor와 일반 descriptor의 기여를 분리해 비교했다.
- [ ] 동일 scaffold가 학습·검증·테스트에 걸쳐 중복되지 않음을 검사했다.
- [ ] 전처리 및 피처 선택 과정에서 테스트 누수가 없다.
- [ ] 모든 모델의 결과와 실패 사례를 기록했다.
- [ ] scaffold 기반 성능을 기준으로 모델을 선택했다.
- [ ] 최종 모델과 메타데이터를 저장했다.
- [ ] 저장된 모델로 새로운 SMILES 추론을 실행했다.
- [ ] 실행 명령과 의존성 설치 방법을 문서화했다.
- [ ] 보고서의 수치가 실제 실행 결과와 일치한다.

최종 보고에서는 다음 질문에 직접 답하라.

1. 어떤 fingerprint 구성이 가장 좋은 기준선인가?
2. RDKit descriptor가 기준선 대비 성능을 개선했는가?
3. hERG 도메인 기반 descriptor가 일반 물성 descriptor보다 추가적인 이점을 제공했는가?
4. Mordred descriptor는 추가적인 이점을 제공했는가, 아니면 노이즈와 계산 비용만 증가시켰는가?
5. 어떤 모델과 피처 조합이 scaffold 기반 평가에서 가장 우수했는가?
6. 성능 개선은 여러 scaffold 분할에서도 안정적으로 나타났는가?
7. 가장 중요한 피처와 오분류 패턴은 무엇인가?
8. 최종 모델의 한계와 실제 활용 시 주의점은 무엇인가?

**최종 원칙:** 가장 높은 숫자를 얻는 것보다, 성능 향상의 원인을 설명할 수 있고 새로운 분자 구조에 대한 일반화 가능성을 검증한 실험 결과를 얻는 것이 우선이다. 결과가 개선되지 않았다면 그 사실을 그대로 보고하고, descriptor 추가가 효과적이지 않았다는 결론도 허용하라.