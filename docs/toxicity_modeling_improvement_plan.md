# Toxicity 분류 성능 개선 제안

> `toxicity_eda.ipynb`, `train.csv` (8,349행 × 3,078열) 기반 분자 독성 이진 분류 모델의 성능 개선 로드맵입니다.  
> 라벨 정의: **1 = 독성 없음(non-toxic, Positive)**, **0 = 독성 있음(toxic, Negative)** (PDF 기준)

---

## 1. 현재 상태

### 데이터 구조

- 8,349행 × 3,078열
- `SMILES` 1개
- ECFP 1,024 bits + FCFP 1,024 bits + PTFP 1,024 bits
- 기술자 4개: `MolWt`, `clogp`, `sa_score`, `qed`
- `label`: 1 = non-toxic (4,542), 0 = toxic (3,807)
- PTFP 상수 컬럼 49개
- 같은 2D fingerprint를 가진 stereoisomer 그룹 242개 / 536행
- 그중 라벨이 0/1로 혼재한 그룹 34개 / 75행

### 현재 베이스라인

Group CV 5-fold, LogisticRegression 기준:

| 모델 | ROC-AUC | PR-AUC | F1 | Toxic recall |
|---|---:|---:|---:|---:|
| Descriptors 4개 | 0.668 | 0.709 | 0.673 | 0.499 |
| ECFP | 0.790 | 0.807 | 0.739 | 0.676 |
| All fingerprints | 0.832 | 0.843 | 0.778 | 0.729 |
| **Descriptors + all fingerprints** | **0.836** | **0.846** | **0.781** | **0.733** |

현재 구조는 전형적인 **QSAR/tabular classification**입니다.  
3,072개의 sparse binary feature와 8,349행 규모에서는 복잡한 딥러닝보다 **LightGBM/XGBoost + feature engineering + Group CV**가 가장 현실적인 개선 축입니다.

---

## 2. 개선 우선순위

| 우선순위 | 방법 | 비용 | 기대 효과 |
|---|---|---|---|
| 1 | 혼재 라벨 75행 감사 + threshold tuning | 낮음 | F1/toxic recall 개선 가능 |
| 2 | LightGBM/XGBoost + feature selection | 중간 | 가장 유력한 성능 개선 |
| 3 | RDKit descriptor + structural alert 추가 | 낮음~중간 | 중간 |
| 4 | 유사도/kNN 파생 변수 | 중간 | 중간 |
| 5 | Ensemble + calibration | 낮음~중간 | 소폭~중간 |
| 6 | GNN, SMILES transformer, 3D fingerprint | 높음 | 불확실 |

---

## 3. 파생 변수

### 3-1. Fingerprint 요약 변수

현재 fingerprint는 binary bit만 있습니다. 여기서 다음을 파생할 수 있습니다.

- fingerprint별 active bit 수: `ecfp_n_bits`, `fcfp_n_bits`, `ptfp_n_bits`
- 밀도: `n_bits / 1024`
- fingerprint 간 Jaccard 유사도:
  - `ECFP–FCFP`
  - `ECFP–PTFP`
  - `FCFP–PTFP`
- 세 fingerprint의 consensus bit 수
- count fingerprint로 재계산: 같은 부분구조가 여러 번 나오는 경우 count 정보가 추가됨
- multi-radius fingerprint: ECFP2/ECFP4/ECFP6를 각각 계산해 concatenate

> 현재 데이터의 ECFP/FCFP/PTFP는 생성 설정(radius 등)이 metadata에 없습니다.  
> 재계산 실험은 **새 baseline**으로 다뤄야 하며, 기존 fingerprint와 동일 조건이라고 가정하면 안 됩니다.

### 3-2. RDKit 화학 descriptor

현재 4개 descriptor만으로는 부족할 수 있습니다. RDKit으로 다음을 추가할 수 있습니다.

| 범주 | 예시 |
|---|---|
| 극성/수소결합 | TPSA, HBD, HBA |
| 유연성/고리 | Rotatable bonds, ring count, aromatic ring count |
| 조성 | Heavy atom count, heteroatom count, halogen count |
| 3차원 성격 | Fraction Csp3 |
| 전하 | Formal charge, max/min partial charge |
| 복잡도 | Amide bond, spiro atom, bridgehead atom |
| 입체화학 | Stereocenter 수, unspecified stereocenter 수 |
| 물성 | Molecular refractivity, Labute ASA |
| topology | Balaban J, Bertz CT, Chi/Kappa shape indices |

비율·상호작용 변수도 유용합니다.

- `TPSA / MolWt`
- `HBD / (HBA + 1)`
- `clogp - qed`
- `MolWt * clogp`
- `aromatic_rings / ring_count`

트리 모델은 상호작용을 스스로 학습하지만, 선형 모델에서는 이런 비율 변수가 도움이 됩니다.

### 3-3. Structural alert / functional group

독성 예측에서는 구조 경고(alert)가 domain knowledge를 제공합니다.

- PAINS, Brenk, NIH filter catalog
- Michael acceptor, epoxide, nitroaromatic, azo, aniline, halogenated aromatic
- reactive metabolite 관련 substructure
- SMARTS pattern match count/flag

RDKit의 `FilterCatalog`를 사용하면 PAINS/Brenk 등을 flag로 만들 수 있습니다.  
단, alert가 label과 직접 겹칠 수 있으므로 **Group CV에서 실제 기여도**를 확인해야 합니다.

### 3-4. 유사도/kNN 파생 변수

QSAR에서 매우 유용한 방식입니다.

- 각 분자의 **train set 내 최대 Tanimoto 유사도**:
  - `max_sim_to_toxic` (label 0)
  - `max_sim_to_non_toxic` (label 1)
  - `diff = max_sim_to_non_toxic - max_sim_to_toxic`
- top-k 이웃의 평균 유사도
- kNN 투표 비율
- nearest neighbor의 label

중요: 이 feature는 **반드시 CV fold 안에서 train set만 사용해 계산**해야 합니다.  
전체 데이터로 미리 계산하면 leakage입니다.

### 3-5. Scaffold / ring 시스템

- Murcko scaffold
- scaffold 빈도
- ring system 수, ring size 분포
- spiro/bridgehead 수
- scaffold별 out-of-fold target encoding (smoothing 적용)

Scaffold target encoding은 강력할 수 있지만 **leakage 위험이 큽니다.**  
Group CV와 함께 쓰려면 out-of-fold 방식으로만 계산해야 합니다.

### 3-6. SMILES 기반 저비용 변수

- SMILES 길이
- `@`, `/`, `\` 개수
- 괄호/대괄호/숫자(ring closure) 개수
- 원자 token 수
- charge 표기 수

화학적으로는 약하지만, 빠르게 만들 수 있고 오류 분석에 유용합니다.

### 3-7. 3D / conformer 변수

- ETKDG conformer 생성
- USR, USRCAT, WHIM
- 3D pharmacophore fingerprint

3D shape/conformation이 독성에 중요할 수 있지만 계산 비용이 크고, 8,349건에서는 효과가 불확실합니다. **후순위**로 두는 것이 좋습니다.

---

## 4. 이 데이터 구조에 어울리는 모델

### 1순위: LightGBM / XGBoost

Sparse binary fingerprint와 descriptor 조합에 잘 맞습니다.

- 비선형 상호작용 학습
- 3,000개 feature에서도 비교적 빠름
- `feature_fraction`, `bagging`, `min_child_samples`로 과적합 제어
- `class_weight='balanced'` 또는 `scale_pos_weight` 선택

```python
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
)
```

### 2순위: LogisticRegression + L1/ElasticNet

현재 baseline(0.836)과 비교할 강한 선형 모델입니다.

- high-dimensional sparse fingerprint에 적합
- L1은 feature selection 효과
- ElasticNet은 correlated bit를 다루기 좋음
- 계수 해석이 가능

### 3순위: BernoulliNB

Binary fingerprint와 궁합이 좋은 단순 베이스라인입니다.

- 매우 빠름
- sparse binary feature에 자연스러운 모델
- Logistic/LightGBM과 앙상블하면 diversity 확보

### 4순위: RandomForest / ExtraTrees

- robust하고 튜닝이 쉬움
- fingerprint의 일부 feature만 사용하는 `max_features` 설정이 중요
- LightGBM보다 성능이 낮을 수 있지만 앙상블 재료로 유용

### 5순위: kNN + Tanimoto/Jaccard

- 화학적 유사도 기반
- 해석이 직관적
- 단독 성능은 낮을 수 있지만 **similarity feature**나 ensemble로 가치가 있음

### 6순위: SVM (RBF)

- SVD/PCA로 차원을 줄인 뒤 적용
- 소규모 QSAR에서 강력할 수 있음
- 3,000차원 raw sparse에는 비효율적

### 7순위: GNN / SMILES transformer

- MPNN/GIN, ChemBERTa/MolFormer
- 분자 그래프와 입체화학을 직접 다룰 수 있음
- 다만 8,349건은 딥러닝에는 작은 편
- scaffold split에서 성능이 크게 흔들릴 수 있음
- **LightGBM을 이긴다는 보장이 없으므로 후순위**

### 앙상블

서로 다른 inductive bias를 가진 모델을 섞는 것이 좋습니다.

- LogisticRegression + LightGBM + BernoulliNB
- 또는 LightGBM + ExtraTrees + kNN
- Soft voting, rank averaging, out-of-fold stacking
- 최종적으로 **calibration** 적용

---

## 5. Feature selection

3,072개 fingerprint bit는 대부분 희귀하거나 중복 정보입니다.

- 상수 bit 제거
- prevalence < 0.1~1%인 rare bit 제거 (선형 모델에서 특히)
- `VarianceThreshold`
- `SelectKBest(chi2 또는 mutual_info_classif)`
- L1 regularization
- LightGBM `feature_importance` 기반 selection
- fingerprint group별로 묶어 permutation importance 평가

주의: feature selection도 **CV 안에서** 수행해야 합니다.  
전체 데이터에서 먼저 selection하면 leakage입니다.

---

## 6. 평가·학습 전략

### Group CV 유지

현재처럼 `fp_group` 기준 `StratifiedGroupKFold`를 사용합니다.

- 같은 2D 골격의 입체이성질체가 train/valid에 동시에 들어가는 것을 방지
- 가능하면 RDKit Murcko scaffold 기준 split도 비교
- random split 결과만 보고 과대평가하지 않기

### 반복 CV

- 5-fold × 여러 seed (예: 5×3)
- 평균 ± 표준편차로 모델 비교
- 단일 fold 결과로 결론 내리지 않기

### 지표

- ROC-AUC, PR-AUC
- F1, Precision, Recall
- **독성(label=0) recall**과 specificity
- calibration: Brier score, reliability curve
- threshold별 confusion matrix

### Threshold tuning

AUC는 threshold와 무관하지만, 실제 분류 성능은 threshold에 크게 좌우됩니다.

- 기본 0.5는 탐색용
- 안전 목적이면 **독성 recall(label 0)** 우선
- F1, F2, 비용 기반 threshold 비교
- label polarity(1=non-toxic) 주의

### Class imbalance

현재 54:46으로 경미합니다.

- `class_weight='balanced'`는 선택적으로 실험
- SMOTE는 binary fingerprint에서 비현실적인 분자를 만들 수 있으므로 권장하지 않음
- threshold tuning이 더 안전

---

## 7. 추천 실험 순서

1. **라벨 감사**
   - 혼재 라벨 75행의 원본 assay/source 확인
   - 라벨 오류라면 정리
2. **현재 baseline 재확인**
   - Logistic + all fingerprints + descriptors: Group CV 0.836
3. **RDKit descriptor 추가**
   - TPSA, HBD/HBA, rotatable bonds, ring count, stereocenter 등
   - Logistic/LightGBM 비교
4. **LightGBM/XGBoost**
   - raw fingerprint + descriptor
   - feature selection 전/후 비교
5. **Structural alert / similarity feature**
   - PAINS/Brenk, kNN Tanimoto feature
   - out-of-fold 계산 준수
6. **Feature selection + tuning**
   - rare bit 제거, L1, LightGBM importance
   - RandomizedSearch/Optuna + nested Group CV
7. **Ensemble + calibration**
   - Logistic + LightGBM + BernoulliNB
   - threshold tuning
8. **고급 모델**
   - GNN, SMILES transformer, 3D fingerprint는 마지막

---

## 8. 실험 매트릭스

| 실험 | Feature | 모델 | 비교 목적 |
|---|---|---|---|
| A | 현재 fingerprint + descriptor | Logistic | baseline |
| B | A + RDKit descriptor | Logistic | descriptor 추가 효과 |
| C | A + RDKit descriptor | LightGBM | 비선형 모델 효과 |
| D | C + feature selection | LightGBM | rare bit 제거 효과 |
| E | D + structural alert | LightGBM | domain feature 효과 |
| F | D + similarity feature | LightGBM | kNN/QSAR 효과 |
| G | E + F | LightGBM | 결합 효과 |
| H | G + ensemble | Logistic/LGBM/NB | 앙상블 효과 |
| I | H + calibration/threshold | 최종 | 실전 지표 개선 |

모든 실험은 **동일한 Group CV fold**와 **여러 seed**로 비교합니다.

---

## 9. Leakage / 주의사항

- **Random split 금지**: stereoisomer가 train/valid에 나뉘면 누수
- **Similarity feature**: fold 내 train set으로만 계산
- **Target encoding**: out-of-fold + smoothing
- **Feature selection**: fold 내에서만 수행
- **Hyperparameter tuning**: nested CV 권장
- **SMOTE**: binary fingerprint에서는 비현실적 분자 생성 가능
- **PTFP**: dense(42% nonzero)이므로 포함/제외 ablation 필요
- **상수 컬럼**: PTFP 49개 제거
- **라벨 극성**: 1=non-toxic, 0=toxic; toxic recall은 `pos_label=0`
- **Chiral feature**: 혼재 그룹 75행에 대해 제한적 효과. 전체 성능 개선 카드로 기대하지 말 것

---

## 10. 현실적인 기대

- 현재 ROC-AUC 0.836은 이미 강한 베이스라인입니다.
- **가장 큰 개선 가능성**은 label audit, threshold tuning, LightGBM/XGBoost, feature selection 조합입니다.
- Chiral fingerprint는 혼재 그룹이 75행뿐이라 전체 성능 개선 효과가 제한적이었습니다.
- GNN/transformer는 데이터가 8,349건이라 LightGBM을 안정적으로 이기기 어려울 수 있습니다.
- 따라서 **“LightGBM + RDKit descriptor + structural alert + similarity feature + Group CV + threshold tuning”**을 1차 목표로 잡는 것이 좋습니다.

---

## 11. 관련 문서 및 산출물

- `toxicity_concepts_qa.md` — fingerprint, QED, clogP, SA score 개념 Q&A
- `toxicity_stereochemistry_followup.md` — chiral fingerprint 실험, 혼재 라벨 분석
- `toxicity_chiral_embedding_geometry.md` — 임베딩 공간 관점의 chiral 분석, 비중복 입체 subset
- `toxicity_mordred_feature_proposal.md` — Mordred descriptor 전체 검토 및 추가 우선순위
- `mixed_label_groups.csv` — 혼재 라벨 75행 전체
- `mixed_label_group_summary.csv` — 혼재 라벨 34개 그룹 요약
- `toxicity_eda.ipynb` — 기본 EDA 및 베이스라인

### 다음 단계

1. `toxicity_modeling.ipynb` 새 파일 생성
2. 실험 A~D부터 순차 실행
3. 동일 Group CV fold로 성능 비교
4. 최종 모델은 calibration + threshold tuning 후 저장

---

## 12. 참고

- Bickerton et al. (2012). *Quantifying the chemical beauty of drugs.* Nature Chemistry, 4, 90–98.
- Rogers & Hahn (2010). *Extended-connectivity fingerprints.* J Chem Inf Model, 50, 742–754.
- Ertl & Schuffenhauer (2009). *Estimation of synthetic accessibility score.* J Cheminform, 1, 8.
- Wildman & Crippen (1999). *Prediction of physicochemical parameters by atomic contributions.* J Chem Inf Comput Sci, 39, 868–873.
- Lipinski et al. (1997). *Experimental and computational approaches to estimate solubility and permeability.* Adv Drug Deliv Rev, 23, 3–25.
- RDKit Documentation. https://www.rdkit.org/docs/
