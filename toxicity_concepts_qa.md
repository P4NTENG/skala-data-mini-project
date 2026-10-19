# 분자 독성 EDA 개념 Q&A

> `toxicity_eda.ipynb`와 `train.csv`(8,349행 × 3,078열) 분석 과정에서 나온 개념 질문을 정리한 문서입니다.  
> Python/ML 경험은 있지만 cheminformatics 용어가 익숙하지 않은 독자를 대상으로 합니다.

## 문서 개요

- **데이터**: `train.csv` — SMILES 1개, ECFP/FCFP/PTFP 각 1,024 bits, 기술자 4개(`MolWt`, `clogp`, `sa_score`, `qed`), `label`
- **타깃**: `label` — **1 = 독성 없음(non-toxic, Positive)**, **0 = 독성 있음(toxic, Negative)**  
  PDF 15페이지 기준입니다. 일반적인 “1=독성” 관례와 반대이므로 지표 해석 시 반드시 확인해야 합니다.
- **핵심 구분**: 구조 표현(fingerprint, SMILES), 물성(QED, clogP, SA score), 독성 label은 서로 다른 개념입니다.

---

## Q1. Fingerprint와 3D/입체화학

> “fingerprint는 2차원 프로젝션 구조를 벡터화한 것이므로 3차원 이성질체를 고려하지 않았지만, 이것은 SMILES로 구별할 수 있다?”

### 정정: fingerprint는 “2D 프로젝션”이 아니다

Fingerprint는 분자를 평면에 투영한 이미지를 벡터화한 것이 아니라, **분자 그래프의 원자 환경과 부분구조를 일정한 길이의 비트로 해시한 2D topological fingerprint**입니다.

- **ECFP**: 원자 주변 환경을 반복적으로 확장해 부분구조를 해시
- **FCFP**: pharmacophore(수소결합 주개/받개, 전하, 방향족 등) 중심
- **PTFP**: 부분구조 패턴 기반

### 3D conformation과 configuration은 다르다

| 구분 | 예 | SMILES 표현 | 기본 fingerprint |
|---|---|---|---|
| **Configuration**(입체화학) | R/S, E/Z, cis/trans | isomeric SMILES로 가능 | 구현 설정에 따라 가능 |
| **Conformation**(배좌) | 회전, 폴딩, 실제 3D 모양 | 불가능 | 불가능 |

- Fingerprint는 3D 좌표나 conformation을 담지 않습니다.
- 하지만 **configuration**은 fingerprint 구현에서 옵션으로 포함할 수 있습니다. RDKit의 Morgan/ECFP는 `useChirality=True`로 설정하면 chirality를 반영하며, 기본값은 `False`인 경우가 많습니다.
- **Isomeric SMILES**는 `@`, `@@`, `/`, `\` 표기로 R/S, E/Z를 구별합니다.
- **Non-isomeric SMILES**는 입체화학을 잃습니다.
- SMILES는 특정 3D conformation을 담지 않으며, 일부 복잡한 입체화학은 CXSMILES 같은 확장 표기가 필요할 수 있습니다.

### 이 데이터에서의 증거

- 전체 8,349행, SMILES 문자열 8,349개(문자열 중복 0)
- 정확히 같은 fingerprint 벡터를 가진 그룹: **242개 / 536행**
- 그중 label이 0과 1로 혼재한 그룹: **34개 / 75행**
- 중복 예시들은 `@`/`@@`만 다른 **입체이성질체**였습니다.

따라서 제공된 ECFP/FCFP/PTFP는 chirality를 포함하지 않는 설정으로 생성된 것으로 보입니다. **SMILES에는 구별 정보가 남아 있지만, fingerprint가 그 정보를 버린 경우**입니다.

```python
# RDKit에서 chirality를 포함해 fingerprint를 계산하는 예
from rdkit.Chem import rdMolDescriptors

fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,          # ECFP4
    nBits=1024,
    useChirality=True, # 입체화학 반영
)
```

---

## Q2. SMILES와 fingerprint의 역할 차이

> “SMILES만을 사용할 수 없는 이유는 고유값은 보장되지만 결합구조를 고려하지 않아서 정보가 부족하다. 그러므로 fingerprint를 사용해야 한다?”

### 정정 1: SMILES도 결합구조를 담고 있다

SMILES는 단순 문자열이 아니라 **분자 그래프의 직렬화 표현**입니다.

- 원자 종류
- 결합 차수(`-`, `=`, `#`, 방향족)
- 가지 구조(`()`)
- 고리(`1`, `2`, …)
- 전하, 동위원소
- 입체화학(`@`, `@@`, `/`, `\`)

즉, “SMILES는 결합구조를 고려하지 않는다”는 설명은 틀립니다. SMILES는 결합 연결성과 결합 차수를 명시적으로 담습니다.

### 정정 2: SMILES의 유일성은 보장되지 않는다

- **Canonical SMILES**는 특정 toolkit/버전 기준으로 유일하지만, 모든 SMILES 문자열이 canonical인 것은 아닙니다.
- 같은 분자가 tautomer, protonation state, salt form, stereoisomer에 따라 여러 문자열로 표현될 수 있습니다.
- 이 데이터의 “SMILES 문자열 중복 0”은 **문자열로서 중복이 없다**는 뜻이지, 화학적으로 완전히 유일하다는 보장은 아닙니다.
- Fingerprint도 **해시 충돌**과 chirality 손실 때문에 유일하지 않습니다.

### 정정 3: Fingerprint는 필수품이 아니다

Fingerprint는 여러 featurization 중 하나입니다. 대안으로는 다음이 있습니다.

- 기술자(descriptor) 조합
- MACCS keys
- Graph Neural Network(GNN)
- SMILES transformer, 문자 n-gram

이 프로젝트에서 fingerprint를 주력으로 사용한 이유는 **고정 길이 sparse vector**라 classical ML에 바로 넣기 좋고, 8,349건 규모에서 강한 베이스라인을 주기 때문입니다.

Group CV 기준 ROC-AUC:

| 모델 | ROC-AUC |
|---|---:|
| Descriptors 4개 | 0.668 |
| ECFP | 0.790 |
| All fingerprints | 0.832 |
| Descriptors + all fingerprints | **0.836** |

fingerprint도 정보를 압축하며 손실이 있으므로, “SMILES보다 항상 우월하다”고 말할 수는 없습니다. **표현 방식과 모델에 맞는 선택**이 중요합니다.

---

## Q3. QED의 의미

> “QED는 RDKit이라는 툴로 toxicity를 나이브하게 수치화한 것이다?”

### 정정: QED는 toxicity가 아니라 drug-likeness 점수다

- **QED** = **Quantitative Estimate of Drug-likeness**
- Bickerton et al. (2012)이 제안한 heuristic
- **0~1 범위**, 높을수록 약물처럼 유리한 물성
- 분자량, logP, 수소결합 주개/받개, PSA, 회전 가능 결합, 방향족 고리, structural alerts 등을 **desirability function**으로 결합한 weighted geometric mean
- **독성 데이터로 학습한 값이 아니며, toxicity predictor가 아닙니다.**

RDKit은 QED를 계산하는 대표적인 구현 중 하나일 뿐, QED의 정의 자체는 아닙니다. Structural alerts 항목이 있어 독성과 간접적으로 연관될 수는 있지만, QED가 높다고 “안전하다”고 판단할 수는 없습니다.

```python
from rdkit.Chem import QED
score = QED.qed(mol)   # 0 ~ 1
```

### 이 데이터에서의 QED

| 항목 | 값 |
|---|---:|
| 평균 | 0.559 |
| 범위 | 0.024 ~ 0.947 |
| label 1(독성 없음) 평균 | 0.573 |
| label 0(독성 있음) 평균 | 0.542 |
| label과의 상관 | **+0.084** (매우 약함) |

QED는 독성 label이 아니라 **보조 기술자**로만 해석해야 합니다.

---

## Q4. clogP / logP / logD의 의미

> “clogP, 분배계수는 무슨 의미를 갖는 거지?”

### logP

`logP`는 **옥탄올/물 분배계수(partition coefficient)** 의 로그값입니다.

\[
\log P = \log_{10} \frac{[\text{solute}]_{\text{octanol}}}{[\text{solute}]_{\text{water}}}
\]

- **높은 logP** = 지용성/친유성(lipophilic)이 강함 → 옥탄올 층에 더 많이 분포
- **낮은 logP** = 친수성(hydrophilic)이 강함 → 물 층에 더 많이 분포
- `clogP`는 이 logP를 **계산(calculated)** 한 값입니다. 대표적으로 Wildman–Crippen 방식이 있습니다.

### logD와의 차이

- **logP**: 중성(neutral) 형태의 분배계수
- **logD**: 특정 pH에서 이온화 형태까지 포함한 **분포계수(distribution coefficient)**
- 생리적 pH 7.4에서는 logD가 더 현실적인 지표인 경우가 많습니다.

### 약물 설계에서의 의미

- logP가 너무 높으면: 용해도 저하, 비특이적 결합(promiscuity), 축적, 독성 가능성 증가
- logP가 너무 낮으면: 세포막 투과성 저하
- Lipinski의 Rule of 5에서는 대략 **logP ≤ 5**를 권장합니다.

### 이 데이터에서의 clogP

| 항목 | 값 |
|---|---:|
| 평균 | 3.795 |
| 범위 | -4.05 ~ 9.43 |
| label 1(독성 없음) 평균 | 3.450 |
| label 0(독성 있음) 평균 | 4.206 |
| label과의 상관 | **-0.273** |

이 데이터에서는 clogP가 높을수록 독성(label 0) 쪽과 약하게 연관됩니다. 다만 상관계수 -0.273은 중간 이하이고 다른 구조적 요인이 많으므로 **인과관계로 해석하면 안 됩니다.**

---

## Q5. SA score의 의미

> “sa_score, 합성 가능성은 이 분자를 만드는 난이도를 이야기하는 건가?”

### 맞다. 합성 난이도 점수다

- **SA score** = **Synthetic Accessibility score**
- **낮을수록** 합성하기 쉬움 (대략 1)
- **높을수록** 합성하기 어려움 (대략 10)
- fragment 기반 기여도 + 고리 복잡도, 입체중심, macrocycle 등의 penalty로 계산
- RDKit의 `sascorer`가 대표적인 구현

### 주의

- 실제 **retrosynthesis 경로를 설계해 주는 값이 아닙니다.**
- “이 분자는 반드시 합성 가능/불가능하다”를 보장하지 않습니다.
- 시판 building block, 보호기 전략, 반응 조건 등은 고려하지 않는 heuristic입니다.

### 이 데이터에서의 SA score

| 항목 | 값 |
|---|---:|
| 평균 | 3.188 |
| 범위 | 1.28 ~ 7.31 |
| label 1(독성 없음) 평균 | 3.240 |
| label 0(독성 있음) 평균 | 3.126 |
| label과의 상관 | **+0.078** (매우 약함) |

---

## 한눈에 보는 비교표

| 개념 | 측정 대상 | 값의 의미 | 독성과의 직접 관계 |
|---|---|---|---|
| **Fingerprint** | 2D 부분구조 패턴 | 부분구조 존재 여부(0/1 bit) | 간접적 |
| **SMILES** | 분자 그래프의 문자열 표현 | 구조를 직렬화한 식별자 | 간접적 |
| **QED** | 약물 유사성(drug-likeness) | 높을수록 더 drug-like | **없음(직접)** |
| **clogP** | 지용성(logP 계산값) | 높을수록 더 친유성 | 약한 연관 |
| **SA score** | 합성 난이도 | 높을수록 더 합성하기 어려움 | 약한 연관 |

---

## 이 프로젝트에서의 실무적 의미

1. **라벨 극성**: `1 = non-toxic`, `0 = toxic`입니다. Precision/Recall/Specificity를 해석할 때 반드시 이 정의를 확인합니다.
2. **피처**: ECFP + FCFP + PTFP + 4개 기술자를 사용했습니다. Group CV 기준 combined ROC-AUC는 **0.836**이었습니다.
3. **검증**: 동일 fingerprint를 가진 입체이성질체가 242그룹/536행 있으므로, `fp_group` 기준 **StratifiedGroupKFold**를 사용했습니다. 34그룹/75행은 같은 fingerprint인데 label이 0/1로 혼재합니다.
4. **QED/clogP/SA score**: 독성 label이 아니라 **보조 기술자**입니다. 단독으로 “독성”을 판단하는 지표로 쓰면 안 됩니다.
5. **SMILES**: 식별자와 입체화학 확인용으로 사용합니다. SMILES 기반 모델(GNN, transformer)은 후속 비교 대상이 될 수 있습니다.
6. **다음 단계**: LightGBM/XGBoost 비교, threshold tuning, `predict_input.csv` 927행 추론, `SMILES,label` 제출 파일 생성.

---

## 참고문헌

- Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). *Quantifying the chemical beauty of drugs.* Nature Chemistry, 4(2), 90–98. https://doi.org/10.1038/nchem.1243
- Rogers, D., & Hahn, M. (2010). *Extended-connectivity fingerprints.* Journal of Chemical Information and Modeling, 50(5), 742–754. https://doi.org/10.1021/ci100050t
- Ertl, P., & Schuffenhauer, A. (2009). *Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.* Journal of Cheminformatics, 1, 8. https://doi.org/10.1186/1758-2946-1-8
- Wildman, S. A., & Crippen, G. M. (1999). *Prediction of physicochemical parameters by atomic contributions.* Journal of Chemical Information and Computer Sciences, 39(5), 868–873. https://doi.org/10.1021/ci990307l
- Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (1997). *Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings.* Advanced Drug Delivery Reviews, 23(1–3), 3–25. https://doi.org/10.1016/S0169-409X(96)00423-1
- RDKit Documentation. https://www.rdkit.org/docs/

---

## 후속 문서

- [입체화학 Fingerprint와 혼재 라벨 분석](toxicity_stereochemistry_followup.md)
- [Toxicity 분류 성능 개선 제안](toxicity_modeling_improvement_plan.md)
