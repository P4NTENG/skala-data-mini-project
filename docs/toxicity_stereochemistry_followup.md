# 입체화학 Fingerprint와 혼재 라벨 분석

> `toxicity_concepts_qa.md` 이후의 후속 논의를 정리한 문서입니다.  
> 대상: `toxicity_eda.ipynb`, `train.csv` (8,349행 × 3,078열)  
> 환경: Python 3.11.9, uv `.venv`, RDKit 2026.9.1  
> 라벨 정의: **1 = 독성 없음(non-toxic, Positive)**, **0 = 독성 있음(toxic, Negative)** (PDF 기준)

---

## 0. 용어 재정리

### Fingerprint는 어떻게 만들어지는가

정확히는 SMILES 문자열을 직접 해시하는 것이 아니라, **SMILES를 파싱해 만든 분자 그래프**에서 특징을 추출하고 해시합니다.

```text
SMILES
  → 분자 그래프 파싱
  → 원자 환경/부분구조 추출
  → 해시 함수로 고정 길이 bit 위치에 매핑
  → 1,024-bit vector
```

- **ECFP**: circular atom environment 기반
- **FCFP**: pharmacophore/functional-class feature 기반
- **PTFP**: 패턴/부분구조 기반
- Fingerprint는 **학습된 neural embedding이 아니라 hashed featurization**입니다.
- 해시 충돌이 가능하고, bit index 자체는 화학적 의미를 직접 갖지 않습니다.

### QED와 clogP

- **QED**: drug-likeness(약물 유사성) 0~1 점수. toxicity 점수가 아닙니다.
- **clogP**: 계산된 logP. 물/옥탄올 분배계수로 지용성을 나타냅니다. 용해도(solubility)와는 다릅니다.

자세한 내용은 `toxicity_concepts_qa.md`를 참고하세요.

---

## 1. 입체화학을 고려한 Fingerprint 도입 방법

### 1.1 기존 fingerprint를 나중에 수정할 수는 없다

기존 ECFP/FCFP/PTFP는 이미 **chirality를 포함하지 않는 설정**으로 해시된 결과입니다.  
따라서 기존 1,024-bit 벡터에 chirality 정보를 사후에 덧붙일 수는 없습니다.

가능한 방법은 두 가지입니다.

1. SMILES에서 분자를 다시 파싱해 **chirality-aware fingerprint를 재계산**
2. 기존 fingerprint는 유지하고 **stereo descriptor를 추가**

### 1.2 SMILES 기준 매핑

SMILES에 `@`, `@@`, `/`, `\`가 있으면 RDKit이 파싱할 때 입체화학을 보존합니다.  
이 데이터에는 해당 표기가 존재하므로 **행 단위 매핑이 가능**합니다.

```python
from rdkit import Chem
import pandas as pd

records = []
for idx, smi in df['SMILES'].items():
    mol = Chem.MolFromSmiles(smi)
    records.append({
        'idx': idx,
        'valid': mol is not None,
        'mol': mol,
    })
```

- `df.index`를 유지해 feature를 원래 행에 정확히 매핑합니다.
- `mol is None`인 경우를 대비해 `valid` mask를 함께 관리합니다.
- SMILES 문자열이 유일하더라도, **index 기반 매핑이 더 안전**합니다.

### 1.3 Chiral ECFP 계산

RDKit의 최신 API를 사용하는 예시입니다.

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,          # ECFP4
    fpSize=1024,
    includeChirality=True,
)

X_chiral = np.zeros((len(df), 1024), dtype=np.uint8)
valid = np.zeros(len(df), dtype=bool)

for i, smi in enumerate(df['SMILES'].tolist()):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = gen.GetFingerprint(mol)
    ConvertToNumpyArray(fp, X_chiral[i])
    valid[i] = True
```

- `includeChirality=True`는 tetrahedral chirality와 cis/trans double bond stereo를 반영합니다.
- `radius=2`는 ECFP4입니다. 원본 데이터의 radius가 metadata에 없으므로, **기존 ECFP와 동일 조건이라고 가정하면 안 됩니다.**
- 원본이 ECFP6이라면 `radius=3`으로 별도 실험해야 합니다.

> 참고: `AllChem.GetMorganFingerprintAsBitVect(..., useChirality=True)`도 사용할 수 있지만, 최신 RDKit에서는 `MorganGenerator` 사용이 권장됩니다.

### 1.4 FCFP와 PTFP의 chirality 주의점

#### FCFP
RDKit 2026.9.1에서 feature invariant 기반 FCFP에 `includeChirality=True`를 적용해도 **R/S enantiomer가 같은 fingerprint로 나오는 것**을 확인했습니다.

```python
# R/S 테스트 결과: 동일한 bit set
# C[C@H](F)Cl vs C[C@@H](F)Cl
```

즉, FCFP는 단순히 chirality flag를 켜는 것만으로는 chiral 정보를 얻기 어렵습니다.  
chiral FCFP가 필요하면 **feature invariant 자체에 chirality를 포함**하도록 별도 설계해야 합니다.

#### PTFP
PTFP는 표준 circular fingerprint가 아니라 패턴 기반입니다.  
원본 생성기가 chirality를 지원하는지 확인해야 하며, 지원하지 않으면 **achiral PTFP를 유지**하고 chiral ECFP/FCFP 또는 stereo descriptor를 추가하는 것이 현실적입니다.

### 1.5 Stereo descriptor 추가

Chiral fingerprint를 새로 만들지 않아도, 저차원 stereo descriptor를 추가할 수 있습니다.

```python
from rdkit.Chem import rdMolDescriptors

records = []
for idx, smi in df['SMILES'].items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        records.append({
            'idx': idx,
            'n_stereo_centers': np.nan,
            'n_unspecified_stereo': np.nan,
            'n_specified_stereo': np.nan,
        })
        continue

    n_total = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_unspec = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)

    records.append({
        'idx': idx,
        'n_stereo_centers': n_total,
        'n_unspecified_stereo': n_unspec,
        'n_specified_stereo': n_total - n_unspec,
    })

stereo = pd.DataFrame(records).set_index('idx')
df = df.join(stereo)
```

추가로 고려할 수 있는 값:

- `n_double_bond_stereo`: E/Z 이중결합 수
- `has_stereo`: 입체중심이 하나라도 있는지
- SMILES의 `@` 개수, `/`·`\` 개수 (간단한 프록시)
- InChIKey의 connectivity block과 full key

### 1.6 Bit-to-substructure mapping

특정 bit가 어떤 부분구조인지 확인하려면 `bitInfo`를 사용할 수 있습니다.

```python
bit_info = {}
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=1024, includeChirality=True
)
fp = gen.GetFingerprint(mol, bitInfo=bit_info)
```

- `bit_info`는 bit index → (atom index, radius) 정보를 제공합니다.
- bit index는 **radius, fpSize, 해시 방식**에 따라 달라집니다.
- 서로 다른 fingerprint 설정끼리 bit ID를 직접 비교하면 안 됩니다.
- 원본 PTFP의 bit mapping은 생성기 정보가 없으면 재현하기 어렵습니다.

---

## 2. Chiral fingerprint 성능 실험

### 2.1 실험 설정

- 데이터: `train.csv` 8,349행
- 검증: `StratifiedGroupKFold` 5-fold, `groups = fp_group` (achiral fingerprint hash)
- 모델: LogisticRegression (`liblinear`, C=1.0)
- Feature: 기술자 4개 + fingerprint
- Chiral ECFP: `radius=2`, `fpSize=1024`, `includeChirality=True`
- 평가: ROC-AUC, PR-AUC, F1(label=1), toxic recall(label=0)
- 혼재 라벨 그룹: 34개 / 75행

### 2.2 결과

| 설정 | ROC-AUC | PR-AUC | F1 | Toxic recall | 혼재 그룹 오류 |
|---|---:|---:|---:|---:|---:|
| 기존 fingerprint | **0.8358** | **0.8450** | 0.7814 | 0.7334 | 35/75 |
| + 단순 stereo descriptor | 0.8356 | 0.8447 | 0.7822 | 0.7344 | 34/75 |
| chiral ECFP로 교체 | 0.8353 | 0.8439 | 0.7834 | **0.7418** | 39/75 |
| chiral ECFP 추가 | 0.8335 | 0.8425 | 0.7805 | 0.7376 | 37/75 |

### 2.3 해석

- **전체 ROC-AUC/PR-AUC는 상승하지 않았습니다.**
- F1과 toxic recall은 설정에 따라 소폭 변했지만 일관된 개선이 아니었습니다.
- chiral ECFP는 혼재 그룹 34개 중 **27개를 실제로 구분**했습니다. 즉, 표현은 성공했지만 분류 성능으로 이어지지 않았습니다.
- 혼재 그룹 오류는 35건에서 39건/37건으로 오히려 늘었습니다.
  - chiral ECFP 교체: 5건 수정, 9건 악화
  - chiral ECFP 추가: 5건 수정, 7건 악화

### 2.4 왜 기대만큼 오르지 않았나

1. **혼재 라벨 그룹이 전체의 0.9%** (75/8,349)에 불과합니다.
2. **Group CV는 같은 2D 골격의 입체이성질체를 같은 fold에 묶습니다.**  
   따라서 모델이 “같은 골격의 다른 입체이성질체”를 train에서 학습할 기회가 없습니다.
3. 원본 fingerprint의 radius가 metadata에 없어, chiral ECFP4로 재계산한 결과는 **동일 조건 비교가 아닙니다.**
4. FCFP는 feature invariant 때문에 chirality flag만으로 R/S를 구분하지 못했습니다.
5. 단일 5-fold logistic 실험이므로, 모델/튜닝에 따라 결과가 달라질 수 있습니다.

### 2.5 그럼에도 도입할 가치가 있는 경우

- 라벨이 다른 입체이성질체 쌍이 훨씬 많을 때
- 독성이 특정 입체화학에 강하게 의존한다는 도메인 근거가 있을 때
- 전체 AUC가 아니라 **입체이성질체 쌍의 구분 정확도**가 목표일 때
- GNN, pairwise/contrastive learning, 3D conformer fingerprint를 사용할 때
- “새 2D 골격”이 아니라 “알려진 골격의 새 입체이성질체”를 예측할 때

이 경우에는 Group CV와 별도로 **stereoisomer-aware split**을 설계할 수 있습니다.  
다만 이는 낙관적인 평가가 될 수 있으므로, 최종 성능은 scaffold 기준 Group CV와 함께 보고해야 합니다.

---

## 3. 혼재 라벨 분석

### 3.1 규모

- 혼재 라벨 그룹: **34개**
- 해당 행: **75개**
- 구성:
  - 크기 2: 28개 그룹
  - 크기 3: 5개 그룹
  - 크기 4: 1개 그룹
- 라벨 패턴:
  - `01`: 28개 그룹
  - `011`: 3개 그룹
  - `001`: 2개 그룹
  - `0011`: 1개 그룹

### 3.2 피처별 진단

| 피처 | 그룹 내 변화 | 라벨 구분 가능? |
|---|---|---|
| ECFP/FCFP/PTFP 3,072 bits | **34/34 그룹에서 완전 동일** | 불가능 |
| `MolWt` | 34/34 그룹에서 동일 | 불가능 |
| `clogp` | 34/34 그룹에서 동일 | 불가능 |
| `qed` | 34/34 그룹에서 동일 | 불가능 |
| `sa_score` | **1/34 그룹에서만 다름** | 사실상 불가능 |
| SMILES `@` 개수 | 30/34 그룹에서 다름 | 원리적으로 가능하지만 방향이 일관되지 않음 |
| SMILES 문자열 | 모두 다름 | 입체화학 정보는 있지만 현재 모델 피처로 쓰지 않음 |

**33/34 그룹은 현재 모델이 보는 모든 수치 피처가 완전히 동일합니다.**  
같은 입력 벡터에는 같은 예측이 나올 수밖에 없으므로, 라벨이 다르면 구조적으로 오류가 불가피합니다.

`n_at`은 30/34 그룹에서 변하지만, label 방향은 일관되지 않습니다.

- label 1의 `n_at`이 더 높은 그룹: 14개
- label 1의 `n_at`이 더 낮은 그룹: 14개
- 두 label의 `n_at`이 같은 그룹: 6개

따라서 `n_at` 하나를 전역 선형 feature로 추가해도 방향이 상쇄되어, 단순 stereo descriptor 모델은 혼재 그룹 오류를 35건 → 34건으로 1건 줄이는 데 그쳤습니다.

### 3.3 이론적 최소 오류

동일한 피처를 가진 그룹에서는 다수 라벨로 예측할 수밖에 없습니다.

- 33개 그룹(2행 27개, 3행 5개, 4행 1개)의 이론적 최소 오류:
  - 2행 27개 → 각 1건 = 27
  - 3행 5개 → 각 1건 = 5
  - 4행 1개(2:2) → 2건
  - **합계 34건**
- 나머지 1개 그룹(19905244)은 `sa_score`가 달라 이론적으로는 구분 가능하지만, 2행뿐입니다.

베이스라인은 혼재 그룹에서 **35건**을 틀렸습니다. 이는 33개 동일 피처 그룹의 최소 오류 34건에 `sa_score` 예외 그룹의 1건이 더해진 수준입니다. 즉, 현재 피처 기준 최선(34건)에 **1건 차이**로 거의 도달해 있습니다.

- 전체 오류 1,994건 중 혼재 그룹 오류는 35건(1.8%)
- 이 35건을 모두 없애도 accuracy 상승은 약 **0.42%p**
- 나머지 94.4%의 오류는 중복 없는 일반 분자에서 발생

### 3.4 혼재 라벨 그룹 목록

`n_at`은 SMILES의 `@` 문자 수(입체화학 표기의 대략적 프록시)입니다.  
전체 ID와 SMILES는 `mixed_label_group_summary.csv`, `mixed_label_groups.csv`에 있습니다.

| 그룹 | 행 수 | 라벨 | `n_at` (label 0 / label 1) | `sa_score` 범위 |
|---|---:|---|---|---:|
| 61412490 | 4 | 0011 | 6,6 / 6,6 | 0.000 |
| 24265550 | 3 | 001 | 0,1 / 2 | 0.000 |
| 62750325 | 3 | 011 | 1 / 0,2 | 0.000 |
| 85012080 | 3 | 001 | 0,3 / 3 | 0.000 |
| 11635679 | 3 | 011 | 1 / 0,2 | 0.000 |
| 13803131 | 3 | 011 | 4 / 3,3 | 0.000 |
| 55835975 | 2 | 01 | 3 / 4 | 0.000 |
| 61671790 | 2 | 01 | 1 / 2 | 0.000 |
| 13217385 | 2 | 01 | 2 / 1 | 0.000 |
| **19905244** | 2 | 01 | 6 / 0 | **0.307** |
| 37952587 | 2 | 01 | 3 / 3 | 0.000 |
| 49438012 | 2 | 01 | 0 / 2 | 0.000 |
| 56335736 | 2 | 01 | 4 / 3 | 0.000 |
| 79137887 | 2 | 01 | 1 / 2 | 0.000 |
| 85124326 | 2 | 01 | 3 / 3 | 0.000 |
| 89577415 | 2 | 01 | 3 / 3 | 0.000 |
| 93664464 | 2 | 01 | 4 / 3 | 0.000 |
| 10409521 | 2 | 01 | 3 / 4 | 0.000 |
| 10612815 | 2 | 01 | 1 / 0 | 0.000 |
| 10768322 | 2 | 01 | 2 / 0 | 0.000 |
| 10899613 | 2 | 01 | 1 / 2 | 0.000 |
| 11094106 | 2 | 01 | 4 / 3 | 0.000 |
| 11882787 | 2 | 01 | 2 / 1 | 0.000 |
| 12243044 | 2 | 01 | 3 / 2 | 0.000 |
| 12830016 | 2 | 01 | 2 / 1 | 0.000 |
| 13398867 | 2 | 01 | 1 / 0 | 0.000 |
| 14124203 | 2 | 01 | 1 / 2 | 0.000 |
| 14686893 | 2 | 01 | 2 / 3 | 0.000 |
| 14743536 | 2 | 01 | 2 / 1 | 0.000 |
| 16044998 | 2 | 01 | 1 / 2 | 0.000 |
| 16570566 | 2 | 01 | 2 / 1 | 0.000 |
| 18068908 | 2 | 01 | 1 / 2 | 0.000 |
| 18367005 | 2 | 01 | 9 / 10 | 0.000 |
| 18418482 | 2 | 01 | 3 / 4 | 0.000 |

### 3.5 주목할 케이스

#### 그룹 61412490 — 4개 입체이성질체, 라벨 0/0/1/1
- `n_at`이 모두 6으로 같고, `MolWt`/`clogp`/`sa_score`/`qed`도 모두 동일
- 현재 피처로는 **최소 2건의 오류가 불가피**
- 2D 골격은 같고 입체화학만 다른데 라벨이 갈리는 전형적인 사례

#### 그룹 19905244 — 유일하게 `sa_score`가 다른 그룹
- label 1: 입체화학 표기 없는 SMILES, `sa_score` 4.4898
- label 0: `@` 6개를 가진 SMILES, `sa_score` 4.7970
- `MolWt`/`clogp`/`qed`는 동일
- 이론적으로는 `sa_score`로 구분할 수 있지만 **2행뿐**이라 일반화 근거로 쓰기 어렵습니다.

### 3.6 InChIKey 감사

RDKit으로 각 행의 InChIKey를 계산했습니다.

- **34/34 그룹**: 동일한 connectivity block을 가짐 → 같은 2D 골격
- **34/34 그룹**: 서로 다른 full InChIKey를 가짐 → 입체화학이 다른 stereoisomer
- **같은 full InChIKey인데 라벨이 다른 경우: 0건** → 완전히 동일한 분자의 라벨 충돌은 아님

즉, 단순한 “같은 분자가 두 번 라벨링된 오류”는 아닙니다.  
가능성은 두 가지입니다.

1. **진짜 입체화학 의존적 독성** — R/S, diastereomer에 따라 독성이 실제로 다를 수 있음
2. **assay 노이즈/라벨 오류** — 실험 조건, 농도, 세포주 등에 따른 변동

현재 피처만으로는 어느 쪽인지 판별할 수 없습니다. 원본 assay metadata나 출처를 확인해야 합니다.

---

## 4. 결론 및 권고

### 결론

- 현재 피처로 혼재 라벨을 구분하는 것은 **원리적으로 불가능한 그룹이 33/34**입니다.
- 33개 동일 피처 그룹의 최소 오류는 **34건**이며, 베이스라인은 `sa_score` 예외 1건을 포함해 **35건**으로 거의 최선에 도달했습니다.
- chiral ECFP는 기술적으로 chirality를 표현했지만, **전체 분류 성능 개선으로 이어지지 않았습니다.**
- 혼재 그룹은 전체 오류의 1.8%에 불과하므로, 전체 성능 개선의 우선순위는 낮습니다.

### 권고

1. **75행 라벨 감사**
   - 원본 assay/source metadata를 확인
   - 같은 connectivity + 다른 입체화학 + 다른 라벨이 실제인지, 라벨 오류인지 판단
   - 라벨 오류라면 정리하는 것이 chiral feature 추가보다 효과적일 수 있음
2. **Chiral feature는 ablation으로**
   - chiral ECFP + stereo descriptor를 추가해 Group CV로 비교
   - 여러 seed/fold에서 평균과 표준편차를 확인
   - 전체 AUC가 일관되게 오를 때만 채택
3. **우선순위**
   - LightGBM/XGBoost 비교
   - feature selection
   - threshold tuning / calibration
   - 라벨 품질 개선
4. **평가 설계**
   - 최종 지표는 `fp_group` 기준 Group CV 유지
   - 입체이성질체 전용 평가가 필요하면 별도의 stereoisomer-aware split을 추가로 보고

---

## 5. 재현 및 산출물

### 환경

```bash
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe pandas numpy matplotlib seaborn scikit-learn scipy ipykernel jupyterlab rdkit
```

### 관련 파일

- `mixed_label_groups.csv` — 혼재 라벨 75행 전체 (SMILES, label, descriptor, `n_at`, InChIKey, connectivity)
- `mixed_label_group_summary.csv` — 혼재 라벨 34개 그룹 요약
- `toxicity_eda.ipynb` — 기본 EDA 및 베이스라인
- `toxicity_concepts_qa.md` — 선행 개념 Q&A

> `toxicity_eda.ipynb`는 RDKit 설치 전에 실행되어 optional RDKit 셀 출력이 “not installed”로 저장되어 있습니다.  
> 현재 `.venv`에는 RDKit이 설치되어 있으므로, 해당 셀을 다시 실행하면 SMILES 유효성 검사를 수행합니다.

---

## 6. 참고

- Bickerton et al. (2012). *Quantifying the chemical beauty of drugs.* Nature Chemistry, 4, 90–98.
- Rogers & Hahn (2010). *Extended-connectivity fingerprints.* J Chem Inf Model, 50, 742–754.
- Ertl & Schuffenhauer (2009). *Estimation of synthetic accessibility score.* J Cheminform, 1, 8.
- Wildman & Crippen (1999). *Prediction of physicochemical parameters by atomic contributions.* J Chem Inf Comput Sci, 39, 868–873.
- Lipinski et al. (1997). *Experimental and computational approaches to estimate solubility and permeability.* Adv Drug Deliv Rev, 23, 3–25.
- RDKit Documentation. https://www.rdkit.org/docs/

---

## 7. 관련 문서

- [분자 독성 EDA 개념 Q&A](toxicity_concepts_qa.md)
- [Chiral Embedding의 공간 구조와 비중복 입체 분자](toxicity_chiral_embedding_geometry.md)
- [Toxicity 분류 성능 개선 제안](toxicity_modeling_improvement_plan.md)
- [Mordred Descriptor 검토 및 추가 제안](toxicity_mordred_feature_proposal.md)
