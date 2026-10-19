# Chiral Embedding의 공간 구조와 비중복 입체 분자

> `toxicity_stereochemistry_followup.md` 이후의 심화 논의입니다.  
> 핵심 질문: **chirality-aware embedding이 기존 fingerprint와 겹치지 않는 입체 분자에서도 성능을 올릴 수 있는가?**  
> 대상: `train.csv` (8,349행 × 3,078열), `toxicity_eda.ipynb`  
> 라벨: **1 = non-toxic (Positive)**, **0 = toxic (Negative)** (PDF 기준)

---

## 1. 질문의 재정의

처음에는 “분자의 결합 연결성(topology)이 바뀌는가”로 이해할 수 있지만, 실제 논의는 **임베딩 공간에서 정보의 위치·이웃·밀도가 어떻게 바뀌는가**에 대한 것입니다.

정확히는:

> **Chiral fingerprint는 분자 그래프의 연결성을 바꾸지 않는다.  
> 대신 원자 환경/부분구조를 해시할 때 chirality를 포함해, 같은 분자가 임베딩 공간의 다른 위치로 매핑되도록 만든다.**

이 문서에서는 “토폴로지” 대신 **임베딩 공간의 분할(partition), 이웃 구조(neighborhood), 밀도(density)**라는 표현을 사용합니다.

---

## 2. 임베딩 공간에서 변하는 것

### Achiral fingerprint

- R/S enantiomer가 **같은 bit/bucket**에 매핑
- 두 점의 거의 동일한 위치 → Hamming/Jaccard 거리 ≈ 0
- 해당 영역에 분자가 많이 모여 **표본 밀도가 높음**
- 장점: 통계적 지지력이 높고 안정적
- 단점: R/S의 라벨 차이를 원리적으로 구분할 수 없음

### Chiral fingerprint

- R/S가 **서로 다른 bit로 분리**
- 두 점 사이의 거리가 커지고, 이웃 관계가 바뀜
- 각 bucket에 속하는 분자 수가 줄어 **표본 밀도가 낮아짐**
- 장점: 같은 2D 구조의 R/S를 구분할 수 있음
- 단점: bit가 희귀해지고, 결정 경계가 불안정해질 수 있음

```text
Achiral:  R,S  →  [ same bucket ]        (high density)
Chiral:   R    →  [ bucket A ]           (lower density)
          S    →  [ bucket B ]           (lower density)
```

즉, **해상도(resolution)는 올라가지만 표본 밀도(density)는 떨어지는 trade-off**가 발생합니다.

> 이산적인 bit 재배치이므로, 연속적인 “공간 변형”이라기보다는 **feature partition의 재구성**에 가깝습니다.

---

## 3. 비중복 입체 분자 subset 결과

Group CV 5-fold LogisticRegression, OOF 예측 기준입니다.  
“비중복”은 정확히 같은 fingerprint 벡터를 가진 그룹에 속하지 않는 행을 의미합니다.

| Subset | n | Baseline AUC | Chiral ECFP 교체 | Chiral ECFP 추가 |
|---|---:|---:|---:|---:|
| 전체 | 8,349 | **0.8358** | 0.8353 | 0.8335 |
| 비중복 전체 | 7,813 | **0.8319** | 0.8309 | 0.8291 |
| **비중복 + 입체화학** | **3,484** | **0.8274** | **0.8249** | **0.8246** |
| 비중복 + 비입체 | 4,329 | 0.8328 | 0.8332 | 0.8302 |
| 중복 fingerprint 행 | 536 | 0.8897 | **0.8943** | 0.8932 |

### 해석

- **비중복 입체 3,484행**에서 baseline 0.8274 → chiral 교체 0.8249로 **-0.0025** 하락
- chiral 추가도 0.8246으로 **-0.0028** 하락
- **중복 fingerprint 행 536행**에서는 0.8897 → 0.8943으로 **소폭 상승**
- 전체 AUC는 0.8358 → 0.8353/0.8335로 상승하지 않음

### F1 / toxic recall의 미세한 변화

비중복 입체 subset:

| 설정 | F1 | Toxic recall |
|---|---:|---:|
| Baseline | 0.7984 | 0.7068 |
| Chiral 교체 | 0.7990 | 0.7213 |
| Chiral 추가 | 0.7960 | 0.7075 |

F1과 toxic recall은 일부 소폭 올랐지만, ROC-AUC/PR-AUC는 하락했습니다.  
즉, **일관된 성능 개선이 아니라 threshold/클래스별 trade-off**로 해석해야 합니다.

---

## 4. 왜 비중복 입체 분자에서 효과가 작았나

### 4-1. 해상도 vs 표본 밀도

Chiral bit는 R/S를 분리하지만, 각 bit의 양성 빈도가 낮아집니다.

- 1,024-bit 공간에서 8,349개 분자
- chiral bit 중 상당수는 수십 개 이하의 분자에서만 등장
- 모델 입장에서는 **희귀하고 불안정한 feature**
- 기존 achiral bit가 공유하던 통계적 지지력을 잃음

이는 **bias-variance trade-off**로 설명할 수 있습니다.

- Achiral: 해상도 낮음, 분산 낮음 (많은 표본)
- Chiral: 해상도 높음, 분산 높음 (적은 표본)

비중복 분자에서는 새로운 scaffold로 일반화해야 하므로, 분산이 높아지는 것이 특히 불리합니다.

### 4-2. 같은 scaffold의 R/S 비교 쌍이 없다

중복 그룹에서는 같은 2D 구조의 R/S가 함께 있어 chiral bit가 실제 라벨 차이를 분리합니다.  
비중복 분자에서는 비교할 R/S 쌍이 없고, 모델은 다른 scaffold에서 배운 chirality 패턴을 전이해야 합니다.

- chirality 효과가 scaffold-specific하면 → 전이 실패
- chirality 효과가 전역적이어도 → 표본이 적어 학습이 어려움

### 4-3. Chemical space의 분리 ≠ Label space의 정렬

Chiral ECFP는 **화학 공간에서 R/S를 분리**합니다.  
하지만 toxicity label이 그 분리 방향과 일치한다는 보장은 없습니다.

- 화학적으로는 R/S가 다르게 표현됨
- 독성은 다른 구조적 요인에 더 크게 좌우될 수 있음
- R/S 차이의 방향이 scaffold마다 다를 수 있음

성능 향상은 **chemical space의 재배치**와 **label space의 구조**가 정렬될 때만 발생합니다.

### 4-4. Group CV는 scaffold 단위로 hold-out한다

현재 Group CV는 `fp_group` 기준으로 같은 2D 구조를 통째로 train/valid에 나눕니다.

- 모델은 “새로운 2D 골격”에 대해 chirality 패턴을 일반화해야 함
- 같은 scaffold의 R/S를 train에서 보고 valid에서 맞히는 상황이 아님
- 따라서 chiral feature의 이점이 **중복 그룹 내부**에 머무르기 쉬움

### 4-5. Label noise 가능성

34개 혼재 그룹/75행은 실제 stereoisomer이지만,

1. 진짜 입체화학 의존적 독성
2. assay noise / label 오류

를 구조만으로 구분할 수 없습니다.  
상당수가 라벨 노이즈라면 chiral embedding을 아무리 잘 만들어도 성능이 오르지 않습니다.

---

## 5. Hand-crafted chiral fingerprint vs learned embedding

| 구분 | Chiral ECFP | Chiral GNN / metric learning |
|---|---|---|
| 임베딩 공간 변화 | **label과 무관하게** 고정된 규칙으로 재배치 | **label을 보고** 공간을 학습 |
| R/S 분리 | 보장 | 학습에 따라 달라짐 |
| 비중복 일반화 | 전역 규칙이 없으면 어려움 | 전역 규칙을 학습할 수 있음 |
| 데이터 요구량 | 상대적으로 적음 | 많음 |
| 리스크 | bit 희소성, 정보 손실 | 과적합, scaffold split |

Chiral ECFP는 **“화학적으로 더 세밀한 표현”**을 주지만, **“독성 예측에 유리한 방향으로 공간을 재배치”**하지는 않습니다.  
반면 chiral GNN이나 metric learning은 label을 이용해 **toxicity-relevant geometry**를 학습할 수 있습니다.

즉, “chiral ECFP가 비중복에서 안 된다”가 “학습형 chiral embedding도 안 된다”는 뜻은 아닙니다.

---

## 6. “전면 교체”의 리스크

| 개입 | 기존 achiral 정보 | chirality 정보 | 리스크 |
|---|---|---|---|
| Chiral ECFP 추가 | 유지 | 추가 | 차원 증가, 희소성 |
| ECFP만 교체 | 일부 상실 | 추가 | 기존 ECFP 정보 손실 |
| **전체 fingerprint 교체** | **대부분 상실 가능** | 추가 | 기존 신호 손실 |
| Chiral GNN | 표현 자체를 학습 | 모델이 학습 | 데이터 규모/튜닝 리스크 |
| 3D conformer fingerprint | 2D 정보 일부 상실 | conformation 추가 | conformer 비용/재현성 |

전면 교체는 이론적으로 더 큰 향상을 줄 수 있지만, **기존 achiral fingerprint가 이미 0.836을 내고 있는 상황**에서 이를 제거하는 것은 정보 손실 리스크가 큽니다.  
현재 chiral ECFP 교체에서 비중복 입체 subset이 하락한 것이 그 신호입니다.

또한 “전면 교체”를 하려면 FCFP/PTFP의 chiral 버전이 필요합니다.

- FCFP: feature invariant 기반이라 `includeChirality=True`만으로 R/S가 구분되지 않았음
- PTFP: 패턴 기반, chirality 지원 불확실
- 따라서 실제로는 **chiral ECFP + achiral FCFP/PTFP** 하이브리드가 되기 쉬움

---

## 7. 제대로 검증하려면

### 7-1. Subset metric

- `stereo & unique` 3,484행에서 AUC/PR/F1/toxic recall 비교
- 전체 AUC가 아니라 이 subset을 primary로
- `nonstereo & unique`는 대조군으로 함께 확인

### 7-2. Within-group pairwise accuracy

- 34개 혼재 그룹 각각에서
  - “toxic stereoisomer에 더 높은 toxic probability를 주는가”
- 그룹별 pairwise accuracy의 평균과 신뢰구간 비교
- 전체 AUC보다 chirality의 직접 효과를 측정

### 7-3. Stereoisomer-aware split

- 같은 2D connectivity 그룹 안에서 일부 stereoisomer를 train, 다른 stereoisomer를 validation
- “새 입체이성질체 예측” 시나리오
- 단, 같은 scaffold를 공유하므로 낙관적일 수 있음
- 최종 성능은 기존 `fp_group` Group CV와 함께 보고

### 7-4. 표현 ablation

- 1,024 vs 2,048/4,096 bits
- radius 2 vs 3
- binary vs count fingerprint
- `useChirality=True` ECFP only vs chiral FCFP/PTFP (가능한 경우)
- chiral ECFP + stereo descriptor

### 7-5. Label audit

- 75행의 원본 assay/source 확인
- 실제 chirality-dependent인지, 라벨 오류인지 판단
- 라벨 오류라면 feature 개선보다 라벨 정리가 우선

---

## 8. 결론

> **Chiral-aware embedding은 임베딩 공간에서 R/S를 분리해 이웃 구조와 밀도를 바꾼다.  
> 그러나 그 재배치가 label space와 정렬되지 않거나, 각 영역의 표본이 너무 적으면, 비중복 입체 분자에서는 오히려 일반화가 어려워질 수 있다.**

정리하면:

1. “임베딩 공간의 위치 형태가 바뀐다”는 이해는 **맞습니다.**
2. 다만 그것은 **분자 그래프의 topology 변형이 아니라, feature partition의 재구성**입니다.
3. Chiral ECFP는 **chemical space를 더 세밀하게** 만들지만, **toxicity label과의 정렬**을 보장하지 않습니다.
4. 현재 실험에서는 **비중복 입체 3,484행에서 AUC가 하락**했습니다.
5. 효과는 **중복 fingerprint 그룹 536행**에서만 소폭 나타났습니다.
6. **전면 교체는 기존 achiral 정보를 잃을 위험**이 있어, 추가 실험 없이 채택하면 안 됩니다.
7. 학습형 chiral GNN / 3D conformer fingerprint는 **아직 미검증**이며, 별도 비용/데이터 규모 검토가 필요합니다.

---

## 9. 관련 문서

- [분자 독성 EDA 개념 Q&A](toxicity_concepts_qa.md)
- [입체화학 Fingerprint와 혼재 라벨 분석](toxicity_stereochemistry_followup.md)
- [Toxicity 분류 성능 개선 제안](toxicity_modeling_improvement_plan.md)
- [Mordred Descriptor 검토 및 추가 제안](toxicity_mordred_feature_proposal.md)

## 10. 참고

- Rogers & Hahn (2010). *Extended-connectivity fingerprints.* J Chem Inf Model, 50, 742–754.
- RDKit Documentation. https://www.rdkit.org/docs/
- Mordred Descriptor List. https://mordred-descriptor.github.io/documentation/master/descriptors.html
