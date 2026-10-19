# Mordred Descriptor 검토 및 추가 제안

> `train.csv` (8,349행 × 3,078열) 기반 분자 독성 이진 분류 모델에 **Mordred descriptor**를 추가할 때의 우선순위와 근거를 정리한 문서입니다.  
> 기준 문서: Mordred 1.2.1a1 documentation, Descriptor List  
> 라벨 정의: **1 = 독성 없음(non-toxic, Positive)**, **0 = 독성 있음(toxic, Negative)** (PDF 기준)

---

## 1. 현재 상황

- 데이터: 8,349개 분자
- Feature:
  - ECFP 1,024 bits
  - FCFP 1,024 bits
  - PTFP 1,024 bits
  - 기존 descriptor 4개: `MolWt`, `clogp`, `sa_score`, `qed`
- Group CV 5-fold LogisticRegression 기준:
  - Descriptors 4개만: ROC-AUC **0.668**
  - ECFP: **0.790**
  - All fingerprints: **0.832**
  - Descriptors + all fingerprints: **0.836**

기존 descriptor 4개를 추가했을 때의 상승은 0.832 → 0.836으로 **+0.004** 수준이었습니다.  
즉, 현재 fingerprint가 이미 강한 신호를 가지고 있으며, Mordred descriptor도 **전부 추가하기보다는 fingerprint가 놓치는 축을 골라 증분 추가**해야 합니다.

---

## 2. 검토 범위

Mordred 문서의 **1,826개 descriptor, 50개 모듈**을 확인했습니다.

- **2D descriptor: 1,613개**
- **3D descriptor: 213개**
- 2D-only 모듈: 44개
- 3D descriptor를 포함한 모듈: 6개
  - `MoRSE` (160)
  - `CPSA` (41 of 43)
  - `GeometricalIndex` (4)
  - `GravitationalIndex` (4)
  - `MomentOfInertia` (3)
  - `PBF` (1)

### 전체 모듈 요약

| 모듈 | descriptor 수 | 차원 | 대표 descriptor |
|---|---:|---|---|
| `Autocorrelation` | 606 | 2D 606 | `ATS0dv`, `ATS1dv`, `ATS2dv`, ... |
| `EState` | 316 | 2D 316 | `NsLi`, `NssBe`, `NssssBe`, ... |
| `MoRSE` | 160 | 3D 160 | `Mor01`, `Mor02`, `Mor03`, ... |
| `RingCount` | 138 | 2D 138 | `nRing`, `n3Ring`, `n4Ring`, ... |
| `BaryszMatrix` | 104 | 2D 104 | `SpAbs_DzZ`, `SpMax_DzZ`, ... |
| `Chi` | 56 | 2D 56 | `Xch-3d`, `Xch-4d`, ... |
| `MoeType` | 53 | 2D 53 | `LabuteASA`, `PEOE_VSA1`, ... |
| `ExtendedTopochemicalAtom` | 45 | 2D 45 | `ETA_alpha`, `AETA_alpha`, ... |
| `CPSA` | 43 | 3D 41 / 2D 2 | `PNSA1`, `PNSA2`, ... |
| `InformationContent` | 42 | 2D 42 | `IC0`, `IC1`, `IC2`, ... |
| `BCUT` | 24 | 2D 24 | `BCUTc-1h`, `BCUTdv-1h`, ... |
| `PathCount` | 21 | 2D 21 | `MPC2`, `MPC3`, ... |
| `TopologicalCharge` | 21 | 2D 21 | `GGI1`, `GGI2`, ... |
| `WalkCount` | 21 | 2D 21 | `MWC01`, `MWC02`, ... |
| `MolecularDistanceEdge` | 19 | 2D 19 | `MDEC-11`, `MDEC-12`, ... |
| `AtomCount` | 17 | 2D 17 | `nAtom`, `nHeavyAtom`, `nSpiro`, ... |
| `Constitutional` | 16 | 2D 16 | `SZ`, `Sm`, `Sv`, ... |
| `DetourMatrix` | 14 | 2D 14 | `SpAbs_Dt`, `SpMax_Dt`, ... |
| `AdjacencyMatrix` | 12 | 2D 12 | `SpAbs_A`, `SpMax_A`, ... |
| `DistanceMatrix` | 12 | 2D 12 | `SpAbs_D`, `SpMax_D`, ... |
| `MolecularId` | 12 | 2D 12 | `MID`, `AMID`, ... |
| `CarbonTypes` | 11 | 2D 11 | `C1SP1`, `C1SP2`, `FCSP3`, ... |
| `BondCount` | 9 | 2D 9 | `nBonds`, `nBondsO`, `nBondsD`, ... |
| `GeometricalIndex` | 4 | 3D 4 | `GeomDiameter`, `GeomRadius`, ... |
| `GravitationalIndex` | 4 | 3D 4 | `GRAV`, `GRAVH`, ... |
| `TopologicalIndex` | 4 | 2D 4 | `Diameter`, `Radius`, ... |
| `ZagrebIndex` | 4 | 2D 4 | `Zagreb1`, `Zagreb2`, ... |
| `KappaShapeIndex` | 3 | 2D 3 | `Kier1`, `Kier2`, `Kier3` |
| `MomentOfInertia` | 3 | 3D 3 | `MOMI-X`, `MOMI-Y`, `MOMI-Z` |
| `ABCIndex` | 2 | 2D 2 | `ABC`, `ABCGG` |
| `AcidBase` | 2 | 2D 2 | `nAcid`, `nBase` |
| `Aromatic` | 2 | 2D 2 | `nAromAtom`, `nAromBond` |
| `HydrogenBond` | 2 | 2D 2 | `nHBAcc`, `nHBDon` |
| `Lipinski` | 2 | 2D 2 | `Lipinski`, `GhoseFilter` |
| `Polarizability` | 2 | 2D 2 | `apol`, `bpol` |
| `RotatableBond` | 2 | 2D 2 | `nRot`, `RotRatio` |
| `SLogP` | 2 | 2D 2 | `SLogP`, `SMR` |
| `TopoPSA` | 2 | 2D 2 | `TopoPSA(NO)`, `TopoPSA` |
| `Weight` | 2 | 2D 2 | `MW`, `AMW` |
| `WienerIndex` | 2 | 2D 2 | `WPath`, `WPol` |
| `BalabanJ` | 1 | 2D 1 | `BalabanJ` |
| `BertzCT` | 1 | 2D 1 | `BertzCT` |
| `EccentricConnectivityIndex` | 1 | 2D 1 | `ECIndex` |
| `FragmentComplexity` | 1 | 2D 1 | `fragCpx` |
| `Framework` | 1 | 2D 1 | `fMF` |
| `LogS` | 1 | 2D 1 | `FilterItLogS` |
| `McGowanVolume` | 1 | 2D 1 | `VMcGowan` |
| `PBF` | 1 | 3D 1 | `PBF` |
| `VdwVolumeABC` | 1 | 2D 1 | `Vabc` |
| `VertexAdjacencyInformation` | 1 | 2D 1 | `VAdjMat` |

> 상위 4개 모듈(`Autocorrelation` 606, `EState` 316, `MoRSE` 160, `RingCount` 138)이 전체 1,826개의 약 **66.8%**를 차지합니다. 이들을 전부 넣으면 차원과 다중공선성이 급격히 커집니다.

---

## 3. 핵심 판단

### 3-1. 1,826개를 전부 넣지 말 것

- `Autocorrelation` 606개, `EState` 316개, `BaryszMatrix` 104개, `Chi` 56개 등은 서로 고도로 상관된 topological/electronic descriptor입니다.
- 8,349행에 1,826개를 더하면 **p/n 비율이 나빠지고**, LightGBM/logistic 모두 노이즈를 학습할 위험이 커집니다.
- 기존 fingerprint 3,072 bits와 중복되는 정보도 많습니다.

### 3-2. Fingerprint가 놓치는 축을 우선

ECFP/FCFP/PTFP는 **국소 부분구조의 존재 여부**를 봅니다.  
Mordred의 강점은 **전역 물성·전자 상태·골격 복잡도**입니다.

현재 4개 descriptor에 없는 축:

- Polar surface area (`TopoPSA`)
- Hydrogen bonding (`nHBAcc`, `nHBDon`)
- Ionization (`nAcid`, `nBase`)
- Flexibility (`nRot`, `RotRatio`)
- Aromaticity/ring system (`nAromAtom`, `nARing`, `nAHRing`)
- Halogen/heteroatom composition (`nX`, `nCl`, `nBr`, `nI`, `nN`, `nO`, `nS`)
- sp3 character (`FCSP3`, `HybRatio`)
- Refractivity/polarizability (`SMR`, `apol`, `bpol`)
- Solubility (`FilterItLogS`)
- Framework complexity (`fMF`, `fragCpx`, `nSpiro`, `nBridgehead`)

### 3-3. 기존 descriptor와 중복되는 것은 후순위

- `MW`, `AMW` → `MolWt`와 중복
- `SLogP` → `clogp`와 중복 가능
- `Lipinski`, `GhoseFilter` → QED와 rule-based 중복 가능
- `TopoPSA`와 3D CPSA의 `TPSA` → 2D `TopoPSA`를 먼저
- `RingCount` 138개 전체 → 대표 ring count만
- `Autocorrelation` 606개 전체 → lag 0~4의 일부 weight만

---

## 4. 우선순위 descriptor

### Tier 1 — 가장 먼저 추가할 약 40개

| Descriptor | 모듈 | 근거 |
|---|---|---|
| `TopoPSA`, `TopoPSA(NO)` | TopoPSA | 극성 표면적. 막 투과성·수송체·조직 분포와 관련 |
| `nHBAcc`, `nHBDon` | HydrogenBond | 수소결합. 용해도·투과성·hERG/phospholipidosis와 간접 연관 |
| `nAcid`, `nBase` | AcidBase | 이온화 작용기. 생리적 pH 전하 상태·흡수·분포 |
| `nRot`, `RotRatio` | RotatableBond | 유연성. promiscuity·막 상호작용·대사 안정성 |
| `nAromAtom`, `nAromBond` | Aromatic | 방향족성/평면성. hERG·mutagenicity·CYP |
| `nRing`, `nARing`, `nHRing`, `nAHRing`, `nFRing` | RingCount | 방향족/헤테로방향족/융합 고리. 수용체 결합·독성 motif |
| `nSpiro`, `nBridgehead` | AtomCount | 골격 복잡도·3D shape·대사 안정성 |
| `nHeavyAtom`, `nHetero` | AtomCount | 분자 크기·헤테로원자 조성 |
| `nX`, `nF`, `nCl`, `nBr`, `nI` | AtomCount | 할로겐화. 지용성·reactive metabolite·간독성 |
| `nN`, `nO`, `nS`, `nP` | AtomCount | 헤테로원자 조성. 금속 킬레이션·산화환원·대사 |
| `FCSP3`, `HybRatio` | CarbonTypes | sp3 비율. 용해도·선택성·대사 안정성 |
| `SMR` | SLogP | molar refractivity. 크기/전자 분포 보완 |
| `apol`, `bpol` | Polarizability | 전자 분포·막 상호작용 |
| `FilterItLogS` | LogS | 예측 수용해도. ADMET/노출 |
| `fMF`, `fragCpx` | Framework/FragmentComplexity | 골격 복잡도·shape |
| `Kier1`, `Kier2`, `Kier3` | KappaShapeIndex | 분자 shape/topology |

#### Tier 1 shortlist

```text
nAcid, nBase,
nHBAcc, nHBDon,
TopoPSA, TopoPSA(NO),
nRot, RotRatio,
nAromAtom, nAromBond,
nRing, nARing, nHRing, nAHRing, nFRing,
nSpiro, nBridgehead,
nHeavyAtom, nHetero, nX, nF, nCl, nBr, nI, nN, nO, nS, nP,
FCSP3, HybRatio,
SMR, apol, bpol,
FilterItLogS,
fMF, fragCpx,
Kier1, Kier2, Kier3
```

약 **40개**입니다.  
이 block은 3,072-bit fingerprint와 함께 사용해도 차원 부담이 작고, toxicity/ADMET와 연결되는 해석 가능한 축을 제공합니다.

### Tier 2 — 중간 우선순위

| Block | 추천 descriptor | 근거 |
|---|---|---|
| `BCUT` | `BCUTc-1h` ~ `BCUTi-1l` (24개) | connectivity/electronic QSAR descriptor. 24개는 부담이 작음 |
| `ExtendedTopochemicalAtom` | `ETA_alpha`, `AETA_alpha`, `ETA_shape_*`, `ETA_beta*`, `ETA_eta*` 등 45개 | 전자/위상 특성. regularization과 함께 시도 |
| `MoeType` | `LabuteASA`, `PEOE_VSA1-14`, `SlogP_VSA1-12`, `EState_VSA1-11` 등 53개 | surface area·partial charge·lipophilicity VSA. ADMET와 연관 |
| `EState` | `NsNH2`, `NaaN`, `NsOH`, `NdO`, `NssS`, `NaaO`, `NssssC`, `NsCl`, `NsBr`, `NsI` 등 | 원자 환경별 electronic state. 316개 전체는 과다 |
| `Autocorrelation` | `ATS0-4Z/m/v/c`, `ATSC0-4c`, `AATS0-4*` 등 | 606개 전체는 과다; lag 0~4의 일부 weight만 |
| `Chi` / `InformationContent` / `Kappa` | `Xp-*`, `AXp-*`, `IC*`, `Kier*` | graph topology/shape. fingerprint와 중복 가능 |
| `PathCount` / `WalkCount` / `TopologicalCharge` | `MPC*`, `MWC*`, `GGI*` | 구조 topology. 대부분 size/fingerprint와 중복 가능 |

### Tier 3 — 3D descriptor, 후순위

- `CPSA`: `PNSA*`, `PPSA*`, `FNSA*`, `FPSA*`, `WNSA*`, `WPSA*`, `TASA`, `RASA`, `RPSA`
- `MoRSE`: `Mor01` ~ `Mor32` 및 weight variants
- `PBF`
- `GeomShapeIndex`, `GeomDiameter`, `GeomRadius`, `GeomPetitjeanIndex`
- `GRAV`, `GRAVH`, `GRAVp`, `GRAVHp`
- `MOMI-X`, `MOMI-Y`, `MOMI-Z`

3D shape/surface는 hERG·수용체 매개 독성에 유용할 수 있지만, **conformer 생성 비용과 재현성 문제**가 있습니다.  
2D descriptor로 성능이 plateau에 도달한 뒤 시도하는 것이 좋습니다.

### 낮은 우선순위 / 제외

- `MW`, `AMW` — `MolWt`와 중복
- `Lipinski`, `GhoseFilter` — rule-based flag, QED와 중복 가능
- `BaryszMatrix`, `AdjacencyMatrix`, `DistanceMatrix`, `DetourMatrix`, `MolecularId` — 고차원 topological, fingerprint와 중복 가능
- `RingCount` 138개 전체 — 대표 ring count만
- `EState` 316개 전체 — selected atom type만
- `Autocorrelation` 606개 전체 — selected lag/weight만
- `Constitutional` 16개 — 원자 물성 합/평균, 대부분 중복
- `VMcGowan`, `Vabc` — volume, `MolWt`와 중복 가능
- 3D `MoRSE`/`CPSA` 전체 — conformer 비용

---

## 5. 왜 이 descriptor들이 toxicity에 유망한가

Toxicity는 단일 substructure보다 **여러 물성의 조합**으로 나타나는 경우가 많습니다.

| Toxicity 측면 | 관련 Mordred descriptor |
|---|---|
| hERG 억제 | `SMR`, `SLogP`, `nBase`, `nAromAtom`, `nARing`, `TopoPSA`, 3D shape |
| 간독성/DILI | `SLogP`, `nX`, `nAromAtom`, `nHetero`, `FilterItLogS`, `fragCpx` |
| Mutagenicity | `nAromAtom`, `nARing`, `nN`, `nO`, `nX`, 대표 ring/heteroatom |
| Phospholipidosis | `nBase`, `SLogP`, `TopoPSA`, `nRot` |
| 일반 cytotoxicity | `SLogP`, `TopoPSA`, `nRot`, `nHetero`, `FCSP3` |
| Reactive metabolite | `nX`, `nN`, `nO`, `nS`, `nAromBond` |

단, Mordred descriptor는 **toxicity-specific predictor가 아닙니다.**  
label이 특정 endpoint(hERG, DILI, mutagenicity 등)인지, 광범위한 “toxic/non-toxic”인지에 따라 우선순위가 달라집니다.  
PAINS/Brenk 같은 structural alert는 Mordred에 직접 포함되지 않으므로, 별도 RDKit `FilterCatalog`와 함께 쓰는 것이 좋습니다.

---

## 6. 추천 실험 순서

| 단계 | 추가 feature | 비교 목적 |
|---|---|---|
| 0 | 현재 fingerprint + 4 descriptor | baseline (Group CV ROC-AUC 0.836) |
| 1 | Tier 1 약 40개 | 극성·이온화·유연성·방향족·할로겐·복잡도 |
| 2 | Ring/Composition block | ring/heteroatom 조성 세분화 |
| 3 | BCUT 24 + ETA 45 | 전자/연결성 QSAR |
| 4 | 선택 EState + 선택 autocorrelation | 원자 환경/전역 전자 분포 |
| 5 | 3D CPSA/MoRSE/PBF | shape/surface, 비용 큼 |

각 단계는 **동일한 `fp_group` Group CV fold**와 여러 seed로 비교합니다.

### Feature 처리

- Mordred descriptor는 **2D부터** `ignore_3D=True`로 계산
- `NaN`/invalid descriptor는 median imputation + missing indicator
- 선형 모델은 standardization, tree 모델은 raw 사용
- constant/near-constant 제거
- 상관계수 0.95 이상 pruning
- L1/LightGBM importance로 최종 selection
- **imputation, scaling, feature selection은 CV 안에서** 수행

### 검증 지표

- ROC-AUC, PR-AUC
- F1, Precision, Recall
- **Toxic recall(label 0)** / specificity
- calibration(Brier, reliability curve)
- threshold별 confusion matrix

---

## 7. Mordred 사용 시 주의사항

- 현재 `.venv`에는 Mordred가 설치되어 있지 않습니다. 구현 단계에서 `uv pip install` 필요.
- 원본 Mordred는 오래된 패키지라 최신 RDKit/Python과 호환성 문제가 있을 수 있습니다. 필요하면 `mordredcommunity` 같은 유지보수 fork를 검토하세요.
- 2D descriptor 1,613개는 계산 시간이 꽤 걸릴 수 있습니다. `n_jobs=-1`, `ignore_3D=True`로 시작하세요.
- 3D descriptor는 RDKit conformer 생성이 필요하고, conformer 설정에 따라 값이 달라질 수 있습니다.
- Mordred 계산은 unsupervised이므로 전체 데이터에 미리 계산해도 label leakage는 아닙니다.  
  하지만 **descriptor selection, imputation, scaling은 CV 안에서** 수행해야 합니다.
- 75행 혼재 라벨(34개 stereoisomer 그룹)은 Mordred 2D descriptor로 해결되지 않습니다.  
  3D descriptor도 75행 규모에서는 효과가 불확실하므로, **라벨 감사**가 우선입니다.

---

## 8. 기대 효과

- 가장 유망한 것은 **Tier 1 physicochemical block**입니다.
- 현재 descriptor 4개 추가 효과가 +0.004 AUC였던 점을 고려하면, Mordred의 현실적인 기대 상승폭은 **AUC +0.005~0.02** 수준입니다.
- 데이터 label이 광범위하거나 noisy하면 **중립 또는 소폭 악화**도 가능합니다.
- BCUT/ETA/EState/Autocorrelation은 **feature selection을 전제로** 소폭 기여할 수 있습니다.
- 3D descriptor는 특정 endpoint(hERG 등)에서는 도움이 될 수 있지만, 전체 toxicity 분류에서는 불확실합니다.

---

## 9. 결론

1. **Mordred 1,826개를 전부 넣지 않는다.**
2. **Tier 1 약 40개**를 먼저 추가한다.
   - `TopoPSA`, `nHBAcc/nHBDon`, `nAcid/nBase`, `nRot`, `nAromAtom`, `nARing`, `nX`, `FCSP3`, `SMR`, `apol/bpol`, `FilterItLogS`, `fMF/fragCpx`
3. **BCUT 24 + ETA 45**를 두 번째 block으로 실험한다.
4. **EState/Autocorrelation**은 selected feature만 사용한다.
5. **3D descriptor**는 마지막에 검토한다.
6. 모든 비교는 **fp_group 기준 Group CV**와 반복 seed로 수행한다.

---

## 10. 관련 문서

- [분자 독성 EDA 개념 Q&A](toxicity_concepts_qa.md)
- [입체화학 Fingerprint와 혼재 라벨 분석](toxicity_stereochemistry_followup.md)
- [Toxicity 분류 성능 개선 제안](toxicity_modeling_improvement_plan.md)

## 11. 참고

- Mordred Descriptor List: https://mordred-descriptor.github.io/documentation/master/descriptors.html
- Mordred documentation: https://mordred-descriptor.github.io/documentation/master/
- RDKit Documentation: https://www.rdkit.org/docs/
