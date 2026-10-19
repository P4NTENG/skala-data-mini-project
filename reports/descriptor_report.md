# Descriptor 정의 · 계산 성공률 · 결측/중복 분석

> 양성 클래스 = hERG blocker (`label = 0`), 음성 클래스 = non-blocker (`label = 1`)

## 1. 계산 엔진과 성공률

- **RDKit descriptor 수**: 251
- **RDKit 계산 성공**: 8349 / 8349
- **RDKit 계산 시간(초)**: 0.0
- **Mordred 사용 가능**: True
- **Mordred descriptor 수**: 1613
- **Mordred 실패/미설치 사유**: 해당 없음

3D 좌표가 필요한 descriptor는 제외했다(데이터셋에 conformer가 없음). RDKit `CalcMolDescriptors`는 2D descriptor만 반환하며, 여기에 hERG 도메인 해석을 위한 파생 feature(원자 조성, 염기성/비염기성 N 환경, 전하 관련 34개)를 추가했다. 따라서 `rdkit_all`은 `CalcMolDescriptors` 217개 + 파생 34개 = 251개다.

## 2. 그룹별 descriptor 수

| group | n_descriptors |
|---|---|
| basic_amine | 63 |
| aromaticity_shape | 44 |
| surface_electronic | 37 |
| misc_rdkit | 35 |
| lipophilicity | 24 |
| physicochemical | 20 |
| polarity_hbond | 20 |
| charge | 8 |
| rdkit_all | 251 |

그룹은 서로 겹치지 않도록 배타적으로 배정했다. 어떤 그룹에도 명시적으로 배정되지 않은 descriptor는 `misc_rdkit`으로 모아 보고한다. `rdkit_all`은 전체 RDKit descriptor의 합집합이다.

## 3. 결측치 · 상수 · 중복

- **결측률 > 0 인 feature 수**: 8
- **상수 feature 수(학습 split 기준)**: 13
- **학습 split에서 제거된 feature**: {"all_missing": 0, "high_missing": 0, "constant": 13, "duplicate": 11, "scaffold_leak": 0}

- **상수(학습 split)** (13): `NumRadicalElectrons`, `SMR_VSA8`, `SlogP_VSA9`, `fr_azo`, `fr_diazo`, `fr_epoxide`, `fr_isocyan`, `fr_isothiocyan`, `fr_nitroso`, `fr_phos_acid`, `fr_phos_ester`, `fr_thiocyan`, `n_imine_oxime`
- **중복 열** (11) (표기: 제거될 열 ← 유지되는 열): `MaxEStateIndex`←`MaxAbsEStateIndex`, `fr_Nhpyrrole`←`fr_Ar_NH`, `fr_amide`←`NumAmideBonds`, `fr_phenol_noOrthoHbond`←`fr_phenol`, `n_halogen`←`fr_halogen`, `n_hetero`←`NumHeteroatoms`, `n_total_n`←`n_N`, `n_nitro`←`fr_nitro`, `n_nitrile`←`fr_nitrile`, `n_azo`←`fr_azide`, `n_aromatic_n`←`fr_Ar_N`

결측치 대체(중앙값)와 스케일링은 **학습 split에서만** 적합한다. 결측률이 설정값(`descriptors.mordred.max_missing_rate`)을 넘는 Mordred descriptor는 학습 split에서 제거된다.

### 결측률 상위 feature

| feature | missing_rate | n_unique | is_constant |
|---|---|---|---|
| BCUT2D_MRHI | 0.0010 | 6,385 | 0 |
| BCUT2D_MRLOW | 0.0010 | 6,385 | 0 |
| BCUT2D_MWHI | 0.0010 | 6,384 | 0 |
| BCUT2D_MWLOW | 0.0010 | 6,377 | 0 |
| BCUT2D_CHGHI | 0.0010 | 6,385 | 0 |
| BCUT2D_CHGLO | 0.0010 | 6,385 | 0 |
| BCUT2D_LOGPHI | 0.0010 | 6,385 | 0 |
| BCUT2D_LOGPLOW | 0.0010 | 6,385 | 0 |
| MaxAbsEStateIndex | 0.0000 | 6,302 | 0 |
| MaxEStateIndex | 0.0000 | 6,302 | 0 |
| MinAbsEStateIndex | 0.0000 | 6,181 | 0 |
| NumRadicalElectrons | 0.0000 | 1 | 1 |
| NumValenceElectrons | 0.0000 | 124 | 0 |
| ExactMolWt | 0.0000 | 5,134 | 0 |
| HeavyAtomMolWt | 0.0000 | 4,246 | 0 |

## 4. 중복 계산 여부

기존 `MolWt`, `clogp`, `qed`는 RDKit `Descriptors.MolWt`, `Crippen.MolLogP`, `QED.qed`와 사실상 같은 값이다(재계산 오차는 데이터 품질 보고서 §9 참조). RDKit descriptor와 함께 쓰이는 실험에서는 학습 split에서 값이 완전히 동일한 열을 자동으로 제거하며, 제거된 열 이름은 위 `중복 열` 목록에 그대로 남긴다.

## 5. 고상관 descriptor 쌍

|Pearson r| ≥ 0.95 인 쌍: 99개

| feature_a | feature_b | pearson_r |
|---|---|---|
| MaxAbsEStateIndex | MaxEStateIndex | 1.0000 |
| MolWt | HeavyAtomMolWt | 0.9978 |
| MolWt | ExactMolWt | 1.0000 |
| MolWt | NumValenceElectrons | 0.9784 |
| MolWt | Chi0 | 0.9761 |
| MolWt | Chi0v | 0.9647 |
| MolWt | Chi1 | 0.9622 |
| MolWt | Kappa1 | 0.9601 |
| MolWt | LabuteASA | 0.9819 |
| MolWt | HeavyAtomCount | 0.9740 |
| HeavyAtomMolWt | ExactMolWt | 0.9977 |
| HeavyAtomMolWt | NumValenceElectrons | 0.9672 |
| HeavyAtomMolWt | Chi0 | 0.9701 |
| HeavyAtomMolWt | Chi1 | 0.9550 |
| HeavyAtomMolWt | LabuteASA | 0.9722 |
| HeavyAtomMolWt | HeavyAtomCount | 0.9677 |
| ExactMolWt | NumValenceElectrons | 0.9790 |
| ExactMolWt | Chi0 | 0.9767 |
| ExactMolWt | Chi0v | 0.9648 |
| ExactMolWt | Chi1 | 0.9629 |
