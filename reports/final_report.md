# hERG blocker 예측 ML 실험 최종 보고서

> 양성 클래스 = hERG blocker (`label = 0`), 음성 클래스 = non-blocker (`label = 1`)

> 예측 종말점은 **hERG 차단 활성**이다. 임상적 QT 연장이나 부정맥을 직접 예측한다고 표현하지 않는다.

## 0. 실행 요약

- **config**: D:\git\skala-data-mini-project\configs\experiments.yaml
- **budget mode**: quick
- **primary split seed**: 42
- **실험 수(설정 조합)**: 24
- **학습된 (실험, 모델) 조합**: 50
- **모델 하이퍼파라미터 탐색**: 5 trials / 모델 (validation balanced_accuracy 기준)
- **bootstrap**: 1000회 cluster bootstrap (단위: scaffold)

## 1. 데이터

- **rows**: 8,349
- **blocker (label=0)**: 3,807
- **non-blocker (label=1)**: 4,542
- **RDKit 파싱 성공**: 8,349

분할 결과(실제 비율은 목표 비율과 다를 수 있으며 그대로 보고한다):

| split | n_rows | fraction | n_blocker | n_non_blocker | blocker_fraction | n_unique_scaffolds | strategy |
|---|---|---|---|---|---|---|---|
| train | 6,679 | 0.8000 | 3,080 | 3,599 | 0.4611 | 2,490 | scaffold |
| val | 835 | 0.1000 | 346 | 489 | 0.4144 | 835 | scaffold |
| test | 835 | 0.1000 | 381 | 454 | 0.4563 | 835 | scaffold |
| train | 6,679 | 0.8000 | 3,046 | 3,633 | 0.4561 | 3,539 | random |
| val | 835 | 0.1000 | 381 | 454 | 0.4563 | 685 | random |
| test | 835 | 0.1000 | 380 | 455 | 0.4551 | 686 | random |

- **scaffold 누수 (train∩val / train∩test / val∩test)**: 0 / 0 / 0

## 2. 실험 설정 (E0–E11 및 추가 실험)

| id | name | description | blocks | group |
|---|---|---|---|---|
| A_aromaticity_shape | ablation: + aromaticity_shape | best fingerprint baseline plus the single descriptor group 'aromaticity_shape' | fp:fcfp + existing_descriptors + aromaticity_shape | ablation |
| A_basic_amine | ablation: + basic_amine | best fingerprint baseline plus the single descriptor group 'basic_amine' | fp:fcfp + existing_descriptors + basic_amine | ablation |
| A_charge | ablation: + charge | best fingerprint baseline plus the single descriptor group 'charge' | fp:fcfp + existing_descriptors + charge | ablation |
| A_lipophilicity | ablation: + lipophilicity | best fingerprint baseline plus the single descriptor group 'lipophilicity' | fp:fcfp + existing_descriptors + lipophilicity | ablation |
| A_physicochemical | ablation: + physicochemical | best fingerprint baseline plus the single descriptor group 'physicochemical' | fp:fcfp + existing_descriptors + physicochemical | ablation |
| A_polarity_hbond | ablation: + polarity_hbond | best fingerprint baseline plus the single descriptor group 'polarity_hbond' | fp:fcfp + existing_descriptors + polarity_hbond | ablation |
| A_surface_electronic | ablation: + surface_electronic | best fingerprint baseline plus the single descriptor group 'surface_electronic' | fp:fcfp + existing_descriptors + surface_electronic | ablation |
| E0 | ecfp baseline | baseline: shipped ECFP bits + shipped descriptors | fp:ecfp + existing_descriptors | main |
| E0b | ecfp only | ECFP bits without any descriptor | fp:ecfp | fingerprint_only |
| E1 | fcfp baseline | fingerprint family comparison: FCFP | fp:fcfp + existing_descriptors | main |
| E10 | best fp + rdkit + selection | E7 feature set with train-only feature selection | fp:fcfp + existing_descriptors + rdkit_all | main |
| E11 | best fp + rdkit + selection (all models) | identical features to E10, expanded to every available model family | fp:fcfp + existing_descriptors + rdkit_all | model_comparison |
| E1b | fcfp only | FCFP bits without any descriptor | fp:fcfp | fingerprint_only |
| E2 | ptfp baseline | fingerprint family comparison: PTFP (pattern) | fp:ptfp + existing_descriptors | main |
| E2b | ptfp only | PTFP bits without any descriptor | fp:ptfp | fingerprint_only |
| E3 | best fingerprint baseline | validation-selected single fingerprint + shipped descriptors | fp:fcfp + existing_descriptors | main |
| E3c | combined fingerprints | ECFP+FCFP+PTFP concatenated + shipped descriptors | all_fingerprints + existing_descriptors | fingerprint_combo |
| E4 | rdkit descriptors only | no fingerprint at all: full RDKit 2D descriptor set | rdkit_all | main |
| E5 | best fp + physicochemical | general property descriptors on top of the best fingerprint | fp:fcfp + existing_descriptors + physicochemical | main |
| E6 | best fp + herg domain descriptors | physicochemical + all six hERG-motivated descriptor groups on top of the best fingerprint | fp:fcfp + existing_descriptors + physicochemical + lipophilicity + polarity_hbond + basic_amine + charge + aromaticity_shape + surface_electronic | main |
| E7 | best fp + all rdkit descriptors | full RDKit 2D descriptor expansion (domain groups + misc) | fp:fcfp + existing_descriptors + rdkit_all | main |
| E8 | best fp + mordred 2d | Mordred 2D descriptor expansion (unselected) | fp:fcfp + existing_descriptors + mordred_2d | main |
| E8b | best fp + selected mordred | Mordred expansion with train-only feature selection | fp:fcfp + existing_descriptors + mordred_2d | mordred |
| E9 | best fp + rdkit + mordred | RDKit and Mordred descriptors combined | fp:fcfp + existing_descriptors + rdkit_all + mordred_2d | main |

모든 실험은 **동일한 분할**, **동일한 탐색 예산**, **동일한 학습 내부 전처리**를 사용한다. 하이퍼파라미터 탐색과 임계값 선택은 validation split에서만 수행하고, test split은 최종 모델 확정 후 1회만 평가한다(§7).

## 3. Fingerprint 계열 비교 (Q1)

| experiment_id | model | tuned_balanced_accuracy | tuned_roc_auc | tuned_pr_auc |
|---|---|---|---|---|
| E0 | lightgbm | 0.7397 | 0.8242 | 0.7852 |
| E0 | logistic_regression | 0.6795 | 0.7421 | 0.6658 |
| E0 | random_forest | 0.7514 | 0.8166 | 0.7859 |
| E0b | lightgbm | 0.7243 | 0.7943 | 0.7553 |
| E1 | lightgbm | 0.7653 | 0.8354 | 0.7955 |
| E1 | logistic_regression | 0.6966 | 0.7420 | 0.6347 |
| E1 | random_forest | 0.7534 | 0.8357 | 0.7998 |
| E1b | lightgbm | 0.7257 | 0.8030 | 0.7596 |
| E2 | lightgbm | 0.7236 | 0.8145 | 0.7794 |
| E2 | logistic_regression | 0.6875 | 0.7557 | 0.6624 |
| E2 | random_forest | 0.7417 | 0.8176 | 0.7807 |
| E2b | lightgbm | 0.7285 | 0.8007 | 0.7690 |
| E3c | lightgbm | 0.7562 | 0.8411 | 0.8060 |

→ validation 기준 최고 fingerprint 구성은 **fcfp** 이며, 이후 모든 `best fp + …` 실험은 해당 fingerprint를 사용한다.

## 4. 기준선과 descriptor 확장 비교 (Q2, Q3, Q4)

| 실험 | validation BA | ROC-AUC | 모델 | features |
|---|---|---|---|---|
| E3 fingerprint baseline | 0.7653 | 0.8354 | lightgbm | 1,027 |
| E4 RDKit descriptors only | 0.7609 | 0.8337 | random_forest | 227 |
| E5 + physicochemical | 0.7652 | 0.8453 | random_forest | 1,046 |
| E6 + hERG domain groups | 0.7799 | 0.8467 | lightgbm | 1,223 |
| E7 + all RDKit descriptors | 0.7778 | 0.8483 | lightgbm | 1,253 |
| E8 + Mordred 2D (전체) | 0.7685 | 0.8419 | lightgbm | 2,306 |
| E8b + Mordred (선별) | 0.7644 | 0.8365 | lightgbm | 300 |
| E9 + RDKit + Mordred | 0.7625 | 0.8223 | random_forest | 2,502 |
| E10 + RDKit + feature selection | 0.7598 | 0.8438 | lightgbm | 300 |
| E11 = E10 피처, 전체 모델 | 0.7598 | 0.8438 | lightgbm | 300 |

각 행은 해당 실험에서 validation BA가 가장 높았던 **단일 모델**의 값이다(모델별 전체 값은 `reports/experiment_results.csv`). E0b/E1b/E2b와 `A_*` ablation은 예산상 lightgbm 단일 모델로만 실행했고, 나머지는 quick 모드의 3개 모델을 모두 탐색했다.

## 5. Descriptor 그룹별 ablation

| experiment_id | label | model | baseline_balanced_accuracy | ablated_balanced_accuracy | delta_balanced_accuracy | baseline_roc_auc | ablated_roc_auc | delta_roc_auc |
|---|---|---|---|---|---|---|---|---|
| A_aromaticity_shape | E3 + aromaticity_shape | lightgbm | 0.7653 | 0.7690 | 0.0037 | 0.8354 | 0.8369 | 0.0015 |
| A_lipophilicity | E3 + lipophilicity | lightgbm | 0.7653 | 0.7683 | 0.0030 | 0.8354 | 0.8411 | 0.0057 |
| A_surface_electronic | E3 + surface_electronic | lightgbm | 0.7653 | 0.7678 | 0.0025 | 0.8354 | 0.8436 | 0.0082 |
| A_physicochemical | E3 + physicochemical | lightgbm | 0.7653 | 0.7650 | -0.0002 | 0.8354 | 0.8420 | 0.0066 |
| A_basic_amine | E3 + basic_amine | lightgbm | 0.7653 | 0.7628 | -0.0025 | 0.8354 | 0.8346 | -0.0008 |
| A_polarity_hbond | E3 + polarity_hbond | lightgbm | 0.7653 | 0.7623 | -0.0030 | 0.8354 | 0.8475 | 0.0121 |
| A_charge | E3 + charge | lightgbm | 0.7653 | 0.7560 | -0.0093 | 0.8354 | 0.8376 | 0.0022 |

`delta_balanced_accuracy`가 0 근처면 해당 그룹이 fingerprint 기준선에 추가 정보를 주지 못한 것이다.

## 6. 모델 비교 (Q5)

| model | tuned_balanced_accuracy | tuned_roc_auc | tuned_pr_auc | best_fit_seconds |
|---|---|---|---|---|
| lightgbm | 0.7598 | 0.8438 | 0.8129 | 3.3000 |
| logistic_regression | 0.7093 | 0.7719 | 0.6903 | 0.8500 |
| random_forest | 0.7557 | 0.8367 | 0.7993 | 0.8900 |

## 7. 최종 모델과 test 성능

- **선택된 실험/모델**: E6 / lightgbm
- **선택 근거**: validation tuned_balanced_accuracy 최대 (test는 사용하지 않음)
- **임계값(validation 선택)**: 0.7139
- **test rows**: 835 (blocker 381)

| metric | value |
|---|---|
| threshold | 0.7139 |
| n | 835 |
| n_blocker | 381 |
| n_non_blocker | 454 |
| roc_auc | 0.8475 |
| pr_auc | 0.8257 |
| balanced_accuracy | 0.7320 |
| accuracy | 0.7473 |
| precision | 0.8346 |
| recall | 0.5564 |
| f1 | 0.6677 |
| mcc | 0.5023 |
| specificity | 0.9075 |
| npv | 0.7091 |
| tn | 412 |
| fp | 42 |
| fn | 169 |
| tp | 212 |

- **confusion matrix (blocker = positive)**: TP 212, FP 42, FN 169, TN 412
- **threshold 0.5 결과**: BA 0.7614, ROC-AUC 0.8475

최종 모델은 train+val로 재적합했고 임계값은 validation에서 선택했다(train+val). test는 이 시점에 처음 평가되었다.

**임계값 전이 실패 가능성:** validation에서 고른 임계값의 test BA가 기본 0.5보다 0.0294만큼 낮다. validation 835행(blocker 381행 규모)에서 임계값을 고르는 것 자체가 잡음에 민감하다는 뜻이며, 순위 지표(ROC-AUC/PR-AUC)는 validation과 test에서 거의 같다. 실제 사용 시 임계값은 별도의 보정 데이터로 정하고, 이 값은 'validation에서 고른 값'으로만 읽어야 한다.

## 7b. 참고: random split과의 비교

| experiment_id | model | tuned_balanced_accuracy_scaffold | roc_auc_scaffold | tuned_balanced_accuracy_random | roc_auc_random | delta_ba_random_minus_scaffold |
|---|---|---|---|---|---|---|
| E3 | lightgbm | 0.7653 | 0.8354 | 0.8146 | 0.8859 | 0.0493 |
| E3 | logistic_regression | 0.6966 | 0.7420 | 0.7649 | 0.8278 | 0.0683 |
| E3 | random_forest | 0.7534 | 0.8357 | 0.8040 | 0.8783 | 0.0506 |

random split은 참고용이다. 같은 scaffold가 train/val/test에 걸쳐 있어(데이터 품질 보고서 §7) 성능이 낙관적으로 나오며, 위 차이는 **일반화 격차의 크기**를 보여줄 뿐 어느 쪽이 더 좋은 모델인지를 말하지 않는다. 최종 모델 선택에는 scaffold split만 썼다.

## 8. Scaffold 분할 반복 안정성 (Q6)

상위 설정을 **scaffold split seed 3개**(42 = 위 표들의 기본 seed, 101, 2024)에서 다시 학습한 결과다. `count`가 3이면 세 seed 모두 실행된 것이고, `std`가 seed 간 변동성이다. seed 하나에서의 순위 차이를 과대해석하지 않기 위해 사용한다.

| experiment_id | model | mean | std | count |
|---|---|---|---|---|
| A_aromaticity_shape | lightgbm | 0.7585 | 0.0097 | 3 |
| A_lipophilicity | lightgbm | 0.7650 | 0.0044 | 3 |
| A_surface_electronic | lightgbm | 0.7657 | 0.0061 | 3 |
| E3 | lightgbm | 0.7576 | 0.0100 | 3 |

## 9. 중요 피처와 오분류 패턴 (Q7)

| feature | importance_mean | importance_std | source |
|---|---|---|---|
| fcfp_926 | 0.0390 | 0.0082 | permutation |
| TPSA | 0.0358 | 0.0023 | permutation |
| VSA_EState2 | 0.0310 | 0.0060 | permutation |
| NumRotatableBonds | 0.0294 | 0.0011 | permutation |
| VSA_EState6 | 0.0294 | 0.0011 | permutation |
| SMR_VSA4 | 0.0278 | 0.0020 | permutation |
| PEOE_VSA6 | 0.0278 | 0.0039 | permutation |
| VSA_EState5 | 0.0270 | 0.0011 | permutation |
| clogp | 0.0270 | 0.0011 | permutation |
| Kappa3 | 0.0270 | 0.0023 | permutation |
| BCUT2D_MWHI | 0.0262 | 0.0045 | permutation |
| fcfp_424 | 0.0262 | 0.0011 | permutation |
| EState_VSA2 | 0.0262 | 0.0011 | permutation |
| BCUT2D_LOGPLOW | 0.0262 | 0.0011 | permutation |
| fcfp_825 | 0.0262 | 0.0011 | permutation |
| PEOE_VSA8 | 0.0262 | 0.0011 | permutation |
| sa_score | 0.0262 | 0.0011 | permutation |
| FpDensityMorgan3 | 0.0262 | 0.0041 | permutation |
| MinAbsEStateIndex | 0.0254 | 0.0020 | permutation |
| MaxPartialCharge | 0.0254 | 0.0020 | permutation |
| SMR_VSA9 | 0.0254 | 0.0020 | permutation |
| SMR_VSA6 | 0.0254 | 0.0020 | permutation |
| VSA_EState1 | 0.0254 | 0.0020 | permutation |
| EState_VSA7 | 0.0254 | 0.0020 | permutation |
| SMR_VSA7 | 0.0254 | 0.0034 | permutation |

그룹 단위 중요도:

| group | n_features | total_importance | mean_importance | max_importance |
|---|---|---|---|---|
| fingerprint_fcfp | 1,023 | 18.7976 | 0.0184 | 0.0390 |
| basic_amine | 49 | 0.8974 | 0.0183 | 0.0214 |
| aromaticity_shape | 43 | 0.8542 | 0.0199 | 0.0294 |
| surface_electronic | 36 | 0.7968 | 0.0221 | 0.0310 |
| lipophilicity | 22 | 0.4750 | 0.0216 | 0.0278 |
| polarity_hbond | 19 | 0.4171 | 0.0220 | 0.0358 |
| physicochemical | 21 | 0.3928 | 0.0187 | 0.0222 |
| charge | 8 | 0.1620 | 0.0202 | 0.0254 |
| existing_descriptor | 2 | 0.0533 | 0.0266 | 0.0270 |

오분류 요약(validation):

| error_type | n | mean_probability |
|---|---|---|
| correct | 674 | 0.4436 |
| false_negative | 105 | 0.2772 |
| false_positive | 56 | 0.8802 |

오분류가 많은 chemotype (validation, element-stripped generic scaffold 계열, 3개 이상):

| scaffold_id | n_molecules | n_blocker | blocker_fraction | mean_score | accuracy | balanced_accuracy | n_errors | error_rate |
|---|---|---|---|---|---|---|---|---|
| S1 | 3 | 3 | 1.0000 | 0.1392 | 0.0000 | n/a | 3 | 1.0000 |
| S2 | 3 | 1 | 0.3333 | 0.3631 | 0.3333 | 0.2500 | 2 | 0.6667 |
| S3 | 3 | 3 | 1.0000 | 0.6493 | 0.3333 | n/a | 2 | 0.6667 |
| S4 | 3 | 3 | 1.0000 | 0.3290 | 0.3333 | n/a | 2 | 0.6667 |
| S5 | 3 | 2 | 0.6667 | 0.2272 | 0.3333 | 0.5000 | 2 | 0.6667 |
| S6 | 3 | 1 | 0.3333 | 0.9858 | 0.3333 | 0.5000 | 2 | 0.6667 |
| S7 | 3 | 2 | 0.6667 | 0.2840 | 0.3333 | 0.5000 | 2 | 0.6667 |
| S8 | 3 | 2 | 0.6667 | 0.1005 | 0.3333 | 0.5000 | 2 | 0.6667 |
| S9 | 4 | 2 | 0.5000 | 0.9425 | 0.5000 | 0.5000 | 2 | 0.5000 |
| S10 | 4 | 1 | 0.2500 | 0.4916 | 0.5000 | 0.3333 | 2 | 0.5000 |
| S11 | 11 | 5 | 0.4545 | 0.3298 | 0.8182 | 0.8000 | 2 | 0.1818 |
| S12 | 3 | 1 | 0.3333 | 0.2797 | 0.6667 | 0.5000 | 1 | 0.3333 |
| S13 | 3 | 2 | 0.6667 | 0.8905 | 0.6667 | 0.5000 | 1 | 0.3333 |
| S14 | 4 | 1 | 0.2500 | 0.0623 | 0.7500 | 0.5000 | 1 | 0.2500 |
| S15 | 4 | 2 | 0.5000 | 0.6481 | 0.7500 | 0.7500 | 1 | 0.2500 |

scaffold 구조 문자열은 원본 구조의 재배포 제한 때문에 보고서에 넣지 않았다(`reports/scaffold_performance.csv`, 로컬 전용). scaffold split에서는 validation 분자의 Bemis–Murcko scaffold가 모두 서로 다르므로(설계상 그렇다), scaffold 단위 성능은 원소를 제거한 generic scaffold 계열로 묶어서만 계산할 수 있다. 이 표는 '어떤 chemotype 계열에서 오류가 몰리는가'를 보여준다.

## 10. 클래스 불균형 처리

| experiment_id | model | class_weight | threshold_source | threshold | balanced_accuracy | precision | recall | f1 | mcc | roc_auc | pr_auc | n_features |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E6 | lightgbm | n/a | 0.5 | 0.5000 | 0.7799 | 0.7185 | 0.7746 | 0.7455 | 0.5547 | 0.8467 | 0.8133 | 1,223 |
| E6 | lightgbm | n/a | validation-selected | 0.7139 | 0.7910 | 0.8114 | 0.6965 | 0.7496 | 0.5989 | 0.8467 | 0.8133 | 1,223 |
| E6 | lightgbm | balanced | 0.5 | 0.5000 | 0.7750 | 0.7131 | 0.7688 | 0.7399 | 0.5450 | 0.8462 | 0.8148 | 1,223 |
| E6 | lightgbm | balanced | validation-selected | 0.4667 | 0.7785 | 0.7083 | 0.7861 | 0.7452 | 0.5506 | 0.8462 | 0.8148 | 1,223 |

## 11. 한계와 실제 활용 시 주의점 (Q8)

- 라벨이 대략 `IC50 < 10 µM` 기준으로 만들어졌을 가능성이 있으나, 모든 라벨이 단일 assay·단일 규칙으로 생성되었다고 단정할 수 없다. 라벨 노이즈가 남아 있다.
- validation/test 성능은 모두 Bemis-Murcko scaffold split 기준이며, 학습에 등장하지 않은 chemotype에 대한 성능이다. 따라서 random split 성능과 직접 비교해서는 안 된다.
- 확률값은 저장된 모델이 산출한 값이며, 보정(calibration)과 외부 검증 없이 임상적 독성 확률로 해석해서는 안 된다.
- fingerprint 기반 모델은 학습 데이터가 커버하는 화학 공간 밖에서 급격히 성능이 떨어진다. 적용 전에 applicability domain 확인이 필요하다.
- feature importance는 상관관계 수준의 설명이며 작용 기전의 증거가 아니다.

### 이 실행에서 추가로 확인된 한계

- ablation과 모델 비교는 기본 임계값 0.5의 validation balanced accuracy 기준이다.
- 이 실행은 mode=quick으로 모델 3종(logistic_regression, random_forest, lightgbm)만 탐색했다. extra_trees, hist_gradient_boosting, xgboost, svm는 실행하지 않았으므로 '모델 계열 간 비교'는 이 부분집합 안에서만 유효하다.
- 최종 모델은 train+val로 전처리를 다시 적합하므로, 학습 split 기준 실험 (1,223개)보다 feature가 +1개 다르다(최종 1,224개). 상수/중복 열 판정이 재적합 시점에 달라지기 때문이다.
