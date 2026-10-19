# SKALA 데이터분석 미니 프로젝트 — 10반 3조

신약 후보물질의 **독성(toxicity)** 을 예측하기 위한 EDA 및 피처 설계 프로젝트입니다.  
SMILES, ECFP/FCFP/PTFP fingerprint, 물성 descriptor를 사용해 “어떤 구조적 신호가 독성과 연결되는가”를 분석하고, 이를 바탕으로 파생변수 후보를 규정합니다.

> **현재 단계의 초점:** 모델 성능 자체보다 **EDA → 도메인 해석 → 파생변수 규정**입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 문제 | 화합물의 독성 여부 이진 분류 |
| 데이터 | 8,349개 분자 × 3,078개 컬럼 |
| 입력 | SMILES 1개 + ECFP/FCFP/PTFP 각 1,024-bit + 물성 4개 |
| 라벨 | `1 = 독성 없음(non-toxic, Positive)`, `0 = 독성 있음(toxic, Negative)` |
| 평가지표 | F1 (Positive = 1 기준), 보조로 toxic recall / specificity |

---

## 2. 핵심 EDA 결과

- **라벨 방향:** `1 = 독성 없음`이다. 반대로 해석하면 clogP·SA score의 방향이 모두 뒤집힌다.
- **제공 물성 4개의 한계:** MolWt는 거의 무관(AUC 0.495), SA score/QED는 약함. clogP가 가장 강한 단일 신호다.
- **clogP × TPSA:** 높은 clogP + 낮은 TPSA 구간의 독성률 **76.4%** (전체 45.6%).
- **염기성 아민/CAD:** CAD 해당 62.4% vs 미해당 30.6%. hERG/CAD 도메인 가설과 연결된다.
- **fingerprint:** 활성 bit 수 자체는 label과 거의 무관(상관 ≈ 0.02). **어떤 bit가 켜지는지**가 중요하다.
- **표현의 한계:** 2D fingerprint는 chirality를 반영하지 않아, 같은 fingerprint인데 label이 반대인 입체이성질체 그룹이 존재한다.
- **중복/누수:** 동일 fingerprint 그룹과 scaffold 중복이 있어 무작위 분할은 성능을 약 0.06 부풀릴 수 있다.
- **canonical SMILES:** 같은 분자를 하나의 문자열로 통일해야 중복 탐지·group split·train/predict featurization이 일관된다.

---

## 3. 저장소 구조

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md                  # 원본 데이터는 저장소에 포함하지 않음
├── toxicity_eda.ipynb             # 기본 EDA
├── team/
│   ├── 신약_독성예측_도메인EDA.ipynb
│   ├── 독성예측_팀공유_노트북.ipynb
│   ├── EDA_결과_요약.md
│   ├── 독성예측_팀공유_요약.md
│   ├── 데이터 분석 미니 프로젝트_3조.pdf
│   └── 독성 예측 EDA 정리 (팀 공유용).pdf
├── presentation_assets/           # 발표용 차트 (v1)
├── presentation_assets_v2/        # 발표용 차트 (v2)
├── toxicity_concepts_qa.md
├── toxicity_mordred_feature_proposal.md
├── toxicity_modeling_improvement_plan.md
├── toxicity_stereochemistry_followup.md
└── toxicity_chiral_embedding_geometry.md
```

---

## 4. 실행 방법

### 4-1. 데이터 준비

원본 `train.csv`는 강의 데이터 재배포 제한 및 용량(약 52MB) 문제로 저장소에 포함하지 않았습니다.  
아래 위치에 `train.csv`를 두면 노트북을 그대로 실행할 수 있습니다.

```text
skala-data-mini-project/
└── train.csv
```

`data/README.md`에 컬럼 설명을 정리해 두었습니다.

### 4-2. 환경 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 4-3. 노트북 실행

```bash
jupyter lab
```

- `toxicity_eda.ipynb` — 기본 EDA, descriptor/fingerprint 분석
- `team/신약_독성예측_도메인EDA.ipynb` — 도메인 EDA, 부분구조/작용기 분석
- `team/독성예측_팀공유_노트북.ipynb` — 피처 엔지니어링·모델링 재현

---

## 5. 데이터 및 공개 범위

이 저장소에는 아래 파일을 **포함하지 않았습니다.**

- `train.csv` — 원본 강의 데이터 (용량 및 재배포 제한)
- `other/` — 다른 조의 발표자료 (타 팀 자산)
- 강의안 PDF — 원본 강의자료
- `mixed_label_groups*.csv` — 원본 SMILES를 포함한 파생 데이터
- `.venv/`, 캐시, 개인 설정

프로젝트의 **코드·분석 문서·발표자료·차트**만 공개합니다.

> `team/EDA_결과_요약.md`의 일부 이미지 경로는 원본 작업환경 기준이며, 해당 이미지 파일은 저장소에 포함하지 않았습니다.  
> 발표자료 PDF에는 팀원 이름이 포함되어 있으므로 공개 범위를 확인해 주세요.

---

## 6. 발표자료

- `team/데이터 분석 미니 프로젝트_3조.pdf` — 최종 발표자료
- `team/독성 예측 EDA 정리 (팀 공유용).pdf` — EDA 정리본

---

## 7. 한계와 다음 단계

- 라벨의 실제 독성 endpoint(예: hERG, 급성 독성)가 데이터 설명만으로 확정되지 않는다.
- 중복/입체이성질체/염 형태 처리 기준에 따라 중복 그룹 수가 달라진다.
- 구조 클러스터·scaffold 분할 기준을 명시하고, 파생변수별 ablation과 불확실성(CI)을 함께 보고할 필요가 있다.
- 다음 단계: canonicalization 전후 비교 → 도메인 가설 정교화 → Mordred/SMARTS 파생변수 후보 → cluster CV 기반 채택.

---

## 8. 참고

- Mordred descriptor documentation: <https://mordred-descriptor.github.io/documentation/master/descriptors.html>
- RDKit documentation: <https://www.rdkit.org/docs/>

---

## 9. ML 실험 스캐폴드 (`docs/ml_goal.md` 구현)

`configs/`, `src/herg_model/`, `scripts/`, `tests/` 는 fingerprint/descriptor 표현을
동일한 scaffold split에서 통제 비교하기 위한 재현 가능한 ML 스캐폴드입니다.

- 실행·설정·산출물 문서: **[`HERG_ML.md`](HERG_ML.md)**
- 데이터 품질: [`reports/data_quality_report.md`](reports/data_quality_report.md)
- descriptor 분석: [`reports/descriptor_report.md`](reports/descriptor_report.md)
- 최종 결과: [`reports/final_report.md`](reports/final_report.md)

> 이 스캐폴드는 `docs/ml_goal.md`의 고정 라벨 정의에 따라 **`label = 0`(hERG blocker)을
> 양성 클래스**로 사용합니다. 위 2절 EDA가 사용한 `label = 1` 기준 F1과는 positive 클래스가
> 반대이므로 지표를 직접 비교하면 안 됩니다.

