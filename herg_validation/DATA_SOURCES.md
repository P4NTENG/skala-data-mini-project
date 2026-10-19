# 데이터 출처 · 라이선스 · 인용

작성 기준일: 2026-10-20 (모든 외부 데이터 조회일 = 아래 표 `조회일`)

이 문서는 `validation.py --stage fetch`가 실제로 내려받은 원자료의 출처를 기록한다.
다운로드 원본은 `herg_validation/cache/`에 있으며 (저장소에는 커밋하지 않음), 아래 URL로 재현 가능하다.

## 1. 대상 데이터셋

| 항목 | 내용 |
|---|---|
| 파일 | `data/train.csv` (저장소 미포함, 강의 데이터 재배포 제한) |
| 행/열 | 8,349행 × 3,078열 |
| 라벨 정의 (고정) | `label = 0`: 독성 (hERG 차단 활성 기대) / `label = 1`: 비독성 |
| 사용 컬럼 | `SMILES`, `MolWt`, `clogp`, `sa_score`, `qed`, `label` |
| 지문 컬럼 | `ecfp_*`(1024), `fcfp_*`(1024), `ptfp_*`(1024) — 구조 검색 보조 수단으로만 사용 |

## 2. 외부 실험 데이터

| 소스 | 접근 경로 | 내용 | 라이선스 | 조회일 |
|---|---|---|---|---|
| ChEMBL (hERG target CHEMBL240) | `https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL240` (limit=1000 페이지네이션), `.../assay.json?target_chembl_id=CHEMBL240` | hERG(KCNH2, Homo sapiens) activity 전량 41,078건 / assay 4,829건 | CC BY-SA 3.0 (ChEMBL 데이터 이용 조건) | 2026-10-20 |
| PubChem BioAssay AID588834 | `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/588834/CSV?response_type=display`, `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<cids>/property/InChIKey,SMILES,.../JSON` | Tox21/NCGC qHTS hERG thallium flux (U2OS, FluxOR) 5,384 레코드 | PubChem = 미국 정부 공공데이터(제한 없음) | 2026-10-20 |
| ChEMBL 멀티타깃 activity (종말점 탐색) | `.../activity.json?molecule_chembl_id__in=<ids>&pchembl_value__isnull=false` | 매칭된 화합물의 전 타깃 활성 (hERG 외 종말점 대조) | CC BY-SA 3.0 | 2026-10-20 |
| Tox21 12-endpoint 패널 | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz` (MoleculeNet 배포본) | NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-γ, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53 | 원 데이터: Tox21 (NTP/NCATS/EPA/FDA) — 공공 | 2026-10-20 |
| ClinTox | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz` (MoleculeNet 배포본) | 임상시험 독성 실패(`CT_TOX`), FDA 승인(`FDA_APPROVED`) | 원 데이터: FDA 공공 | 2026-10-20 |
| 논문 보충자료 (Krishna et al. 2022) | MDPI/PMC 보충파일 경로 — 실제 시도 결과는 §3 참조 | Tox21 + ChEMBL 통합 hERG 활성값 (S2/S4) | CC BY 4.0 (Biology, MDPI) | 2026-10-20 |

### 인용 지침 (따름)

- ChEMBL: Davies M. et al. *ChEMBL web services: streamlining access to drug discovery data and utilities.* Nucleic Acids Res. 2015;43:W612–W620. 데이터 라이선스 CC BY-SA 3.0.
- PubChem: Kim S. et al. *PubChem 2023 update.* Nucleic Acids Res. 2023;51:D1373–D1380.
- Tox21 / MoleculeNet: Wu Z. et al. *MoleculeNet: a benchmark for molecular machine learning.* Chem. Sci. 2018;9:513–530. Tox21 원 시험: Attene-Ramos M.S. et al. (NCATS/NTP/EPA/FDA).
- ClinTox: Gayvert K.M. et al. *A data-driven approach to predicting successes and failures of clinical trials.* Cell Chem Biol. 2016;23:1294–1301.
- 논문 (data/README.md 링크 1): Krishna S., Borrel A., Huang R., Zhao J., Xia M., Kleinstreuer N. *High-Throughput Chemical Screening and Structure-Based Models to Predict hERG Inhibition.* Biology 2022;11(2):209. doi:10.3390/biology11020209 (PMC8869358).
- 논문 (data/README.md 링크 2): Didziapetris R., Lanevskij K. *Compilation and physicochemical classification analysis of a diverse hERG inhibition database.* J Comput Aided Mol Des. 2016;30:1175–1188. doi:10.1007/s10822-016-9986-0 (보충 데이터 비공개 — 본 검증에서는 사용하지 못함).

## 3. 다운로드 시도 결과 (자동 기록)

자세한 시도 로그: `herg_validation/cache/supplement_attempts.json`, `herg_validation/cache/alt_datasets_log.json`.

### 3.1 논문 보충자료 (Krishna et al. 2022, PMC8869358)

| 시도 경로 | 결과 | 비고 |
|---|---|---|
| `https://www.mdpi.com/2079-7737/11/2/209/s1` | **403 Forbidden** | MDPI 봇 차단 |
| `https://www.mdpi.com/article/10.3390/biology11020209/s1` | **403 Forbidden** | MDPI 봇 차단 |
| `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8869358/supplementaryFiles` | **200 OK (34,013,988 bytes)** | **성공** → `cache/supplements/EuropePMC_supplementaryFiles.zip` |
| `https://pmc.ncbi.nlm.nih.gov/articles/PMC8869358/bin/biology-11-00209-s001.zip` | 403 Forbidden | — |
| `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8869358/bin/biology-11-00209-s001.zip` | 403 Forbidden | — |
| `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC8869358` | 404 Not Found | 구 엔드포인트 폐지 |

확보한 zip 구조(내부 중첩 zip `biology-11-00209-s001.zip`):

| 파일 | 내용 | 본 검증 사용 |
|---|---|---|
| `SupplementaryB.csv` | 7,871 화합물 Tox21 hERG qHTS, pIC50 + Aff(0/1) = **논문 S2** | **사용** (`cache/supplements/S2_tox21_hERG.csv`, 7,562 레코드) |
| `SupplementaryC.csv`, `SupplementaryE.csv` | QSAR용 descriptor 테이블 (26 / 31 MB) | 미사용 |
| `SupplementaryD.csv` | 외부 검증셋(AID588834) 활성값 | 미사용 (PubChem 원본 사용) |
| `SupplementaryA.pdf`, `SupplementaryF.pdf` | 곡선/부속 문서 | 미사용 |
| `SupplementaryG.csv` | chemotype 빈도 요약 | 미사용 |

> 보충자료 S4(8,311 화합물 Tox21+ChEMBL 통합 활성표)는 배포 zip에 포함되어 있지 않아 확보하지 못했다.
> S2 값은 표준 pIC50(mol/L)이며, vanoxerine(pIC50 7.12 → IC50 0.076 µM) 등 문헌값과 대조해 확인했다.

### 3.2 대체 종말점 데이터 (Tox21 / ClinTox)

| 소스 | URL | 결과 |
|---|---|---|
| Tox21 12-endpoint | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz` | 200 OK, 7,831 화합물 |
| ClinTox | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz` | 200 OK, 1,484 화합물 |

### 3.3 캐시 파일

| 파일 | 내용 |
|---|---|
| `cache/chembl_activity_CHEMBL240.jsonl` | ChEMBL hERG activity 41,078건 |
| `cache/chembl_assays_CHEMBL240.jsonl` | ChEMBL hERG assay 4,829건 |
| `cache/pubchem_aid588834.csv` | PubChem AID588834 원본 CSV |
| `cache/pubchem_cid_props.jsonl` | CID → InChIKey/SMILES (4,743건) |
| `cache/supplements/EuropePMC_supplementaryFiles.zip` | 논문 보충자료 (34 MB) |
| `cache/supplements/S2_tox21_hERG.csv` | 논문 S2 추출본 |
| `cache/multitarget/*.jsonl` | 다중 타깃 activity 66,197건 |
| `cache/tox21.csv.gz`, `cache/clintox.csv.gz` | 대체 종말점 데이터 |
| `cache/external_smiles_keys.csv`, `cache/external_smiles_fps.pkl` | 외부 SMILES 구조 키·지문 캐시 |
| `cache/supplement_attempts.json`, `cache/alt_datasets_log.json` | 다운로드 시도 결과 자동 기록 |

