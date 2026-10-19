# Data

원본 `train.csv`는 저장소에 포함되어 있지 않습니다.  
강의 데이터 재배포 제한과 파일 크기(약 52MB) 때문입니다.

## 사용 방법

아래 위치에 `train.csv`를 두고 노트북을 실행하세요.

```text
skala-data-mini-project/
└── train.csv
```

## 컬럼 구성

| 컬럼 그룹 | 개수 | 설명 |
|---|---:|---|
| `SMILES` | 1 | 분자 구조 문자열 |
| `ecfp_*` | 1,024 | ECFP (Morgan radius 2, 1024-bit) |
| `fcfp_*` | 1,024 | FCFP (functional class fingerprint) |
| `ptfp_*` | 1,024 | PTFP (pattern/substructure fingerprint) |
| `MolWt` | 1 | 분자량 |
| `clogp` | 1 | 계산된 지용성(logP) |
| `sa_score` | 1 | 합성 접근성 점수 |
| `qed` | 1 | 약물 유사성 점수 |
| `label` | 1 | `1 = 독성 없음`, `0 = 독성 있음` |

- 총 8,349행 × 3,078열
- 결측치 0건
- 라벨 분포: 1(비독성) 4,542개(54.4%), 0(독성) 3,807개(45.6%)
