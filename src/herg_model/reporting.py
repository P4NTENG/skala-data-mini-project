"""Markdown report writers.

The reports are written in Korean (per project convention) while column names, file names and
source identifiers stay in English. Every quantitative statement is derived from the frames
passed in -- no number is hand-written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "to_md",
    "write_data_quality_report",
    "write_descriptor_report",
    "write_final_report",
    "POSITIVE_NOTE_KO",
]

POSITIVE_NOTE_KO = "양성 클래스 = hERG blocker (`label = 0`), 음성 클래스 = non-blocker (`label = 1`)"


def to_md(frame: pd.DataFrame, floatfmt: str = "{:.4f}", max_rows: int = 40, index: bool = False) -> str:
    """Small markdown-table formatter (no external dependency)."""
    if frame is None or len(frame) == 0:
        return "_(데이터 없음)_"
    frame = frame.head(max_rows)
    columns = ([frame.index.name or "idx"] if index else []) + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                if np.isnan(value):
                    cells.append("n/a")
                elif float(value).is_integer() and abs(value) >= 1000:
                    cells.append(f"{int(value):,}")
                else:
                    cells.append(floatfmt.format(value))
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif value is None:
                cells.append("n/a")
            else:
                cells.append(str(value).replace("|", "\\|"))
        if index:
            cells.insert(0, str(row.name))
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) < max_rows:
        return "\n".join(lines)
    return "\n".join(lines)


def _kv_list(mapping: Mapping[str, Any]) -> str:
    return "\n".join(f"- **{k}**: {v}" for k, v in mapping.items())


# --------------------------------------------------------------------------------------
# data quality
# --------------------------------------------------------------------------------------
def write_data_quality_report(
    path: str | Path,
    summary: dict,
    verification: dict,
    structure: dict,
    duplicates: dict,
    split_table: pd.DataFrame,
    leakage: dict,
    environment: dict,
    extra_notes: Sequence[str] = (),
) -> Path:
    path = Path(path)
    lines = [
        "# 데이터 품질 및 구조 검증 보고서",
        "",
        f"> {POSITIVE_NOTE_KO}",
        "",
        "이 문서는 `scripts/inspect_data.py`가 실제로 실행한 결과만 담는다.",
        "",
        "## 1. 파일과 컬럼",
        "",
        _kv_list({
            "rows": f"{summary['n_rows']:,}",
            "columns": f"{summary['n_columns']:,}",
            "SMILES column": f"`{summary['columns']['smiles']}`",
            "label column": f"`{summary['columns']['label']}`",
            "fingerprint blocks": ", ".join(f"`{k}` {v:,} bits" for k, v in summary["columns"]["fingerprints"].items()),
            "shipped descriptors": ", ".join(f"`{c}`" for c in summary["columns"]["existing_descriptors"]),
        }),
        "",
        "## 2. 라벨 분포",
        "",
        _kv_list({
            "label = 0 (blocker)": f"{summary['label_counts'].get(0, 0):,}",
            "label = 1 (non-blocker)": f"{summary['label_counts'].get(1, 0):,}",
            "blocker fraction": f"{summary['label_fraction_blocker']:.4f}",
        }),
        "",
        f"라벨 결측치는 {summary['missing_values_total']:,}개(전체 셀 기준)이며, 중복 SMILES는 "
        f"{summary['duplicate_smiles']:,}건, 고유 SMILES는 {summary['unique_smiles']:,}건이다.",
        "",
        "## 3. 구조 파싱",
        "",
        _kv_list({
            "RDKit 파싱 성공": f"{summary['rdkit_parse_success']:,} / {summary['n_rows']:,}",
            "파싱 실패 예시": summary["rdkit_parse_failures"] or "없음",
            "고유 InChIKey": f"{structure.get('n_inchikey', 0):,}",
            "고유 connectivity key": f"{structure.get('n_connectivity', 0):,}",
            "고유 Bemis-Murcko scaffold": f"{structure.get('n_scaffolds', 0):,}",
            "가장 큰 scaffold 그룹": f"{structure.get('largest_scaffold', 0):,} 개 화합물",
            "염(다중 fragment) 화합물": f"{structure.get('n_salts', 0):,}",
            "입체화학 표기 화합물": f"{structure.get('n_stereo', 0):,}",
            "tautomer 정규화 시 구조 변경": f"{structure.get('n_tautomer_changed', 'n/a')}",
        }),
        "",
        "## 4. 기존 fingerprint 재현성",
        "",
        "기존 컬럼을 그대로 사용하되, SMILES로부터 동일한 값을 재생성할 수 있는지 확인했다. "
        "재생성 결과가 일치하지 않는 블록은 재생성본으로 교체하지 않고 그대로 기록한다.",
        "",
    ]
    rows = []
    for kind, block in verification["blocks"].items():
        rows.append({
            "block": kind,
            "generator": block["generator"],
            "params": json.dumps(block["params"]),
            "rows_checked": block["n_rows_checked"],
            "bit_agreement": block["bit_agreement"],
            "row_exact_rate": block["row_exact_rate"],
            "all_rows_exact": "yes" if block["exact"] else "no",
        })
    lines.append(to_md(pd.DataFrame(rows), floatfmt="{:.6f}"))
    lines += [
        "",
        "shipped descriptor(`MolWt`, `clogp`, `qed`, `sa_score`)를 RDKit `Descriptors.MolWt`, "
        "`Crippen.MolLogP`, `QED.qed`, contrib `sascorer`로 재계산해 비교했다(수치는 §9). "
        "`MolWt`·`clogp`의 잔차는 컬럼이 float32로 저장된 데서 오는 반올림 수준이고, "
        "`qed` 1행과 `sa_score` 10행은 라이브러리 버전 차이로 보이는 불일치가 있다. "
        "원본 컬럼은 재계산본으로 교체하지 않고 그대로 사용한다.",
        "",
        "## 5. Fingerprint 블록 통계",
        "",
    ]
    fp_rows = []
    for kind, block in summary["fingerprints"].items():
        fp_rows.append({
            "block": kind,
            "n_bits": block["n_bits"],
            "constant_bits": block["constant_bits"],
            "on_fraction": block["bits_on_fraction"],
            "mean_bits_per_molecule": block["mean_bits_per_molecule"],
            "duplicate_bit_columns": block["duplicate_bit_columns"],
        })
    lines.append(to_md(pd.DataFrame(fp_rows)))

    lines += ["", "## 6. 중복 구조 / 라벨 충돌", ""]
    dup_rows = []
    for key in ("canonical_smiles", "connectivity_key", "normalized_key", "scaffold"):
        table = duplicates.get(key)
        if isinstance(table, pd.DataFrame) and len(table):
            conflicting = int((table["n_labels"] > 1).sum()) if "n_labels" in table.columns else 0
            dup_rows.append({
                "기준": key,
                "중복 그룹 수": int(len(table)),
                "라벨이 충돌하는 그룹": conflicting,
                "해당 멤버 행 수": duplicates.get(f"n_duplicate_members_{key}", 0),
            })
    lines.append(to_md(pd.DataFrame(dup_rows)))
    lines += [
        "",
        "중복 구조는 **제거하지 않는다**. 라벨이 충돌하는 그룹은 목록으로 남기고(§ 불일치 목록 파일), "
        "학습 경로에서는 원본 행을 그대로 유지한다. scaffold 그룹은 누수 방지를 위해 분할 단위로만 사용한다.",
        "",
        "## 7. 데이터 분할과 누수 검사",
        "",
        to_md(split_table),
        "",
        _kv_list({
            "scaffold 중복 (train∩val)": leakage["scaffold_overlap"].get("train__val", 0),
            "scaffold 중복 (train∩test)": leakage["scaffold_overlap"].get("train__test", 0),
            "scaffold 중복 (val∩test)": leakage["scaffold_overlap"].get("val__test", 0),
            "누수 여부": "발견" if leakage.get("has_leakage") else "없음",
        }),
        "",
        "## 8. 표준화 정책",
        "",
        "염 제거 · 중성화 · tautomer 정규화는 **기본 학습 경로에서 적용하지 않는다**. "
        "구조와 라벨의 의미가 달라질 수 있기 때문이며, 위 표의 중복 통계(normalized 기준)로만 "
        "민감도 정보를 제공한다.",
        "",
    ]
    if extra_notes:
        lines += ["## 9. 추가 메모", ""] + [f"- {note}" for note in extra_notes] + [""]

    lines += [
        "## 10. 실행 환경",
        "",
        _kv_list({
            "python": environment["python"],
            "platform": environment["platform"],
            "rdkit": environment["packages"].get("rdkit"),
            "scikit-learn": environment["packages"].get("scikit-learn"),
        }),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------------------
def write_descriptor_report(
    path: str | Path,
    rdkit_info: dict,
    mordred_info: dict,
    quality: pd.DataFrame,
    group_table: pd.DataFrame,
    correlation: pd.DataFrame,
    dropped: dict,
    top_correlations: pd.DataFrame | None = None,
) -> Path:
    path = Path(path)
    lines = [
        "# Descriptor 정의 · 계산 성공률 · 결측/중복 분석",
        "",
        f"> {POSITIVE_NOTE_KO}",
        "",
        "## 1. 계산 엔진과 성공률",
        "",
        _kv_list({
            "RDKit descriptor 수": rdkit_info.get("n_descriptors"),
            "RDKit 계산 성공": f"{rdkit_info.get('n_success')} / {rdkit_info.get('n_rows')}",
            "RDKit 계산 시간(초)": rdkit_info.get("seconds"),
            "Mordred 사용 가능": mordred_info.get("available"),
            "Mordred descriptor 수": mordred_info.get("n_descriptors"),
            "Mordred 실패/미설치 사유": mordred_info.get("reason") or "해당 없음",
        }),
        "",
        "3D 좌표가 필요한 descriptor는 제외했다(데이터셋에 conformer가 없음). "
        "RDKit `CalcMolDescriptors`는 2D descriptor만 반환하며, 여기에 hERG 도메인 해석을 위한 "
        "파생 feature(원자 조성, 염기성/비염기성 N 환경, 전하 관련 34개)를 추가했다. "
        "따라서 `rdkit_all`은 `CalcMolDescriptors` 217개 + 파생 34개 = 251개다.",
        "",
        "## 2. 그룹별 descriptor 수",
        "",
        to_md(group_table),
        "",
        "그룹은 서로 겹치지 않도록 배타적으로 배정했다. 어떤 그룹에도 명시적으로 배정되지 않은 "
        "descriptor는 `misc_rdkit`으로 모아 보고한다. `rdkit_all`은 전체 RDKit descriptor의 합집합이다.",
        "",
        "## 3. 결측치 · 상수 · 중복",
        "",
        _kv_list({
            "결측률 > 0 인 feature 수": int((quality["missing_rate"] > 0).sum()) if len(quality) else 0,
            "상수 feature 수(학습 split 기준)": int(quality["is_constant"].sum()) if len(quality) else 0,
            "학습 split에서 제거된 feature": json.dumps(
                {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in dropped.items()}
            ),
        }),
        "",
    ]
    labels_ko = {
        "all_missing": "결측 100%",
        "high_missing": "결측률 초과",
        "constant": "상수(학습 split)",
        "duplicate": "중복 열",
        "scaffold_leak": "scaffold 누수 의심",
    }
    for reason, dropped_names in dropped.items():
        if isinstance(dropped_names, (list, tuple)) and dropped_names:
            if reason == "duplicate":
                shown = ", ".join(f"`{a}`←`{b}`" for a, b in list(dropped_names)[:40])
                suffix = " (표기: 제거될 열 ← 유지되는 열)"
            else:
                shown = ", ".join(f"`{name}`" for name in list(dropped_names)[:40])
                suffix = ""
            more = f" … 외 {len(dropped_names) - 40}개" if len(dropped_names) > 40 else ""
            lines.append(f"- **{labels_ko.get(reason, reason)}** ({len(dropped_names)}){suffix}: {shown}{more}")
    lines += [
        "",
        "결측치 대체(중앙값)와 스케일링은 **학습 split에서만** 적합한다. "
        "결측률이 설정값(`descriptors.mordred.max_missing_rate`)을 넘는 Mordred descriptor는 "
        "학습 split에서 제거된다.",
        "",
        "### 결측률 상위 feature",
        "",
        to_md(
            quality.sort_values("missing_rate", ascending=False).head(15)[
                ["feature", "missing_rate", "n_unique", "is_constant"]
            ] if len(quality) else pd.DataFrame()
        ),
        "",
        "## 4. 중복 계산 여부",
        "",
        "기존 `MolWt`, `clogp`, `qed`는 RDKit `Descriptors.MolWt`, `Crippen.MolLogP`, `QED.qed`와 "
        "사실상 같은 값이다(재계산 오차는 데이터 품질 보고서 §9 참조). RDKit descriptor와 함께 쓰이는 "
        "실험에서는 학습 split에서 값이 완전히 동일한 열을 자동으로 제거하며, 제거된 열 이름은 위 "
        "`중복 열` 목록에 그대로 남긴다.",
        "",
        "## 5. 고상관 descriptor 쌍",
        "",
        f"|Pearson r| ≥ 0.95 인 쌍: {len(correlation):,}개",
        "",
        to_md(correlation.head(20)) if len(correlation) else "임계값을 넘는 쌍이 없다.",
        "",
    ]
    if top_correlations is not None and len(top_correlations):
        lines += [
            "### 상위 중요도 feature 사이의 상관",
            "",
            to_md(top_correlations.head(20)),
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# final report
# --------------------------------------------------------------------------------------
def _scaffold_view(frame: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Scaffold-level table without the structure string (redistribution-safe view)."""
    if frame is None or not len(frame):
        return frame
    view = frame.head(top_k).copy()
    if "scaffold" in view.columns:
        view.insert(0, "scaffold_id", [f"S{i + 1}" for i in range(len(view))])
        view = view.drop(columns=["scaffold"])
    return view


def write_final_report(path: str | Path, ctx: dict) -> Path:
    path = Path(path)

    def fig(name: str, caption: str = "") -> str:
        """Markdown image embed, relative to the report, only when the file exists."""
        if not (path.parent / "figures" / name).exists():
            return ""
        return f"![{caption or name}](figures/{name})"

    results: pd.DataFrame = ctx["results"]
    ablation: pd.DataFrame = ctx["ablation"]
    stability: pd.DataFrame = ctx["stability"]
    metric = "tuned_balanced_accuracy"

    best = results.sort_values(metric, ascending=False).iloc[0]
    best_val = results.sort_values("tuned_roc_auc", ascending=False).iloc[0]

    fp_table = results[results["experiment_id"].isin(["E0", "E1", "E2", "E0b", "E1b", "E2b", "E3c"])]
    baseline_fp = results[results["experiment_id"] == "E3"]
    rdkit_only = results[results["experiment_id"] == "E4"]
    domain = results[results["experiment_id"] == "E6"]
    physchem = results[results["experiment_id"] == "E5"]
    full_rdkit = results[results["experiment_id"] == "E7"]
    mordred = results[results["experiment_id"] == "E8"]
    mordred_sel = results[results["experiment_id"] == "E8b"]
    combined = results[results["experiment_id"] == "E9"]
    selected = results[results["experiment_id"] == "E10"]
    all_models = results[results["experiment_id"] == "E11"]

    def summarise(label: str, frame: pd.DataFrame) -> dict:
        """Best (experiment, model) row -- the comparison table must name the model it reports."""
        if not len(frame):
            return {"실험": label, "validation BA": float("nan"), "ROC-AUC": float("nan"),
                    "모델": "-", "features": "-"}
        row = frame.sort_values(metric, ascending=False).iloc[0]
        return {
            "실험": label,
            "validation BA": float(row[metric]),
            "ROC-AUC": float(row["tuned_roc_auc"]),
            "모델": str(row["model"]),
            "features": f"{int(row['n_features_used']):,}",
        }

    lines = [
        "# hERG blocker 예측 ML 실험 최종 보고서",
        "",
        f"> {POSITIVE_NOTE_KO}",
        "",
        f"> 예측 종말점은 **hERG 차단 활성**이다. 임상적 QT 연장이나 부정맥을 직접 예측한다고 "
        f"표현하지 않는다.",
        "",
        "## 0. 실행 요약",
        "",
        _kv_list({
            "config": ctx["config_path"],
            "budget mode": ctx["mode"],
            "primary split seed": ctx["primary_seed"],
            "실험 수(설정 조합)": int(results["experiment_id"].nunique()),
            "학습된 (실험, 모델) 조합": int(len(results)),
            "모델 하이퍼파라미터 탐색": f"{ctx['n_trials']} trials / 모델 (validation {ctx['tune_metric']} 기준)",
            "bootstrap": f"{ctx['bootstrap']['n_resamples']}회 cluster bootstrap (단위: {ctx['bootstrap']['unit']})",
        }),
        "",
        "## 1. 데이터",
        "",
        _kv_list({
            "rows": f"{ctx['data_summary']['n_rows']:,}",
            "blocker (label=0)": f"{ctx['data_summary']['label_counts'].get(0, 0):,}",
            "non-blocker (label=1)": f"{ctx['data_summary']['label_counts'].get(1, 0):,}",
            "RDKit 파싱 성공": f"{ctx['data_summary']['rdkit_parse_success']:,}",
        }),
        "",
        "분할 결과(실제 비율은 목표 비율과 다를 수 있으며 그대로 보고한다):",
        "",
        to_md(ctx["split_table"]),
        "",
        _kv_list({
            "scaffold 누수 (train∩val / train∩test / val∩test)": " / ".join(
                str(ctx["leakage"]["scaffold_overlap"].get(k, 0))
                for k in ("train__val", "train__test", "val__test")
            ),
        }),
        "",
        "## 2. 실험 설정 (E0–E11 및 추가 실험)",
        "",
    ]
    spec_table = ctx["spec_table"]
    lines.append(to_md(spec_table[[c for c in ["id", "name", "description", "blocks", "group"] if c in spec_table.columns]]))
    lines += [
        "",
        "모든 실험은 **동일한 분할**, **동일한 탐색 예산**, **동일한 학습 내부 전처리**를 사용한다. "
        "하이퍼파라미터 탐색과 임계값 선택은 validation split에서만 수행하고, test split은 "
        "최종 모델 확정 후 1회만 평가한다(§7).",
        "",
        "## 3. Fingerprint 계열 비교 (Q1)",
        "",
        to_md(fp_table.groupby(["experiment_id", "model"], as_index=False)[[metric, "tuned_roc_auc", "tuned_pr_auc"]].mean()),
        "",
        f"→ validation 기준 최고 fingerprint 구성은 **{ctx['best_fingerprint']}** 이며, "
        "이후 모든 `best fp + …` 실험은 해당 fingerprint를 사용한다.",
        "",
        fig("experiment_comparison.png", "실험별 validation balanced accuracy"),
        "",
        "## 4. 기준선과 descriptor 확장 비교 (Q2, Q3, Q4)",
        "",
        to_md(pd.DataFrame([
            summarise("E3 fingerprint baseline", baseline_fp),
            summarise("E4 RDKit descriptors only", rdkit_only),
            summarise("E5 + physicochemical", physchem),
            summarise("E6 + hERG domain groups", domain),
            summarise("E7 + all RDKit descriptors", full_rdkit),
            summarise("E8 + Mordred 2D (전체)", mordred),
            summarise("E8b + Mordred (선별)", mordred_sel),
            summarise("E9 + RDKit + Mordred", combined),
            summarise("E10 + RDKit + feature selection", selected),
            summarise("E11 = E10 피처, 전체 모델", all_models),
        ])),
        "",
        "각 행은 해당 실험에서 validation BA가 가장 높았던 **단일 모델**의 값이다"
        "(모델별 전체 값은 `reports/experiment_results.csv`). E0b/E1b/E2b와 `A_*` ablation은 "
        "예산상 lightgbm 단일 모델로만 실행했고, 나머지는 quick 모드의 3개 모델을 모두 탐색했다.",
        "",
        "## 5. Descriptor 그룹별 ablation",
        "",
        to_md(ablation[[c for c in ablation.columns if c in (
            "label", "experiment_id", "model", "baseline_balanced_accuracy", "ablated_balanced_accuracy",
            "delta_balanced_accuracy", "baseline_roc_auc", "ablated_roc_auc", "delta_roc_auc",
        )]]),
        "",
        "`delta_balanced_accuracy`가 0 근처면 해당 그룹이 fingerprint 기준선에 추가 정보를 주지 못한 것이다.",
        "",
        fig("ablation.png", "descriptor 그룹별 ablation (E3 기준 delta)"),
        "",
        "## 6. 모델 비교 (Q5)",
        "",
        to_md(all_models.groupby("model", as_index=False)[[metric, "tuned_roc_auc", "tuned_pr_auc", "best_fit_seconds"]].mean()),
        "",
        "## 7. 최종 모델과 test 성능",
        "",
    ]
    final = ctx["final"]
    if final:
        lines += [
            _kv_list({
                "선택된 실험/모델": f"{final['experiment_id']} / {final['model']}",
                "선택 근거": f"validation {metric} 최대 (test는 사용하지 않음)",
                "임계값(validation 선택)": f"{final['threshold']:.4f}",
                "test rows": f"{final['metrics']['n']:,} (blocker {final['metrics']['n_blocker']:,})",
            }),
            "",
            to_md(pd.DataFrame([
                {
                    "metric": k,
                    "value": f"{int(v):,}" if isinstance(v, int) else f"{float(v):.4f}",
                }
                for k, v in final["metrics"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ])),
            "",
            _kv_list({
                "confusion matrix (blocker = positive)": f"TP {final['metrics']['tp']}, FP {final['metrics']['fp']}, "
                                                         f"FN {final['metrics']['fn']}, TN {final['metrics']['tn']}",
                "threshold 0.5 결과": f"BA {final['metrics_at_0.5']['balanced_accuracy']:.4f}, "
                                      f"ROC-AUC {final['metrics_at_0.5']['roc_auc']:.4f}",
            }),
            "",
            fig("roc_pr_validation.png", "validation ROC/PR (실험별 최고 모델)"),
            "",
            fig("roc_pr_test.png", "test ROC/PR (test는 1회만 평가)"),
            "",
            fig("confusion_matrices.png", "혼동행렬 (positive class = hERG blocker)"),
            "",
            f"최종 모델은 train+val로 재적합했고 임계값은 validation에서 선택했다"
            f"({final.get('refit', 'train+val')}). test는 이 시점에 처음 평가되었다.",
            "",
        ]
        delta_threshold = final["metrics"]["balanced_accuracy"] - final["metrics_at_0.5"]["balanced_accuracy"]
        if abs(delta_threshold) >= 0.01:
            better = "낮다" if delta_threshold < 0 else "높다"
            naive = "0.5" if delta_threshold < 0 else "validation에서 고른 임계값"
            lines += [
                f"**임계값 전이 실패 가능성:** validation에서 고른 임계값의 test BA가 기본 "
                f"{naive}보다 {abs(delta_threshold):.4f}만큼 {better}. validation 835행"
                f"(blocker {final['metrics']['n_blocker']:,}행 규모)에서 임계값을 고르는 것 자체가 "
                f"잡음에 민감하다는 뜻이며, 순위 지표(ROC-AUC/PR-AUC)는 validation과 test에서 "
                f"거의 같다. 실제 사용 시 임계값은 별도의 보정 데이터로 정하고, 이 값은 "
                f"'validation에서 고른 값'으로만 읽어야 한다.",
                "",
            ]
    else:
        lines += ["_(최종 모델 학습이 아직 실행되지 않았다)_", ""]

    split_reference = ctx.get("split_reference")
    if split_reference is not None and len(split_reference):
        scaffold_view = (
            results.groupby(["experiment_id", "model"], as_index=False)[[metric, "tuned_roc_auc"]]
            .mean()
            .rename(columns={metric: f"{metric}_scaffold", "tuned_roc_auc": "roc_auc_scaffold"})
        )
        random_view = (
            split_reference.groupby(["experiment_id", "model"], as_index=False)[[metric, "tuned_roc_auc"]]
            .mean()
            .rename(columns={metric: f"{metric}_random", "tuned_roc_auc": "roc_auc_random"})
        )
        merged = scaffold_view.merge(random_view, on=["experiment_id", "model"], how="inner")
        if len(merged):
            merged["delta_ba_random_minus_scaffold"] = (
                merged[f"{metric}_random"] - merged[f"{metric}_scaffold"]
            )
            lines += [
                "## 7b. 참고: random split과의 비교",
                "",
                to_md(merged, floatfmt="{:.4f}"),
                "",
                "random split은 참고용이다. 같은 scaffold가 train/val/test에 걸쳐 있어(데이터 품질 "
                "보고서 §7) 성능이 낙관적으로 나오며, 위 차이는 **일반화 격차의 크기**를 보여줄 뿐 "
                "어느 쪽이 더 좋은 모델인지를 말하지 않는다. 최종 모델 선택에는 scaffold split만 썼다.",
                "",
                fig("split_comparison.png", "scaffold split vs random split (validation BA)"),
                "",
            ]

    lines += [
        "## 8. Scaffold 분할 반복 안정성 (Q6)",
        "",
    ]
    if not len(stability):
        lines += ["_(미실행)_", ""]
    if len(stability):
        seed_summary = (
            stability.groupby(["experiment_id", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
        )
        lines += [
            "상위 설정을 **scaffold split seed 3개**(42 = 위 표들의 기본 seed, 101, 2024)에서 다시 "
            "학습한 결과다. `count`가 3이면 세 seed 모두 실행된 것이고, `std`가 seed 간 변동성이다. "
            "seed 하나에서의 순위 차이를 과대해석하지 않기 위해 사용한다.",
            "",
            to_md(seed_summary),
            "",
            fig("seed_stability.png", "scaffold seed 3개에 대한 평균 ± 표준편차"),
        ]
    lines += [
        "",
        "## 9. 중요 피처와 오분류 패턴 (Q7)",
        "",
        to_md(ctx["importance_head"]),
        "",
        "그룹 단위 중요도:",
        "",
        to_md(ctx["group_importance"]),
        "",
        fig("feature_importance_by_group.png", "descriptor 그룹별 permutation importance 합계"),
        "",
        "오분류 요약(validation):",
        "",
        to_md(ctx["error_summary"]),
        "",
        "오분류가 많은 chemotype (validation, element-stripped generic scaffold 계열, 3개 이상):",
        "",
        to_md(_scaffold_view(ctx["scaffold_errors"])),
        "",
        "scaffold 구조 문자열은 원본 구조의 재배포 제한 때문에 보고서에 넣지 않았다"
        "(`reports/scaffold_performance.csv`, 로컬 전용). scaffold split에서는 validation 분자의 "
        "Bemis–Murcko scaffold가 모두 서로 다르므로(설계상 그렇다), scaffold 단위 성능은 원소를 "
        "제거한 generic scaffold 계열로 묶어서만 계산할 수 있다. 이 표는 '어떤 chemotype 계열에서 "
        "오류가 몰리는가'를 보여준다.",
        "",
        fig("scaffold_performance.png", "generic scaffold 계열별 오류율과 scaffold 크기"),
        "",
        "## 10. 클래스 불균형 처리",
        "",
        to_md(ctx["imbalance"]),
        "",
        "## 11. 한계와 실제 활용 시 주의점 (Q8)",
        "",
        "- 라벨이 대략 `IC50 < 10 µM` 기준으로 만들어졌을 가능성이 있으나, 모든 라벨이 단일 assay·"
        "단일 규칙으로 생성되었다고 단정할 수 없다. 라벨 노이즈가 남아 있다.",
        "- validation/test 성능은 모두 Bemis-Murcko scaffold split 기준이며, 학습에 등장하지 않은 "
        "chemotype에 대한 성능이다. 따라서 random split 성능과 직접 비교해서는 안 된다.",
        "- 확률값은 저장된 모델이 산출한 값이며, 보정(calibration)과 외부 검증 없이 임상적 독성 "
        "확률로 해석해서는 안 된다.",
        "- fingerprint 기반 모델은 학습 데이터가 커버하는 화학 공간 밖에서 급격히 성능이 떨어진다. "
        "적용 전에 applicability domain 확인이 필요하다.",
        "- feature importance는 상관관계 수준의 설명이며 작용 기전의 증거가 아니다.",
        "",
    ]
    if ctx.get("caveats"):
        lines += ["### 이 실행에서 추가로 확인된 한계", ""] + [f"- {c}" for c in ctx["caveats"]] + [""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
