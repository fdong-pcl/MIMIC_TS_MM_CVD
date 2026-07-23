import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from add_test100_seqseverity_labels import (
    DEFAULT_HOSP_ROOT,
    DEFAULT_MAPPING_ROOT,
    build_cvd_details,
    diagnosis_role,
    make_summary,
    validate_outputs,
)
from build_test100_multimodal_compact import build_manifest, ensure_dirs, load_source
from build_test100_pre_ds_onestudy_priority_current import (
    FULL_LABEL_COLUMNS,
    annotate_manifest_with_selected_study,
    build_ds_cutoff,
    build_file_summary,
    build_selected_modality_tables,
    filter_details,
    filter_master,
    read_subjects,
    validate_manifest,
)
from build_test100_subject100_balanced import copy_reusable_files, download_missing_or_empty_manifest


DEFAULT_SOURCE_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10"
DEFAULT_OUTPUT_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact"
DEFAULT_SPLITS_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10/step4_labels_splits/splits"
DEFAULT_MIMICIV_ROOT = "/Users/fandong/Desktop/pcl/Data/mimiciv"
DEFAULT_EXTERNAL_CXR_JPG_ROOT = "/Volumes/UD800/MIMIC_CXR/JPGzip/JPG"
DEFAULT_EXTERNAL_ECHO_ROOT = "/Volumes/thinkplus/echo"
DEFAULT_EXTERNAL_ECG_ZIP = "/Volumes/UD800/MIMICIV/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip"
DEFAULT_REUSE_ROOTS = [
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/subset2_test100_subject100_pre_ds_onestudy_cvd3_fullstudy_backup",
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/中期考核多模态100测试集",
]

DATASET_SPECS = {
    "simple": {
        "dataset_name": "midterm_simple100_pre_ds",
        "target_echo": 13,
        "target_cxr": 87,
        "disease_rule": "single_cvd_primary_seq1",
    },
    "complex5": {
        "dataset_name": "midterm_complex5_100_pre_ds",
        "target_echo": 40,
        "target_cxr": 60,
        "disease_rule": "five_cvd_categories_primary_seq1_cvd",
    },
}


def split_subjects(splits_root: Path) -> Set[int]:
    out: Set[int] = set()
    for name in ["train_subjects_all.txt", "val_subjects_all.txt", "test_subjects_all.txt"]:
        out.update(read_subjects(str(splits_root / name)))
    return out


def add_pre_ds_availability(labels: pd.DataFrame, audit: pd.DataFrame, allowed_subjects: Set[int]) -> pd.DataFrame:
    base = labels[labels["subject_id"].astype(int).isin(allowed_subjects)].copy()
    base["subject_id"] = base["subject_id"].astype(int)
    base["hadm_id"] = base["hadm_id"].astype(int)
    for mod in ["ecg", "cxr", "echo"]:
        hadms = set(audit.loc[audit["modality"].eq(mod), "hadm_id"].astype(int))
        base[f"{mod}_pre_ds"] = base["hadm_id"].isin(hadms)
    base["pre_ds_score"] = base[["ecg_pre_ds", "cxr_pre_ds", "echo_pre_ds"]].sum(axis=1)
    base["dischtime_dt"] = pd.to_datetime(base["dischtime"], errors="coerce")
    return base


def build_disease_pool(labels: pd.DataFrame, hosp_root: Path, mapping_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    details, gems_meta = build_cvd_details(labels, hosp_root, mapping_root)
    grouped = details.groupby(["subject_id", "hadm_id"])
    facts = grouped.agg(
        cvd_diag_rows=("icd_code", "size"),
        cvd_unique_count_detail=("CVD_coarse_category", "nunique"),
        primary_cvd_rows=("seq_num", lambda s: int((s.astype(int) == 1).sum())),
    ).reset_index()
    primary = (
        details[details["seq_num"].astype(int).eq(1)]
        .groupby(["subject_id", "hadm_id"])
        .agg(primary_cvd_coarse_category=("CVD_coarse_category", "first"))
        .reset_index()
    )
    facts = facts.merge(primary, on=["subject_id", "hadm_id"], how="left")
    return facts, details, gems_meta


def select_balanced(pool: pd.DataFrame, target_echo: int, target_cxr: int, target_total: int = 100) -> pd.DataFrame:
    work = pool[pool["ecg_pre_ds"]].copy()
    echo_pool = work[work["echo_pre_ds"]].copy()
    cxr_pool = work[work["cxr_pre_ds"] & ~work["echo_pre_ds"]].copy()

    selected_parts: List[pd.DataFrame] = []
    used_subjects: Set[int] = set()

    def take_balanced(df: pd.DataFrame, n: int) -> pd.DataFrame:
        nonlocal used_subjects
        if n <= 0:
            return df.head(0).copy()
        candidates = df[~df["subject_id"].isin(used_subjects)].copy()
        if len(candidates) < n:
            raise ValueError(f"候选不足: need={n}, available={len(candidates)}")
        candidates = candidates.sort_values(
            ["primary_cvd_count_so_far", "primary_cvd_coarse_category", "pre_ds_score", "dischtime_dt", "hadm_id"],
            ascending=[True, True, False, False, True],
        )
        chosen_rows = []
        counts: Dict[str, int] = {}
        remaining = candidates.copy()
        while len(chosen_rows) < n:
            remaining = remaining[~remaining["subject_id"].isin(used_subjects)]
            if remaining.empty:
                break
            remaining["primary_cvd_count_so_far"] = remaining["primary_cvd_coarse_category"].map(lambda x: counts.get(str(x), 0))
            row = remaining.sort_values(
                ["primary_cvd_count_so_far", "primary_cvd_coarse_category", "pre_ds_score", "dischtime_dt", "hadm_id"],
                ascending=[True, True, False, False, True],
            ).iloc[0]
            chosen_rows.append(row)
            used_subjects.add(int(row["subject_id"]))
            cls = str(row["primary_cvd_coarse_category"])
            counts[cls] = counts.get(cls, 0) + 1
            remaining = remaining.drop(index=row.name)
        if len(chosen_rows) != n:
            raise ValueError(f"均衡选择候选不足: need={n}, got={len(chosen_rows)}")
        return pd.DataFrame(chosen_rows)

    echo_pool["primary_cvd_count_so_far"] = 0
    cxr_pool["primary_cvd_count_so_far"] = 0
    selected_parts.append(take_balanced(echo_pool, target_echo))
    selected_parts.append(take_balanced(cxr_pool, target_cxr))
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != target_total:
        raise ValueError(f"选择数量异常: {len(selected)} != {target_total}")
    if selected["subject_id"].nunique() != target_total or selected["hadm_id"].nunique() != target_total:
        raise ValueError("subject/hadm 唯一性异常")
    return selected.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)


def make_candidate_pools(base: pd.DataFrame, facts: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    pool = base.merge(facts, on=["subject_id", "hadm_id"], how="inner")
    simple = pool[
        pool["cvd_diag_rows"].astype(int).eq(1)
        & pool["primary_cvd_rows"].astype(int).eq(1)
    ].copy()
    complex5 = pool[
        pool["cvd_unique_count_detail"].astype(int).eq(5)
        & pool["primary_cvd_rows"].astype(int).ge(1)
    ].copy()
    return {"simple": simple, "complex5": complex5}


def make_label_diagnosis_summary(seq_details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detail_list_cols = [
        "seq_num",
        "icd_code",
        "icd_version",
        "long_title",
        "CVD_fine_category",
        "norm_icd10_code",
        "norm_icd10_title",
        "norm_method",
    ]
    for (subject_id, hadm_id), group in seq_details.groupby(["subject_id", "hadm_id"], sort=False):
        group = group.sort_values(["seq_num", "icd_code"]).copy()
        category_rows = []
        for coarse, sub in group.groupby("CVD_coarse_category"):
            first = sub.sort_values(["seq_num", "icd_code"]).iloc[0]
            category_rows.append((int(first["seq_num"]), str(coarse)))
        category_rows = sorted(category_rows)
        coarse_sorted = [coarse for _, coarse in category_rows]
        row = {"subject_id": int(subject_id), "hadm_id": int(hadm_id)}
        for col in detail_list_cols:
            values = group[col].where(pd.notna(group[col]), None).tolist()
            row[col] = json.dumps(values, ensure_ascii=False)
        row["CVD_coarse_category"] = json.dumps(coarse_sorted, ensure_ascii=False)
        row["cvd_list"] = json.dumps(coarse_sorted, ensure_ascii=False)
        row["cvd_unique_count"] = int(len(coarse_sorted))
        rows.append(row)
    return pd.DataFrame(rows)


def keep_manifest_selected_studies(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "selected_study_id" not in manifest.columns:
        return manifest

    def belongs_to_selected_study(row: pd.Series) -> bool:
        study_id = str(row.get("selected_study_id", "")).strip()
        if not study_id or study_id.lower() == "nan":
            return False
        study_id = study_id[1:] if study_id.lower().startswith("s") else study_id
        return f"/s{study_id}/" in str(row.get("relative_path", ""))

    out = manifest[manifest.apply(belongs_to_selected_study, axis=1)].copy()
    dropped = len(manifest) - len(out)
    if dropped:
        print(f"Filtered {dropped} manifest files that did not belong to selected studies.")
    return out.reset_index(drop=True)


def _copy_or_link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except Exception:
        shutil.copy2(src, dst)
        return "copied"


def _needs_file(path: Path) -> bool:
    return not (path.exists() and path.is_file() and path.stat().st_size > 0)


def _cxr_zip_entry(relative_path: str) -> Tuple[Optional[Path], Optional[str]]:
    parts = Path(relative_path).parts
    if len(parts) < 4 or parts[0] != "files":
        return None, None
    zip_name = f"{parts[1]}.zip"
    entry = "/".join(parts[2:])
    return Path(zip_name), entry


def copy_from_external_origin_sources(
    manifest: pd.DataFrame,
    out_root: Path,
    cxr_jpg_root: Optional[str],
    echo_root: Optional[str],
    ecg_zip_path: Optional[str],
) -> Dict[str, Dict[str, int]]:
    stats = {
        "cxr_jpg_zip": {"extracted": 0, "exists": 0, "missing": 0},
        "echo_dir": {"linked": 0, "copied": 0, "exists": 0, "missing": 0},
        "ecg_zip": {"extracted": 0, "exists": 0, "missing": 0},
    }
    if manifest.empty:
        return stats

    cxr_root = Path(cxr_jpg_root) if cxr_jpg_root else None
    echo_base = Path(echo_root) / "physionet.org/files/mimic-iv-echo/0.1" if echo_root else None
    ecg_zip = Path(ecg_zip_path) if ecg_zip_path else None
    ecg_prefix = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"

    zip_cache: Dict[Path, zipfile.ZipFile] = {}
    try:
        for row in manifest.to_dict("records"):
            modality = str(row["modality"])
            dst = out_root / str(row["target_path"])
            if not _needs_file(dst):
                if modality == "cxr":
                    stats["cxr_jpg_zip"]["exists"] += 1
                elif modality == "echo":
                    stats["echo_dir"]["exists"] += 1
                elif modality == "ecg":
                    stats["ecg_zip"]["exists"] += 1
                continue

            rel = str(row["relative_path"])
            if modality == "cxr":
                zip_name, entry = _cxr_zip_entry(rel)
                zip_path = cxr_root / zip_name if cxr_root and zip_name else None
                if not zip_path or not zip_path.exists() or not entry:
                    stats["cxr_jpg_zip"]["missing"] += 1
                    continue
                zf = zip_cache.get(zip_path)
                if zf is None:
                    zf = zipfile.ZipFile(zip_path)
                    zip_cache[zip_path] = zf
                try:
                    info = zf.getinfo(entry)
                    if info.file_size <= 0:
                        stats["cxr_jpg_zip"]["missing"] += 1
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dst, "wb") as fout:
                        shutil.copyfileobj(src, fout)
                    stats["cxr_jpg_zip"]["extracted"] += 1
                except KeyError:
                    stats["cxr_jpg_zip"]["missing"] += 1

            elif modality == "echo":
                src = echo_base / rel if echo_base else None
                if not src or not src.exists() or src.stat().st_size <= 0:
                    stats["echo_dir"]["missing"] += 1
                    continue
                action = _copy_or_link(src, dst)
                stats["echo_dir"][action] += 1

            elif modality == "ecg":
                if not ecg_zip or not ecg_zip.exists():
                    stats["ecg_zip"]["missing"] += 1
                    continue
                zf = zip_cache.get(ecg_zip)
                if zf is None:
                    zf = zipfile.ZipFile(ecg_zip)
                    zip_cache[ecg_zip] = zf
                entry = ecg_prefix + rel
                try:
                    info = zf.getinfo(entry)
                    if info.file_size <= 0:
                        stats["ecg_zip"]["missing"] += 1
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dst, "wb") as fout:
                        shutil.copyfileobj(src, fout)
                    stats["ecg_zip"]["extracted"] += 1
                except KeyError:
                    stats["ecg_zip"]["missing"] += 1
    finally:
        for zf in zip_cache.values():
            zf.close()
    return stats


def write_dataset(
    out_root: Path,
    source_labels: pd.DataFrame,
    selected: pd.DataFrame,
    master: pd.DataFrame,
    all_details: Dict[str, Optional[pd.DataFrame]],
    selected_modality_details: Dict[str, pd.DataFrame],
    audit: pd.DataFrame,
    disease_details: pd.DataFrame,
    gems_meta: Dict[str, object],
    args: argparse.Namespace,
    spec: Dict[str, object],
) -> Dict[str, object]:
    dirs = ensure_dirs(str(out_root))
    selected = selected.copy()
    for mod in ["ecg", "cxr", "echo"]:
        pre_col = f"{mod}_pre_ds"
        use_col = f"use_{mod}"
        if use_col not in selected.columns:
            selected[use_col] = selected[pre_col]
        selected[use_col] = selected[use_col].astype(bool)
    selected_hadms = set(selected["hadm_id"].astype(int))

    use_hadms_by_mod = {
        mod: set(selected.loc[selected[f"use_{mod}"], "hadm_id"].astype(int))
        for mod in ["ecg", "cxr", "echo"]
    }
    chosen_modality_details = {}
    for mod, df in selected_modality_details.items():
        chosen_modality_details[mod] = df[df["hadm_id"].astype(int).isin(use_hadms_by_mod.get(mod, set()))].copy()
    filtered_details = filter_details(all_details, selected_hadms, chosen_modality_details)
    filtered_master = filter_master(master, selected_hadms, chosen_modality_details)
    selected_audit_parts = []
    for mod, hadms in use_hadms_by_mod.items():
        selected_audit_parts.append(
            audit[audit["modality"].eq(mod) & audit["hadm_id"].astype(int).isin(hadms)].copy()
        )
    selected_audit = pd.concat(selected_audit_parts, ignore_index=True) if selected_audit_parts else audit.head(0).copy()
    seq_details = disease_details[disease_details["hadm_id"].astype(int).isin(selected_hadms)].copy()
    summary = make_summary(seq_details)
    label_diagnosis = make_label_diagnosis_summary(seq_details)

    labels = source_labels[source_labels["hadm_id"].astype(int).isin(selected_hadms)].copy()
    replace_cols = [
        "seq_num",
        "icd_code",
        "icd_version",
        "long_title",
        "CVD_coarse_category",
        "CVD_fine_category",
        "norm_icd10_code",
        "norm_icd10_title",
        "norm_method",
        "cvd_list",
        "cvd_unique_count",
    ]
    labels = labels.drop(columns=[c for c in replace_cols if c in labels.columns])
    labels = labels.merge(label_diagnosis, on=["subject_id", "hadm_id"], how="left")
    if labels["cvd_list"].isna().any():
        missing = labels[labels["cvd_list"].isna()][["subject_id", "hadm_id"]].to_dict("records")[:10]
        raise ValueError(f"部分 labels 缺少重建诊断标签: {missing}")
    labels = labels.merge(selected[["hadm_id", "use_cxr", "use_ecg", "use_echo"]], on="hadm_id", how="left")
    labels["has_cxr"] = labels["use_cxr"].astype(int)
    labels["has_ecg"] = labels["use_ecg"].astype(int)
    labels["has_echo"] = labels["use_echo"].astype(int)
    labels["modality_score"] = labels[["has_cxr", "has_ecg", "has_echo"]].sum(axis=1)
    labels = labels.drop(columns=["use_cxr", "use_ecg", "use_echo"])
    labels = labels[FULL_LABEL_COLUMNS].sort_values(["subject_id", "hadm_id"])
    labels.to_csv(Path(dirs["labels"]) / "cohort_labels_test100.csv", index=False)
    labels.to_csv(Path(dirs["labels"]) / "cohort_labels_test100.csv.gz", index=False, compression="gzip")

    enriched = labels.merge(summary, on=["subject_id", "hadm_id"], how="left")
    expected_cvd_unique_count = int(spec.get("expected_cvd_unique_count", 1 if spec["disease_rule"] == "single_cvd_primary_seq1" else 5))
    seq_checks = validate_outputs(labels, enriched, seq_details, expected_cvd_unique_count=expected_cvd_unique_count)
    seq_details.to_csv(Path(dirs["labels"]) / "cvd_diagnosis_seq_details_test100.csv", index=False)
    seq_details.to_csv(Path(dirs["labels"]) / "cvd_diagnosis_seq_details_test100.csv.gz", index=False, compression="gzip")
    enriched.to_csv(Path(dirs["labels"]) / "cohort_labels_test100_seqseverity.csv", index=False)
    enriched.to_csv(Path(dirs["labels"]) / "cohort_labels_test100_seqseverity.csv.gz", index=False, compression="gzip")

    filtered_master.to_csv(Path(dirs["timeline"]) / "master_timeline_test100.csv.gz", index=False, compression="gzip")
    for mod, df in filtered_details.items():
        if df is not None:
            df.to_csv(Path(dirs["details"]) / f"details_{mod}_grp_test100.csv.gz", index=False, compression="gzip")
    selected_audit.to_csv(out_root / "modality_diagnosis_time_audit.csv", index=False)

    pairs = selected[["subject_id", "hadm_id"]].sort_values(["subject_id", "hadm_id"])
    pairs.to_csv(out_root / "test_subject_hadm.csv", index=False)
    (out_root / "test_subjects.txt").write_text("\n".join(map(str, pairs["subject_id"].astype(int).tolist())) + "\n")
    (out_root / "test_hadms.txt").write_text("\n".join(map(str, pairs["hadm_id"].astype(int).tolist())) + "\n")

    manifest = build_manifest(
        selected_labels=labels,
        details=filtered_details,
        cxr_base_url=args.cxr_base_url,
        ecg_base_url=args.ecg_base_url,
        echo_base_url=args.echo_base_url,
        cxr_format=args.cxr_format,
        ecg_expand_pairs=args.ecg_expand_pairs,
        echo_expand_all=args.echo_expand_all_dcm_by_study,
        echo_record_list_path=args.echo_record_list_path,
        out_root=str(out_root),
    )
    manifest = annotate_manifest_with_selected_study(manifest, selected_audit)
    manifest = keep_manifest_selected_studies(manifest)
    manifest.to_csv(out_root / "origin_manifest_test100.csv", index=False)
    manifest.to_csv(out_root / "origin_manifest_test100.csv.gz", index=False, compression="gzip")

    external_reuse_stats = {}
    if args.copy_from_external_origin_sources:
        external_reuse_stats = copy_from_external_origin_sources(
            manifest=manifest,
            out_root=out_root,
            cxr_jpg_root=args.external_cxr_jpg_root,
            echo_root=args.external_echo_root,
            ecg_zip_path=args.external_ecg_zip_path,
        )

    reuse_stats = {}
    for reuse_root in args.reuse_roots:
        if reuse_root and Path(reuse_root).exists():
            reuse_stats[str(reuse_root)] = copy_reusable_files(manifest, reuse_root, str(out_root))
    if args.download_origin_data:
        if not args.physionet_user or not args.physionet_password:
            raise ValueError("下载需要 --physionet-user/--physionet-password 或环境变量。")
        download_missing_or_empty_manifest(
            manifest,
            args.physionet_user,
            args.physionet_password,
            str(out_root),
            args.download_workers,
            args.download_retries,
            args.download_timeout_sec,
        )
    else:
        (out_root / "download_success.txt").write_text("")
        (out_root / "download_failed.txt").write_text("")

    file_summary = build_file_summary(manifest, selected_audit)
    stats = {
        "dataset_name": out_root.name,
        "disease_rule": spec["disease_rule"],
        "subject_count": int(labels["subject_id"].nunique()),
        "hadm_count": int(labels["hadm_id"].nunique()),
        "actual_ecg_hadm": int(labels["has_ecg"].sum()),
        "actual_cxr_hadm": int(labels["has_cxr"].sum()),
        "actual_echo_hadm": int(labels["has_echo"].sum()),
        "primary_cvd_distribution": {str(k): int(v) for k, v in selected["primary_cvd_coarse_category"].value_counts().sort_index().items()},
        "manifest_rows": int(len(manifest)),
        "manifest_validation": validate_manifest(str(out_root), manifest),
        "full_study_file_summary": file_summary,
        "external_origin_reuse": external_reuse_stats,
        "reuse": reuse_stats,
        "seqseverity": {
            **gems_meta,
            "checks": seq_checks,
            "detail_rows": int(len(seq_details)),
            "diagnosis_role_distribution": {str(k): int(v) for k, v in seq_details["diagnosis_role"].value_counts().items()},
            "norm_method_distribution": {str(k): int(v) for k, v in seq_details["norm_method"].value_counts().items()},
        },
    }
    (out_root / "stats_test100.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def run_exports(out_root: Path, mimiciv_root: str, expected_cxr: int, expected_echo: int) -> None:
    py = os.environ.get("PYTHON", "/Users/fandong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
    script_dir = Path(__file__).resolve().parent
    commands = [
        [py, str(script_dir / "export_test100_ds_notes.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-notes", "100"],
        [py, str(script_dir / "export_test100_cxr_reports.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-cxr-reports", str(expected_cxr)],
    ]
    if expected_echo > 0:
        commands.append(
            [py, str(script_dir / "export_test100_echo_measurements.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-echo-studies", str(expected_echo)]
        )
    for cmd in commands:
        subprocess.run(cmd, check=True)


def validate_selection(name: str, selected: pd.DataFrame, target_echo: int, target_cxr: int) -> None:
    if selected["subject_id"].nunique() != 100 or selected["hadm_id"].nunique() != 100:
        raise ValueError(f"{name}: subject/hadm 数量异常")
    if int(selected["ecg_pre_ds"].sum()) != 100:
        raise ValueError(f"{name}: ECG != 100")
    if int(selected["echo_pre_ds"].sum()) != target_echo:
        raise ValueError(f"{name}: Echo != {target_echo}")
    if int(selected["cxr_pre_ds"].sum()) != target_cxr:
        raise ValueError(f"{name}: CXR != {target_cxr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--splits-root", default=DEFAULT_SPLITS_ROOT)
    ap.add_argument("--hosp-root", default=DEFAULT_HOSP_ROOT)
    ap.add_argument("--mapping-root", default=DEFAULT_MAPPING_ROOT)
    ap.add_argument("--mimiciv-root", default=DEFAULT_MIMICIV_ROOT)
    ap.add_argument("--dataset-prefix", default="")
    ap.add_argument("--cxr-format", choices=["dcm", "jpg"], default="jpg")
    ap.add_argument("--cxr-base-url", default="https://physionet.org/files/mimic-cxr-jpg/2.1.0")
    ap.add_argument("--ecg-base-url", default="https://physionet.org/files/mimic-iv-ecg/1.0")
    ap.add_argument("--echo-base-url", default="https://physionet.org/files/mimic-iv-echo/0.1")
    ap.add_argument("--ecg-expand-pairs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-expand-all-dcm-by-study", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-record-list-path", default=None)
    ap.add_argument("--reuse-roots", nargs="*", default=DEFAULT_REUSE_ROOTS)
    ap.add_argument("--copy-from-external-origin-sources", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--external-cxr-jpg-root", default=DEFAULT_EXTERNAL_CXR_JPG_ROOT)
    ap.add_argument("--external-echo-root", default=DEFAULT_EXTERNAL_ECHO_ROOT)
    ap.add_argument("--external-ecg-zip-path", default=DEFAULT_EXTERNAL_ECG_ZIP)
    ap.add_argument("--download-origin-data", action="store_true")
    ap.add_argument("--physionet-user", default=os.environ.get("PHYSIONET_USER"))
    ap.add_argument("--physionet-password", default=os.environ.get("PHYSIONET_PASSWORD"))
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--download-retries", type=int, default=5)
    ap.add_argument("--download-timeout-sec", type=int, default=300)
    args = ap.parse_args()

    labels, master, details = load_source(args.source_root, "step_all")
    allowed = split_subjects(Path(args.splits_root))
    diagnostic_time = build_ds_cutoff(details["notes"])
    selected_modality_details, audit = build_selected_modality_tables(details, diagnostic_time)
    base = add_pre_ds_availability(labels, audit, allowed)
    facts, disease_details, gems_meta = build_disease_pool(base[["subject_id", "hadm_id", "cvd_unique_count", "CVD_coarse_category", "seq_num", "icd_code"]], Path(args.hosp_root), Path(args.mapping_root))
    pools = make_candidate_pools(base, facts)

    all_stats = {}
    for key, spec in DATASET_SPECS.items():
        selected = select_balanced(pools[key], int(spec["target_echo"]), int(spec["target_cxr"]))
        validate_selection(key, selected, int(spec["target_echo"]), int(spec["target_cxr"]))
        dataset_name = f"{args.dataset_prefix}{spec['dataset_name']}"
        out_root = Path(args.output_root) / dataset_name
        if out_root.exists() and args.dataset_prefix:
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        stats = write_dataset(
            out_root=out_root,
            source_labels=labels,
            selected=selected,
            master=master,
            all_details=details,
            selected_modality_details=selected_modality_details,
            audit=audit,
            disease_details=disease_details,
            gems_meta=gems_meta,
            args=args,
            spec=spec,
        )
        run_exports(out_root, args.mimiciv_root, int(spec["target_cxr"]), int(spec["target_echo"]))
        all_stats[key] = stats
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
