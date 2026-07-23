import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from build_test100_multimodal_compact import (
    build_manifest,
    ensure_dirs,
    load_source,
    parse_listlike,
)
from build_test100_subject100_balanced import (
    copy_reusable_files,
    download_missing_or_empty_manifest,
)


DEFAULT_SOURCE_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10"
DEFAULT_OUTPUT_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact"
DEFAULT_DATASET_NAME = "subset2_test100_subject100_pre_ds_onestudy_priority_current"
DEFAULT_CURRENT_ROOT = (
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/"
    "subset2_test100_subject100_balanced"
)
DEFAULT_RESTRICT_SUBJECTS = "splitsubject/test_subjects_all.txt"

FULL_LABEL_COLUMNS = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version",
    "seq_num",
    "long_title",
    "CVD_coarse_category",
    "CVD_fine_category",
    "norm_icd10_code",
    "norm_icd10_title",
    "norm_method",
    "admittime",
    "dischtime",
    "curr_stay_id",
    "deathtime",
    "dod",
    "final_death_date",
    "diff_death_days_raw",
    "next_hadm_id",
    "next_admittime",
    "next_stay_id",
    "diff_next_adm_days_raw",
    "mortality_in_hospital",
    "mortality_30d",
    "readmission_30d_hosp",
    "readmission_30d_icu",
    "cvd_list",
    "cvd_unique_count",
    "has_cxr",
    "has_ecg",
    "has_echo",
    "modality_score",
]

CORE_MODALITIES = {
    "ecg": {"time_col": "ecg_time", "study_col": "study_id_list", "path_col": "ecg_path_list"},
    "cxr": {"time_col": "cxr_time", "study_col": "study_id_list", "path_col": "cxr_path_list"},
    "echo": {"time_col": "echo_time", "study_col": "study_id_list", "path_col": "echo_path_list"},
}


def read_subjects(path: str) -> Set[int]:
    with open(path) as f:
        return {int(x.strip()) for x in f if x.strip()}


def as_int_set(values) -> Set[int]:
    return {int(x) for x in values if pd.notna(x)}


def first_study_id(value) -> str:
    values = parse_listlike(value)
    return str(values[0]) if values else ""


def list_size(value) -> int:
    return len(parse_listlike(value))


def paths_json(value) -> str:
    return json.dumps([str(x) for x in parse_listlike(value)], ensure_ascii=False)


def paths_preview(value, n: int = 3) -> str:
    values = [str(x) for x in parse_listlike(value)]
    return json.dumps(values[:n], ensure_ascii=False)


def cvd_category_count(value) -> int:
    return len({str(x).strip() for x in parse_listlike(value) if str(x).strip()})


def build_ds_cutoff(notes: pd.DataFrame) -> pd.Series:
    if notes is None or notes.empty:
        return pd.Series(dtype="datetime64[ns]", name="diagnostic_note_time")
    work = notes[notes["note_type_list"].astype(str).str.contains("DS", regex=False)].copy()
    work["diagnostic_note_time"] = pd.to_datetime(work["charttime"], errors="coerce")
    return work.groupby("hadm_id")["diagnostic_note_time"].min()


def build_selected_modality_tables(
    details: Dict[str, Optional[pd.DataFrame]],
    diagnostic_time: pd.Series,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    selected: Dict[str, pd.DataFrame] = {}
    audit_parts = []

    for mod, cfg in CORE_MODALITIES.items():
        df = details.get(mod)
        if df is None or df.empty:
            selected[mod] = pd.DataFrame()
            continue

        time_col = cfg["time_col"]
        work = df.copy()
        work["hadm_id"] = work["hadm_id"].astype(int)
        work["selected_event_time"] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.merge(
            diagnostic_time.rename("diagnostic_note_time"),
            left_on="hadm_id",
            right_index=True,
            how="left",
        )
        work["candidate_count_total"] = work.groupby("hadm_id")["hadm_id"].transform("size")
        before = work[
            work["diagnostic_note_time"].notna()
            & work["selected_event_time"].notna()
            & (work["selected_event_time"] < work["diagnostic_note_time"])
        ].copy()
        if before.empty:
            selected[mod] = before
            continue

        before["candidate_count_before_diagnosis"] = before.groupby("hadm_id")["hadm_id"].transform("size")
        before["selected_study_id"] = before[cfg["study_col"]].apply(first_study_id)
        before = before.sort_values(["hadm_id", "selected_event_time", "selected_study_id", "reference_id"]).copy()
        first = before.groupby("hadm_id", as_index=False).head(1).copy()
        first["selected_file_count"] = first[cfg["path_col"]].apply(list_size)
        first["selected_paths_preview"] = first[cfg["path_col"]].apply(paths_preview)
        first["selected_paths_all"] = first[cfg["path_col"]].apply(paths_json)
        first["modality"] = mod
        first["hours_before_diagnosis"] = (
            first["diagnostic_note_time"] - first["selected_event_time"]
        ).dt.total_seconds() / 3600.0
        first["selected_reference_id"] = first["reference_id"]
        selected[mod] = first.drop(
            columns=[
                "diagnostic_note_time",
                "selected_event_time",
                "candidate_count_total",
                "candidate_count_before_diagnosis",
                "selected_study_id",
                "selected_file_count",
                "selected_paths_preview",
                "selected_paths_all",
                "modality",
                "hours_before_diagnosis",
                "selected_reference_id",
            ],
            errors="ignore",
        )
        audit_parts.append(
            first[
                [
                    "subject_id",
                    "hadm_id",
                    "diagnostic_note_time",
                    "modality",
                    "selected_study_id",
                    "selected_event_time",
                    "hours_before_diagnosis",
                    "candidate_count_before_diagnosis",
                    "candidate_count_total",
                    "selected_file_count",
                    "selected_paths_preview",
                    "selected_paths_all",
                    "selected_reference_id",
                ]
            ].copy()
        )

    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    return selected, audit


def build_availability(labels: pd.DataFrame, audit: pd.DataFrame, allowed_subjects: Set[int]) -> pd.DataFrame:
    base = labels[labels["subject_id"].astype(int).isin(allowed_subjects)].copy()
    base["subject_id"] = base["subject_id"].astype(int)
    base["hadm_id"] = base["hadm_id"].astype(int)
    for mod in ["ecg", "cxr", "echo"]:
        valid_hadms = set(audit.loc[audit["modality"] == mod, "hadm_id"].astype(int))
        base[f"{mod}_pre_ds"] = base["hadm_id"].isin(valid_hadms)
    base["pre_ds_score"] = (
        base["ecg_pre_ds"].astype(int)
        + base["cxr_pre_ds"].astype(int)
        + base["echo_pre_ds"].astype(int)
    )
    base["dischtime_dt"] = pd.to_datetime(base["dischtime"], errors="coerce")
    return base


def choose_best_per_subject(pool: pd.DataFrame) -> pd.DataFrame:
    work = pool[pool["ecg_pre_ds"]].copy()
    return (
        work.sort_values(
            ["pre_ds_score", "echo_pre_ds", "cxr_pre_ds", "dischtime_dt", "hadm_id"],
            ascending=[False, False, False, False, True],
        )
        .drop_duplicates(subset=["subject_id"], keep="first")
        .copy()
    )


def select_priority_current(
    labels: pd.DataFrame,
    audit: pd.DataFrame,
    current_root: str,
    allowed_subjects: Set[int],
    target_subjects: int,
    target_cxr: int,
    target_echo: int,
    required_cvd_unique_count: Optional[int],
) -> pd.DataFrame:
    current_labels = pd.read_csv(Path(current_root) / "labels" / "cohort_labels_test100.csv")
    current_subjects = as_int_set(current_labels["subject_id"])
    current_hadms = as_int_set(current_labels["hadm_id"])

    availability = build_availability(labels, audit, allowed_subjects)
    if required_cvd_unique_count is not None:
        if "cvd_unique_count" not in availability.columns:
            availability["cvd_unique_count"] = availability["CVD_coarse_category"].apply(cvd_category_count)
        availability["cvd_unique_count"] = availability["cvd_unique_count"].astype(int)
        parsed_counts = availability["CVD_coarse_category"].apply(cvd_category_count)
        availability = availability[
            availability["cvd_unique_count"].eq(required_cvd_unique_count)
            & parsed_counts.eq(required_cvd_unique_count)
        ].copy()
    current_pool = availability[availability["hadm_id"].isin(current_hadms)].copy()
    current_best = choose_best_per_subject(current_pool)

    current_useful = current_best[current_best["cxr_pre_ds"] | current_best["echo_pre_ds"]].copy()
    kept = current_useful.copy()

    cxr_count = int(kept["cxr_pre_ds"].sum())
    echo_count = int(kept["echo_pre_ds"].sum())
    if cxr_count > target_cxr or echo_count > target_echo:
        raise ValueError(
            f"当前保留记录已超过目标: cxr={cxr_count}/{target_cxr}, echo={echo_count}/{target_echo}"
        )

    replacement_pool = availability[
        ~availability["subject_id"].isin(current_subjects)
        & availability["ecg_pre_ds"]
    ].copy()
    replacement_best = choose_best_per_subject(replacement_pool)
    replacement_best["source_status"] = "replacement_from_test_subjects_all"

    selected_parts = []
    kept["source_status"] = "kept_current"
    selected_parts.append(kept)
    used_subjects = set(kept["subject_id"].astype(int))

    def take_candidates(mask: pd.Series, need: int, selected: List[pd.DataFrame]) -> None:
        nonlocal used_subjects
        if need <= 0:
            return
        candidates = replacement_best[mask & ~replacement_best["subject_id"].isin(used_subjects)].copy()
        candidates = candidates.sort_values(
            ["pre_ds_score", "echo_pre_ds", "cxr_pre_ds", "dischtime_dt", "hadm_id"],
            ascending=[False, False, False, False, True],
        )
        chosen = candidates.head(need).copy()
        if len(chosen) < need:
            raise ValueError(f"候选不足: need={need}, got={len(chosen)}")
        used_subjects.update(chosen["subject_id"].astype(int).tolist())
        selected.append(chosen)

    echo_need = target_echo - int(kept["echo_pre_ds"].sum())
    take_candidates(replacement_best["echo_pre_ds"], echo_need, selected_parts)

    tmp = pd.concat(selected_parts, ignore_index=True)
    cxr_need = target_cxr - int(tmp["cxr_pre_ds"].sum())
    take_candidates(replacement_best["cxr_pre_ds"] & ~replacement_best["echo_pre_ds"], cxr_need, selected_parts)

    tmp = pd.concat(selected_parts, ignore_index=True)
    total_need = target_subjects - len(tmp)
    take_candidates(~replacement_best["cxr_pre_ds"] & ~replacement_best["echo_pre_ds"], total_need, selected_parts)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates(subset=["subject_id"], keep="first").copy()
    selected = selected.sort_values(["source_status", "subject_id", "hadm_id"]).reset_index(drop=True)

    checks = {
        "subjects": selected["subject_id"].nunique(),
        "hadms": selected["hadm_id"].nunique(),
        "ecg": int(selected["ecg_pre_ds"].sum()),
        "cxr": int(selected["cxr_pre_ds"].sum()),
        "echo": int(selected["echo_pre_ds"].sum()),
    }
    if required_cvd_unique_count is not None:
        checks["cvd_unique_count"] = int(selected["cvd_unique_count"].astype(int).eq(required_cvd_unique_count).sum())
    if checks != {
        "subjects": target_subjects,
        "hadms": target_subjects,
        "ecg": target_subjects,
        "cxr": target_cxr,
        "echo": target_echo,
        **({"cvd_unique_count": target_subjects} if required_cvd_unique_count is not None else {}),
    }:
        raise ValueError(f"最终约束不满足: {checks}")
    return selected


def filter_details(
    details: Dict[str, Optional[pd.DataFrame]],
    selected_hadms: Set[int],
    selected_modality_details: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[pd.DataFrame]]:
    out: Dict[str, Optional[pd.DataFrame]] = {}
    for mod, df in details.items():
        if df is None:
            out[mod] = None
            continue
        if mod in selected_modality_details:
            d = selected_modality_details[mod]
            out[mod] = d[d["hadm_id"].astype(int).isin(selected_hadms)].copy()
        elif "hadm_id" in df.columns:
            out[mod] = df[df["hadm_id"].astype(int).isin(selected_hadms)].copy()
        else:
            out[mod] = df.copy()
    return out


def filter_master(master: pd.DataFrame, selected_hadms: Set[int], selected_details: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = master[master["hadm_id"].astype(int).isin(selected_hadms)].copy()
    keep_refs = set()
    ref_to_type = {
        "ecg": "ECG_Recording",
        "cxr": "CXR_Imaging",
        "echo": "Echo_Imaging",
    }
    for mod, event_type in ref_to_type.items():
        d = selected_details.get(mod)
        if d is not None and not d.empty and "reference_id" in d.columns:
            keep_refs.update(d["reference_id"].astype(str).tolist())
        mask = out["event_type"].astype(str).eq(event_type)
        out = out[~mask | out["reference_id"].astype(str).isin(keep_refs)].copy()
    return out


def write_outputs(
    out_root: str,
    labels: pd.DataFrame,
    selected: pd.DataFrame,
    master: pd.DataFrame,
    details: Dict[str, Optional[pd.DataFrame]],
    audit: pd.DataFrame,
    manifest: pd.DataFrame,
    stats: Dict[str, object],
) -> None:
    dirs = ensure_dirs(out_root)
    selected_hadms = set(selected["hadm_id"].astype(int))
    selected_labels = labels[labels["hadm_id"].astype(int).isin(selected_hadms)].copy()
    selected_labels = selected_labels.merge(
        selected[["hadm_id", "ecg_pre_ds", "cxr_pre_ds", "echo_pre_ds", "source_status"]],
        on="hadm_id",
        how="left",
    )
    selected_labels["has_ecg"] = selected_labels["ecg_pre_ds"].astype(int)
    selected_labels["has_cxr"] = selected_labels["cxr_pre_ds"].astype(int)
    selected_labels["has_echo"] = selected_labels["echo_pre_ds"].astype(int)
    selected_labels["modality_score"] = selected_labels[["has_ecg", "has_cxr", "has_echo"]].sum(axis=1)
    selected_labels = selected_labels.drop(columns=["ecg_pre_ds", "cxr_pre_ds", "echo_pre_ds"])
    label_cols = FULL_LABEL_COLUMNS
    selected_labels[label_cols].to_csv(os.path.join(dirs["labels"], "cohort_labels_test100.csv"), index=False)
    selected_labels[label_cols].to_csv(
        os.path.join(dirs["labels"], "cohort_labels_test100.csv.gz"),
        index=False,
        compression="gzip",
    )

    master.to_csv(os.path.join(dirs["timeline"], "master_timeline_test100.csv.gz"), index=False, compression="gzip")
    for mod, df in details.items():
        if df is None:
            continue
        df.to_csv(os.path.join(dirs["details"], f"details_{mod}_grp_test100.csv.gz"), index=False, compression="gzip")

    audit = audit[audit["hadm_id"].astype(int).isin(selected_hadms)].copy()
    audit = audit.merge(selected[["hadm_id", "source_status"]], on="hadm_id", how="left")
    audit.to_csv(os.path.join(out_root, "modality_diagnosis_time_audit.csv"), index=False)

    pair_df = selected[["subject_id", "hadm_id", "source_status"]].copy()
    pair_df = pair_df.sort_values(["subject_id", "hadm_id"])
    pair_df.to_csv(os.path.join(out_root, "test_subject_hadm.csv"), index=False)
    with open(os.path.join(out_root, "test_subjects.txt"), "w") as f:
        for sid in sorted(pair_df["subject_id"].astype(int)):
            f.write(f"{sid}\n")
    with open(os.path.join(out_root, "test_hadms.txt"), "w") as f:
        for hid in sorted(pair_df["hadm_id"].astype(int)):
            f.write(f"{hid}\n")

    manifest.to_csv(os.path.join(out_root, "origin_manifest_test100.csv"), index=False)
    manifest.to_csv(os.path.join(out_root, "origin_manifest_test100.csv.gz"), index=False, compression="gzip")
    with open(os.path.join(out_root, "stats_test100.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def validate_manifest(out_root: str, manifest: pd.DataFrame) -> Dict[str, object]:
    out = {}
    for mod in ["ecg", "cxr", "echo"]:
        sub = manifest[manifest["modality"] == mod]
        missing = zero = present = 0
        for _, row in sub.iterrows():
            path = Path(out_root) / str(row["target_path"])
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                present += 1
            else:
                missing += 1
                if path.exists() and path.is_file() and path.stat().st_size == 0:
                    zero += 1
        out[mod] = {
            "manifest_rows": int(len(sub)),
            "present_nonempty": int(present),
            "missing_or_empty": int(missing),
            "zero_size": int(zero),
        }
    return out


def annotate_manifest_with_selected_study(manifest: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    lookup = audit[["hadm_id", "modality", "selected_study_id"]].copy()
    lookup["hadm_id"] = lookup["hadm_id"].astype(int)
    lookup["modality"] = lookup["modality"].astype(str)
    out = manifest.copy()
    out["hadm_id"] = out["hadm_id"].astype(int)
    out["modality"] = out["modality"].astype(str)
    out = out.merge(lookup, on=["hadm_id", "modality"], how="left")
    return out


def build_file_summary(manifest: pd.DataFrame, audit: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    audit_work = audit.copy()
    audit_work["selected_event_time"] = pd.to_datetime(audit_work["selected_event_time"], errors="coerce")
    audit_work["diagnostic_note_time"] = pd.to_datetime(audit_work["diagnostic_note_time"], errors="coerce")
    out["audit_all_before_diagnosis"] = bool((audit_work["selected_event_time"] < audit_work["diagnostic_note_time"]).all())
    out["one_selected_study_per_hadm_modality"] = bool(
        audit_work.groupby(["hadm_id", "modality"])["selected_study_id"].nunique().max() == 1
    )
    out["selected_study_counts"] = {
        str(k): int(v) for k, v in audit_work.groupby("modality")["selected_study_id"].nunique().items()
    }

    file_summary = {}
    for mod, sub in manifest.groupby("modality"):
        per_study = sub.groupby(["hadm_id", "selected_study_id"]).size()
        contains_study = sub.apply(
            lambda r: f"/s{str(r['selected_study_id']).strip()}/" in str(r["relative_path"]),
            axis=1,
        )
        file_summary[str(mod)] = {
            "manifest_file_count": int(len(sub)),
            "files_per_study_min": int(per_study.min()) if len(per_study) else 0,
            "files_per_study_median": float(per_study.median()) if len(per_study) else 0.0,
            "files_per_study_max": int(per_study.max()) if len(per_study) else 0,
            "manifest_paths_match_selected_study": bool(contains_study.all()) if len(sub) else True,
        }
    out["file_summary_by_modality"] = file_summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--new-output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--current-root", default=DEFAULT_CURRENT_ROOT)
    ap.add_argument("--restrict-subjects-file", default=DEFAULT_RESTRICT_SUBJECTS)
    ap.add_argument("--target-subjects", type=int, default=100)
    ap.add_argument("--target-cxr", type=int, default=60)
    ap.add_argument("--target-echo", type=int, default=40)
    ap.add_argument("--required-cvd-unique-count", type=int, default=None)
    ap.add_argument("--cxr-format", choices=["dcm", "jpg"], default="jpg")
    ap.add_argument("--cxr-base-url", default="https://physionet.org/files/mimic-cxr-jpg/2.1.0")
    ap.add_argument("--ecg-base-url", default="https://physionet.org/files/mimic-iv-ecg/1.0")
    ap.add_argument("--echo-base-url", default="https://physionet.org/files/mimic-iv-echo/0.1")
    ap.add_argument("--ecg-expand-pairs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-expand-all-dcm-by-study", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-record-list-path", default=None)
    ap.add_argument("--download-origin-data", action="store_true")
    ap.add_argument("--physionet-user", default=os.environ.get("PHYSIONET_USER"))
    ap.add_argument("--physionet-password", default=os.environ.get("PHYSIONET_PASSWORD"))
    ap.add_argument("--download-workers", type=int, default=2)
    ap.add_argument("--download-retries", type=int, default=5)
    ap.add_argument("--download-timeout-sec", type=int, default=300)
    args = ap.parse_args()

    out_root = os.path.join(args.new_output_root, args.dataset_name)
    labels, master, details = load_source(args.source_root, "step_all")
    allowed_subjects = read_subjects(args.restrict_subjects_file)
    diagnostic_time = build_ds_cutoff(details["notes"])
    selected_modality_details, audit = build_selected_modality_tables(details, diagnostic_time)
    selected = select_priority_current(
        labels=labels,
        audit=audit,
        current_root=args.current_root,
        allowed_subjects=allowed_subjects,
        target_subjects=args.target_subjects,
        target_cxr=args.target_cxr,
        target_echo=args.target_echo,
        required_cvd_unique_count=args.required_cvd_unique_count,
    )
    selected_hadms = set(selected["hadm_id"].astype(int))
    filtered_details = filter_details(details, selected_hadms, selected_modality_details)
    filtered_master = filter_master(master, selected_hadms, selected_modality_details)
    selected_labels_for_manifest = labels[labels["hadm_id"].astype(int).isin(selected_hadms)].copy()

    selected_audit = audit[audit["hadm_id"].astype(int).isin(selected_hadms)].copy()
    manifest = build_manifest(
        selected_labels=selected_labels_for_manifest,
        details=filtered_details,
        cxr_base_url=args.cxr_base_url,
        ecg_base_url=args.ecg_base_url,
        echo_base_url=args.echo_base_url,
        cxr_format=args.cxr_format,
        ecg_expand_pairs=args.ecg_expand_pairs,
        echo_expand_all=args.echo_expand_all_dcm_by_study,
        echo_record_list_path=args.echo_record_list_path,
        out_root=out_root,
    )
    manifest = annotate_manifest_with_selected_study(manifest, selected_audit)

    reuse_stats = copy_reusable_files(manifest, args.current_root, out_root)
    if args.download_origin_data:
        if not args.physionet_user or not args.physionet_password:
            raise ValueError("下载需要 --physionet-user/--physionet-password 或环境变量。")
        download_missing_or_empty_manifest(
            manifest=manifest,
            user=args.physionet_user,
            password=args.physionet_password,
            out_root=out_root,
            download_workers=args.download_workers,
            download_retries=args.download_retries,
            download_timeout_sec=args.download_timeout_sec,
        )
    else:
        Path(out_root).mkdir(parents=True, exist_ok=True)
        open(os.path.join(out_root, "download_success.txt"), "w").close()
        open(os.path.join(out_root, "download_failed.txt"), "w").close()

    manifest_status = validate_manifest(out_root, manifest)
    file_summary = build_file_summary(manifest, selected_audit)
    selected_cvd_dist = {
        str(k): int(v) for k, v in selected["cvd_unique_count"].astype(int).value_counts().sort_index().items()
    }
    stats = {
        "subject_count": int(selected["subject_id"].nunique()),
        "hadm_count": int(selected["hadm_id"].nunique()),
        "kept_current_count": int((selected["source_status"] == "kept_current").sum()),
        "replacement_count": int((selected["source_status"] == "replacement_from_test_subjects_all").sum()),
        "actual_ecg_hadm": int(selected["ecg_pre_ds"].sum()),
        "actual_cxr_hadm": int(selected["cxr_pre_ds"].sum()),
        "actual_echo_hadm": int(selected["echo_pre_ds"].sum()),
        "audit_rows": int(len(audit[audit["hadm_id"].astype(int).isin(selected_hadms)])),
        "manifest_rows": int(len(manifest)),
        "manifest_validation": manifest_status,
        "selected_cvd_unique_count_distribution": selected_cvd_dist,
        "required_cvd_unique_count": args.required_cvd_unique_count,
        "full_study_file_summary": file_summary,
        "reuse": reuse_stats,
        "diagnostic_cutoff": "earliest_DS_charttime",
        "label_source": "ICD_code",
    }
    write_outputs(out_root, labels, selected, filtered_master, filtered_details, audit, manifest, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
