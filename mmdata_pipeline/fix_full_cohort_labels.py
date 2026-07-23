import argparse
import ast
import os
import shutil
from typing import Dict, List, Optional

import pandas as pd


BASE_LABEL_COLUMNS = [
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
]

FULL_LABEL_COLUMNS = BASE_LABEL_COLUMNS + [
    "cvd_list",
    "cvd_unique_count",
    "has_cxr",
    "has_ecg",
    "has_echo",
    "modality_score",
]

DETAIL_MODALITIES = ["cxr", "ecg", "echo"]


def parse_cvd_list(val) -> List[str]:
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]
    if pd.isna(val):
        return []

    text = str(val).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [text]


def read_csv_required(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


def read_detail_csv(details_dir: str, modality: str, suffix: str) -> Optional[pd.DataFrame]:
    candidates = [
        os.path.join(details_dir, f"details_{modality}_grp_{suffix}.csv.gz"),
        os.path.join(details_dir, f"details_{modality}_grp_{suffix}.csv"),
        os.path.join(details_dir, f"details_{modality}_grp_all.csv.gz"),
        os.path.join(details_dir, f"details_{modality}_grp_all.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, usecols=["hadm_id"])
    return None


def compute_modality_coverage(details_dir: str, suffix: str) -> pd.DataFrame:
    frames = []
    for modality in DETAIL_MODALITIES:
        df = read_detail_csv(details_dir, modality, suffix)
        if df is None or df.empty or "hadm_id" not in df.columns:
            continue
        flag_col = f"has_{modality}"
        tmp = df[["hadm_id"]].dropna().drop_duplicates().copy()
        tmp["hadm_id"] = tmp["hadm_id"].astype(int)
        tmp[flag_col] = 1
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["hadm_id", "has_cxr", "has_ecg", "has_echo", "modality_score"])

    coverage = frames[0]
    for frame in frames[1:]:
        coverage = coverage.merge(frame, on="hadm_id", how="outer")

    for col in ["has_cxr", "has_ecg", "has_echo"]:
        if col not in coverage.columns:
            coverage[col] = 0
    coverage[["has_cxr", "has_ecg", "has_echo"]] = coverage[
        ["has_cxr", "has_ecg", "has_echo"]
    ].fillna(0).astype(int)
    coverage["modality_score"] = coverage[["has_cxr", "has_ecg", "has_echo"]].sum(axis=1)
    return coverage


def add_derived_columns(labels: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out["hadm_id"] = out["hadm_id"].astype(int)

    for col in ["has_cxr", "has_ecg", "has_echo", "modality_score"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out["cvd_list"] = out["CVD_coarse_category"].apply(parse_cvd_list)
    out["cvd_unique_count"] = out["cvd_list"].apply(lambda values: len(set(values)))
    out = out.merge(coverage, on="hadm_id", how="left")

    for col in ["has_cxr", "has_ecg", "has_echo", "modality_score"]:
        if col not in out.columns:
            out[col] = 0
    out[["has_cxr", "has_ecg", "has_echo", "modality_score"]] = out[
        ["has_cxr", "has_ecg", "has_echo", "modality_score"]
    ].fillna(0).astype(int)

    missing = [col for col in FULL_LABEL_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Cannot build full labels; missing columns: {missing}")
    return out[FULL_LABEL_COLUMNS].copy()


def backup_if_exists(path: str, make_backup: bool) -> None:
    if not make_backup or not os.path.exists(path):
        return
    backup_path = f"{path}.bak"
    if os.path.exists(backup_path):
        return
    shutil.copy2(path, backup_path)


def write_labels(df: pd.DataFrame, csv_path: str, gz_path: Optional[str], make_backup: bool) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    backup_if_exists(csv_path, make_backup)
    df.to_csv(csv_path, index=False)
    if gz_path:
        backup_if_exists(gz_path, make_backup)
        df.to_csv(gz_path, index=False, compression="gzip")


def build_full_step4_labels(output_root: str, suffix: str) -> pd.DataFrame:
    step1_path = os.path.join(output_root, "step1_cvd_filter", f"step_1_cvd_cohort_{suffix}.csv.gz")
    master_path = os.path.join(output_root, "step3_temporal_timeline", f"master_timeline_{suffix}.csv.gz")
    details_dir = os.path.join(output_root, "step3_temporal_timeline")

    valid_hadms = set(read_csv_required(master_path, usecols=["hadm_id"])["hadm_id"].dropna().astype(int))
    labels = read_csv_required(step1_path)
    labels = labels[labels["hadm_id"].astype(int).isin(valid_hadms)].copy()
    coverage = compute_modality_coverage(details_dir, suffix)
    return add_derived_columns(labels, coverage)


def repair_step4_labels(output_root: str, suffix: str, make_backup: bool) -> pd.DataFrame:
    labels = build_full_step4_labels(output_root, suffix)
    labels_dir = os.path.join(output_root, "step4_labels_splits", "labels")
    csv_path = os.path.join(labels_dir, f"cohort_labels_{suffix}.csv")
    gz_path = os.path.join(labels_dir, f"cohort_labels_{suffix}.csv.gz")
    write_labels(labels, csv_path, gz_path, make_backup)
    return labels


def repair_test100_labels(
    test100_root: str,
    full_labels: pd.DataFrame,
    suffix: str,
    make_backup: bool,
) -> pd.DataFrame:
    labels_dir = os.path.join(test100_root, "labels")
    current_candidates = [
        os.path.join(labels_dir, "cohort_labels_test100.csv.gz"),
        os.path.join(labels_dir, "cohort_labels_test100.csv"),
    ]
    current_path = next((path for path in current_candidates if os.path.exists(path)), None)
    if current_path is None:
        raise FileNotFoundError(f"No current test100 label file found under: {labels_dir}")

    selected_hadms = set(read_csv_required(current_path, usecols=["hadm_id"])["hadm_id"].dropna().astype(int))
    test_labels = full_labels[full_labels["hadm_id"].astype(int).isin(selected_hadms)].copy()

    if test_labels["hadm_id"].nunique() != len(selected_hadms):
        found = set(test_labels["hadm_id"].astype(int))
        missing = sorted(selected_hadms - found)
        raise ValueError(f"Full labels do not cover all current test100 hadm_id values: {missing[:20]}")

    details_dir = os.path.join(test100_root, "details")
    if os.path.isdir(details_dir):
        coverage = compute_modality_coverage(details_dir, "test100")
        test_labels = add_derived_columns(test_labels[BASE_LABEL_COLUMNS], coverage)
    else:
        test_labels = test_labels[FULL_LABEL_COLUMNS].copy()

    csv_path = os.path.join(labels_dir, "cohort_labels_test100.csv")
    gz_path = os.path.join(labels_dir, "cohort_labels_test100.csv.gz")
    write_labels(test_labels, csv_path, gz_path, make_backup)
    return test_labels


def validate_full_labels(df: pd.DataFrame, name: str) -> Dict[str, int]:
    if list(df.columns) != FULL_LABEL_COLUMNS:
        raise ValueError(f"{name} columns do not match FULL_LABEL_COLUMNS")
    if not (df["has_cxr"] + df["has_ecg"] + df["has_echo"]).equals(df["modality_score"]):
        raise ValueError(f"{name} has inconsistent modality_score values")
    expected_counts = df["cvd_list"].apply(lambda values: len(set(parse_cvd_list(values))))
    if not expected_counts.equals(df["cvd_unique_count"].astype(int)):
        raise ValueError(f"{name} has inconsistent cvd_unique_count values")
    return {
        "rows": int(len(df)),
        "hadm_id": int(df["hadm_id"].nunique()),
        "columns": int(len(df.columns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild full Step4 labels and repair cohort_labels_test100 without changing the main pipeline script."
    )
    parser.add_argument(
        "--output-root",
        default="/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10",
        help="Root containing step1_cvd_filter, step3_temporal_timeline, and step4_labels_splits.",
    )
    parser.add_argument(
        "--test100-root",
        default="/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/subset2_test100_balanced",
        help="Root containing the current test100 labels/details directories.",
    )
    parser.add_argument("--suffix", default="all", help="Pipeline suffix, usually all or debug.")
    parser.add_argument("--skip-step4", action="store_true", help="Do not rewrite Step4 labels.")
    parser.add_argument("--skip-test100", action="store_true", help="Do not rewrite test100 labels.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files before overwriting.")
    args = parser.parse_args()

    make_backup = not args.no_backup
    full_labels = build_full_step4_labels(args.output_root, args.suffix)

    if not args.skip_step4:
        full_labels = repair_step4_labels(args.output_root, args.suffix, make_backup)
        print(f"step4: {validate_full_labels(full_labels, 'step4')}")

    if not args.skip_test100:
        test100_labels = repair_test100_labels(args.test100_root, full_labels, args.suffix, make_backup)
        print(f"test100: {validate_full_labels(test100_labels, 'test100')}")


if __name__ == "__main__":
    main()
