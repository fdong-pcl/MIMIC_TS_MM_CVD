import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from build_test100_multimodal_compact import parse_listlike


DEFAULT_DATASET_ROOT = (
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/"
    "midterm_standard100_pre_ds_cvd3"
)
DEFAULT_MIMICIV_ROOT = "/Users/fandong/Desktop/pcl/Data/mimiciv"


def choose_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("未找到可用源文件: " + ", ".join(str(p) for p in paths))


def first_value(value) -> str:
    values = parse_listlike(value)
    return str(values[0]) if values else ""


def load_selected_ecg(dataset_root: Path) -> pd.DataFrame:
    details_path = dataset_root / "details" / "details_ecg_grp_test100.csv.gz"
    if not details_path.exists():
        raise FileNotFoundError(f"未找到 ECG details: {details_path}")
    ecg = pd.read_csv(details_path)
    rows: List[Dict[str, object]] = []
    for _, row in ecg.iterrows():
        study_id = first_value(row.get("study_id_list", "[]"))
        if not study_id:
            continue
        rows.append(
            {
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(float(row["hadm_id"])),
                "study_id": int(study_id),
                "ecg_time": row.get("ecg_time"),
                "reference_id": row.get("reference_id"),
            }
        )
    selected = pd.DataFrame(rows).drop_duplicates(subset=["subject_id", "hadm_id", "study_id"]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("当前 test100 ECG details 中没有找到 selected ECG study")
    return selected


def load_all_pairs(dataset_root: Path) -> pd.DataFrame:
    pair_path = dataset_root / "test_subject_hadm.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"未找到 subject/hadm 对应表: {pair_path}")
    pairs = pd.read_csv(pair_path)
    return pairs[["subject_id", "hadm_id"]].drop_duplicates().copy()


def write_per_study_files(selected: pd.DataFrame, measurements: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        sid = int(row["subject_id"])
        hid = int(row["hadm_id"])
        study_id = int(row["study_id"])
        part = measurements[measurements["study_id"].astype(int).eq(study_id)].copy()
        if part.empty:
            rows.append({**row.to_dict(), "status": "missing_measurement", "target_path": "", "measurement_rows": 0})
            continue
        rel_path = Path("measurements") / str(sid) / str(hid) / f"s{study_id}_machine_measurements.csv"
        dst = output_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(dst, index=False)
        rows.append(
            {
                **row.to_dict(),
                "status": "exported",
                "target_path": str(rel_path),
                "measurement_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--mimiciv-root", default=DEFAULT_MIMICIV_ROOT)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--expected-ecg-studies", type=int, default=100)
    ap.add_argument("--expected-subjects", type=int, default=100)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    mimiciv_root = Path(args.mimiciv_root)
    output_root = Path(args.output_dir) if args.output_dir else dataset_root / "ecg_measurements"
    output_root.mkdir(parents=True, exist_ok=True)

    selected = load_selected_ecg(dataset_root)
    pairs = load_all_pairs(dataset_root)
    measurement_path = choose_existing([mimiciv_root / "ecg" / "machine_measurements.csv"])
    use_subjects = set(selected["subject_id"].astype(int))
    use_studies = set(selected["study_id"].astype(int))
    measurements = pd.read_csv(measurement_path)
    measurements = measurements[
        measurements["subject_id"].astype(int).isin(use_subjects)
        & measurements["study_id"].astype(int).isin(use_studies)
    ].copy()

    manifest = write_per_study_files(selected, measurements, output_root)
    exported = manifest[manifest["status"].eq("exported")].copy()
    if len(exported) != int(args.expected_ecg_studies):
        missing = manifest[~manifest["status"].eq("exported")][["subject_id", "hadm_id", "study_id", "status"]]
        raise ValueError(
            f"导出的 ECG measurement 数量异常: {len(exported)} != {args.expected_ecg_studies}; "
            f"missing={missing.head(10).to_dict('records')}"
        )

    selected_measurements = measurements[
        measurements["study_id"].astype(int).isin(set(exported["study_id"].astype(int)))
    ].copy()
    selected_measurements = selected_measurements.merge(
        exported[["subject_id", "hadm_id", "study_id", "ecg_time", "reference_id"]],
        on=["subject_id", "study_id"],
        how="left",
    )
    selected_measurements.to_csv(output_root / "ecg_machine_measurements_test100.csv.gz", index=False, compression="gzip")

    manifest["source_file"] = str(measurement_path)
    manifest = manifest[
        [
            "subject_id",
            "hadm_id",
            "study_id",
            "ecg_time",
            "reference_id",
            "measurement_rows",
            "source_file",
            "status",
            "target_path",
        ]
    ].sort_values(["subject_id", "hadm_id", "study_id"])
    manifest.to_csv(output_root / "ecg_measurements_manifest.csv", index=False)
    manifest.to_csv(output_root / "ecg_measurements_manifest.csv.gz", index=False, compression="gzip")

    coverage = pairs.merge(
        manifest[["subject_id", "hadm_id", "study_id", "status", "target_path"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    coverage["has_selected_ecg"] = coverage["study_id"].notna()
    coverage["measurement_status"] = coverage["status"].fillna("no_selected_ecg")
    coverage = coverage[
        ["subject_id", "hadm_id", "has_selected_ecg", "study_id", "measurement_status", "target_path"]
    ].sort_values(["subject_id", "hadm_id"])
    coverage.to_csv(output_root / "ecg_measurement_coverage_test100.csv", index=False)

    stats = {
        "dataset_root": str(dataset_root),
        "mimiciv_root": str(mimiciv_root),
        "source_file": str(measurement_path),
        "output_root": str(output_root),
        "subject_count": int(pairs["subject_id"].nunique()),
        "hadm_count": int(pairs["hadm_id"].nunique()),
        "selected_ecg_study_count": int(len(selected)),
        "exported_measurement_study_count": int(len(exported)),
        "selected_measurement_row_count": int(len(selected_measurements)),
        "per_study_csv_count": int(len(list((output_root / "measurements").glob("*/*/*.csv")))),
        "no_selected_ecg_count": int((~coverage["has_selected_ecg"]).sum()),
        "missing_measurement_count": int((manifest["status"] != "exported").sum()),
    }
    if stats["subject_count"] != int(args.expected_subjects):
        raise ValueError(f"subject 数量异常: {stats['subject_count']} != {args.expected_subjects}")
    with open(output_root / "ecg_measurements_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
