import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from build_test100_multimodal_compact import parse_listlike


DEFAULT_DATASET_ROOT = (
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/"
    "subset2_test100_subject100_pre_ds_onestudy_cvd3_fullstudy"
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


def load_selected_echo(dataset_root: Path, mimiciv_root: Path) -> pd.DataFrame:
    details_path = dataset_root / "details" / "details_echo_grp_test100.csv.gz"
    if not details_path.exists():
        raise FileNotFoundError(f"未找到 Echo details: {details_path}")
    echo = pd.read_csv(details_path)
    rows: List[Dict[str, object]] = []
    for _, row in echo.iterrows():
        study_id = first_value(row.get("study_id_list", "[]"))
        if not study_id:
            continue
        rows.append(
            {
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(float(row["hadm_id"])),
                "study_id": int(study_id),
                "echo_time": row.get("echo_time"),
                "reference_id": row.get("reference_id"),
            }
        )
    selected = pd.DataFrame(rows).drop_duplicates(subset=["subject_id", "hadm_id", "study_id"]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("当前 test100 Echo details 中没有找到 selected Echo study")

    study_path = choose_existing([mimiciv_root / "echo" / "echo-study-list.csv"])
    study = pd.read_csv(study_path)
    selected = selected.merge(
        study[["subject_id", "study_id", "study_datetime", "note_id", "note_seq", "note_charttime"]],
        on=["subject_id", "study_id"],
        how="left",
    )
    if selected["study_datetime"].isna().any():
        missing = selected[selected["study_datetime"].isna()][["subject_id", "hadm_id", "study_id"]]
        raise ValueError(f"有 selected Echo study 未在 echo-study-list 中找到: {missing.to_dict('records')[:10]}")
    return selected


def load_all_pairs(dataset_root: Path) -> pd.DataFrame:
    pair_path = dataset_root / "test_subject_hadm.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"未找到 subject/hadm 对应表: {pair_path}")
    pairs = pd.read_csv(pair_path)
    return pairs[["subject_id", "hadm_id"]].drop_duplicates().copy()


def match_measurement_ids(selected: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    selected["study_dt"] = pd.to_datetime(selected["study_datetime"], errors="coerce")
    measurement_events = measurements[["subject_id", "measurement_id", "measurement_datetime"]].drop_duplicates().copy()
    measurement_events["measurement_dt"] = pd.to_datetime(measurement_events["measurement_datetime"], errors="coerce")

    rows = []
    for _, row in selected.iterrows():
        candidates = measurement_events[measurement_events["subject_id"].astype(int).eq(int(row["subject_id"]))].copy()
        if candidates.empty:
            rows.append({**row.to_dict(), "measurement_id": None, "measurement_datetime": None, "match_abs_minutes": None, "measurement_candidate_count": 0})
            continue
        candidates["match_abs_minutes"] = (
            candidates["measurement_dt"] - row["study_dt"]
        ).abs().dt.total_seconds() / 60.0
        best = candidates.sort_values(["match_abs_minutes", "measurement_id"]).iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "measurement_id": int(best["measurement_id"]),
                "measurement_datetime": best["measurement_datetime"],
                "match_abs_minutes": float(best["match_abs_minutes"]),
                "measurement_candidate_count": int(len(candidates)),
            }
        )
    return pd.DataFrame(rows)


def write_per_study_files(selected: pd.DataFrame, measurements: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        if pd.isna(row.get("measurement_id")):
            rows.append({**row.to_dict(), "status": "missing_measurement", "target_path": "", "measurement_rows": 0})
            continue
        measurement_id = int(row["measurement_id"])
        part = measurements[measurements["measurement_id"].astype(int).eq(measurement_id)].copy()
        sid = int(row["subject_id"])
        hid = int(row["hadm_id"])
        study_id = int(row["study_id"])
        rel_path = Path("measurements") / str(sid) / str(hid) / f"s{study_id}_measurement_{measurement_id}.csv"
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
    ap.add_argument("--expected-echo-studies", type=int, default=40)
    ap.add_argument("--expected-subjects", type=int, default=100)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    mimiciv_root = Path(args.mimiciv_root)
    output_root = Path(args.output_dir) if args.output_dir else dataset_root / "echo_measurements"
    output_root.mkdir(parents=True, exist_ok=True)

    selected = load_selected_echo(dataset_root, mimiciv_root)
    pairs = load_all_pairs(dataset_root)
    measurement_path = choose_existing([mimiciv_root / "echo" / "structured-measurement.csv.gz"])
    measurements = pd.read_csv(measurement_path)
    measurements = measurements[measurements["subject_id"].astype(int).isin(set(selected["subject_id"].astype(int)))].copy()

    selected = match_measurement_ids(selected, measurements)
    manifest = write_per_study_files(selected, measurements, output_root)
    exported = manifest[manifest["status"].eq("exported")].copy()
    if len(exported) != int(args.expected_echo_studies):
        missing = manifest[~manifest["status"].eq("exported")][["subject_id", "hadm_id", "study_id", "status"]]
        raise ValueError(
            f"导出的 Echo measurement 数量异常: {len(exported)} != {args.expected_echo_studies}; "
            f"missing={missing.head(10).to_dict('records')}"
        )

    selected_measurements = measurements[measurements["measurement_id"].astype(int).isin(set(exported["measurement_id"].astype(int)))].copy()
    selected_measurements = selected_measurements.merge(
        exported[
            [
                "subject_id",
                "hadm_id",
                "study_id",
                "study_datetime",
                "measurement_id",
                "measurement_datetime",
                "match_abs_minutes",
                "reference_id",
            ]
        ],
        on=["subject_id", "measurement_id", "measurement_datetime"],
        how="left",
    )
    selected_measurements.to_csv(output_root / "echo_structured_measurements_test100.csv.gz", index=False, compression="gzip")

    manifest["source_file"] = str(measurement_path)
    manifest = manifest[
        [
            "subject_id",
            "hadm_id",
            "study_id",
            "study_datetime",
            "echo_time",
            "note_id",
            "note_seq",
            "note_charttime",
            "reference_id",
            "measurement_id",
            "measurement_datetime",
            "match_abs_minutes",
            "measurement_candidate_count",
            "measurement_rows",
            "source_file",
            "status",
            "target_path",
        ]
    ].sort_values(["subject_id", "hadm_id", "study_id"])
    manifest.to_csv(output_root / "echo_measurements_manifest.csv", index=False)
    manifest.to_csv(output_root / "echo_measurements_manifest.csv.gz", index=False, compression="gzip")

    coverage = pairs.merge(
        manifest[["subject_id", "hadm_id", "study_id", "measurement_id", "status", "target_path"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    coverage["has_selected_echo"] = coverage["study_id"].notna()
    coverage["measurement_status"] = coverage["status"].fillna("no_selected_echo")
    coverage = coverage[
        ["subject_id", "hadm_id", "has_selected_echo", "study_id", "measurement_id", "measurement_status", "target_path"]
    ].sort_values(["subject_id", "hadm_id"])
    coverage.to_csv(output_root / "echo_measurement_coverage_test100.csv", index=False)

    stats = {
        "dataset_root": str(dataset_root),
        "mimiciv_root": str(mimiciv_root),
        "source_file": str(measurement_path),
        "output_root": str(output_root),
        "subject_count": int(pairs["subject_id"].nunique()),
        "hadm_count": int(pairs["hadm_id"].nunique()),
        "selected_echo_study_count": int(len(selected)),
        "exported_measurement_study_count": int(len(exported)),
        "selected_measurement_row_count": int(len(selected_measurements)),
        "per_study_csv_count": int(len(list((output_root / "measurements").glob("*/*/*.csv")))),
        "no_selected_echo_count": int((~coverage["has_selected_echo"]).sum()),
        "missing_measurement_count": int((manifest["status"] != "exported").sum()),
        "match_abs_minutes_min": float(exported["match_abs_minutes"].min()),
        "match_abs_minutes_median": float(exported["match_abs_minutes"].median()),
        "match_abs_minutes_max": float(exported["match_abs_minutes"].max()),
    }
    if stats["subject_count"] != int(args.expected_subjects):
        raise ValueError(f"subject 数量异常: {stats['subject_count']} != {args.expected_subjects}")
    with open(output_root / "echo_measurements_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
