import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Set

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


def cxr_report_zip_path(subject_id: int, study_id: int) -> str:
    subject = str(int(subject_id))
    return f"files/p{subject[:2]}/p{subject}/s{int(study_id)}.txt"


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def load_selected_cxr(dataset_root: Path) -> pd.DataFrame:
    details_path = dataset_root / "details" / "details_cxr_grp_test100.csv.gz"
    if not details_path.exists():
        raise FileNotFoundError(f"未找到 CXR details: {details_path}")
    cxr = pd.read_csv(details_path)
    rows: List[Dict[str, object]] = []
    for _, row in cxr.iterrows():
        study_id = first_value(row.get("study_id_list", "[]"))
        if not study_id:
            continue
        subject_id = int(row["subject_id"])
        hadm_id = int(float(row["hadm_id"]))
        rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "study_id": int(study_id),
                "cxr_time": row.get("cxr_time"),
                "reference_id": row.get("reference_id"),
                "zip_member": cxr_report_zip_path(subject_id, int(study_id)),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["subject_id", "hadm_id", "study_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("当前 test100 CXR details 中没有找到 selected CXR study")
    return out


def load_all_pairs(dataset_root: Path) -> pd.DataFrame:
    pair_path = dataset_root / "test_subject_hadm.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"未找到 subject/hadm 对应表: {pair_path}")
    pairs = pd.read_csv(pair_path)
    return pairs[["subject_id", "hadm_id"]].drop_duplicates().copy()


def extract_reports(cxr_rows: pd.DataFrame, zip_path: Path, output_root: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names: Set[str] = set(zf.namelist())
        for _, row in cxr_rows.iterrows():
            member = str(row["zip_member"])
            if member not in names:
                rows.append({**row.to_dict(), "status": "missing_in_zip", "target_path": "", "text_chars": 0, "text_bytes": 0})
                continue
            raw = zf.read(member)
            text = raw.decode("utf-8", errors="replace")
            sid = int(row["subject_id"])
            hid = int(row["hadm_id"])
            study = int(row["study_id"])
            rel_path = Path("reports") / str(sid) / str(hid) / f"s{safe_filename(str(study))}.txt"
            dst = output_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            rows.append(
                {
                    **row.to_dict(),
                    "status": "exported",
                    "target_path": str(rel_path),
                    "text_chars": len(text),
                    "text_bytes": dst.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--mimiciv-root", default=DEFAULT_MIMICIV_ROOT)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--expected-cxr-reports", type=int, default=60)
    ap.add_argument("--expected-subjects", type=int, default=100)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    mimiciv_root = Path(args.mimiciv_root)
    output_root = Path(args.output_dir) if args.output_dir else dataset_root / "cxr_reports"
    output_root.mkdir(parents=True, exist_ok=True)

    selected_cxr = load_selected_cxr(dataset_root)
    pairs = load_all_pairs(dataset_root)
    zip_path = choose_existing([mimiciv_root / "cxr" / "files.zip"])

    manifest = extract_reports(selected_cxr, zip_path, output_root)
    exported = manifest[manifest["status"].eq("exported")].copy()
    if len(exported) != int(args.expected_cxr_reports):
        missing = manifest[~manifest["status"].eq("exported")][["subject_id", "hadm_id", "study_id", "zip_member", "status"]]
        raise ValueError(
            f"导出的 CXR reports 数量异常: {len(exported)} != {args.expected_cxr_reports}; "
            f"missing={missing.head(10).to_dict('records')}"
        )

    report_table = exported.copy()
    texts = []
    for target_path in report_table["target_path"]:
        texts.append((output_root / str(target_path)).read_text(encoding="utf-8"))
    report_table["text"] = texts
    report_table["source_archive"] = str(zip_path)
    report_table.to_csv(output_root / "cxr_reports_test100.csv.gz", index=False, compression="gzip")

    manifest["source_archive"] = str(zip_path)
    manifest = manifest[
        [
            "subject_id",
            "hadm_id",
            "study_id",
            "cxr_time",
            "reference_id",
            "zip_member",
            "source_archive",
            "status",
            "target_path",
            "text_chars",
            "text_bytes",
        ]
    ]
    manifest.to_csv(output_root / "cxr_reports_manifest.csv", index=False)
    manifest.to_csv(output_root / "cxr_reports_manifest.csv.gz", index=False, compression="gzip")

    coverage = pairs.merge(
        manifest[["subject_id", "hadm_id", "study_id", "status", "target_path"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    coverage["has_selected_cxr"] = coverage["study_id"].notna()
    coverage["report_status"] = coverage["status"].fillna("no_selected_cxr")
    coverage = coverage[
        ["subject_id", "hadm_id", "has_selected_cxr", "study_id", "report_status", "target_path"]
    ].sort_values(["subject_id", "hadm_id"])
    coverage.to_csv(output_root / "cxr_report_coverage_test100.csv", index=False)

    stats = {
        "dataset_root": str(dataset_root),
        "mimiciv_root": str(mimiciv_root),
        "source_archive": str(zip_path),
        "output_root": str(output_root),
        "subject_count": int(pairs["subject_id"].nunique()),
        "hadm_count": int(pairs["hadm_id"].nunique()),
        "selected_cxr_study_count": int(len(selected_cxr)),
        "exported_report_count": int(len(exported)),
        "txt_file_count": int(len(list((output_root / "reports").glob("*/*/*.txt")))),
        "no_selected_cxr_count": int((~coverage["has_selected_cxr"]).sum()),
        "missing_report_count": int((manifest["status"] != "exported").sum()),
        "empty_report_count": int((manifest["text_bytes"].fillna(0).astype(int) == 0).sum()),
    }
    if stats["subject_count"] != int(args.expected_subjects):
        raise ValueError(f"subject 数量异常: {stats['subject_count']} != {args.expected_subjects}")
    with open(output_root / "cxr_reports_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
