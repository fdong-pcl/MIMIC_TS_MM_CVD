import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from add_test100_seqseverity_labels import DEFAULT_HOSP_ROOT, DEFAULT_MAPPING_ROOT
from build_midterm_simple_complex100 import (
    DEFAULT_EXTERNAL_CXR_JPG_ROOT,
    DEFAULT_EXTERNAL_ECG_ZIP,
    DEFAULT_EXTERNAL_ECHO_ROOT,
    DEFAULT_MIMICIV_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REUSE_ROOTS,
    DEFAULT_SOURCE_ROOT,
    add_pre_ds_availability,
    build_disease_pool,
    select_balanced,
    write_dataset,
)
from build_test100_multimodal_compact import load_source
from build_test100_pre_ds_onestudy_priority_current import (
    build_ds_cutoff,
    build_selected_modality_tables,
    read_subjects,
)


DEFAULT_DATASET_NAME = "midterm_standard100_pre_ds_cvd3"
DEFAULT_SPLITS_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10/step4_labels_splits/splits"


STANDARD_SPEC = {
    "dataset_name": DEFAULT_DATASET_NAME,
    "target_echo": 40,
    "target_cxr": 60,
    "disease_rule": "three_cvd_categories_primary_seq1_cvd",
    "expected_cvd_unique_count": 3,
}


def test_subjects(splits_root: Path) -> Set[int]:
    return read_subjects(str(splits_root / "test_subjects_all.txt"))


def add_cvd_role_counts(facts: pd.DataFrame, disease_details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject_id, hadm_id), group in disease_details.groupby(["subject_id", "hadm_id"], sort=False):
        min_seq = group.groupby("CVD_coarse_category")["seq_num"].min()
        rows.append(
            {
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "primary_cvd_count_by_coarse": int((min_seq.astype(int) == 1).sum()),
                "comorbidity_cvd_count_by_coarse": int((min_seq.astype(int) != 1).sum()),
            }
        )
    role_counts = pd.DataFrame(rows)
    return facts.merge(role_counts, on=["subject_id", "hadm_id"], how="left")


def make_standard_pool(base: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    pool = base.merge(facts, on=["subject_id", "hadm_id"], how="inner")
    return pool[
        pool["ecg_pre_ds"]
        & pool["cvd_unique_count_detail"].astype(int).eq(3)
        & pool["primary_cvd_count_by_coarse"].astype(int).eq(1)
        & pool["comorbidity_cvd_count_by_coarse"].astype(int).eq(2)
    ].copy()


def select_standard(pool: pd.DataFrame) -> pd.DataFrame:
    selected = select_balanced(pool, target_echo=40, target_cxr=60, target_total=100).copy()
    selected["use_ecg"] = True
    selected["use_echo"] = selected["echo_pre_ds"].astype(bool)
    selected["use_cxr"] = selected["cxr_pre_ds"].astype(bool) & ~selected["use_echo"]
    if int(selected["use_ecg"].sum()) != 100:
        raise ValueError("Standard: ECG 选择数量不是 100")
    if int(selected["use_echo"].sum()) != 40:
        raise ValueError("Standard: Echo 选择数量不是 40")
    if int(selected["use_cxr"].sum()) != 60:
        raise ValueError("Standard: CXR 选择数量不是 60")
    if (selected["use_cxr"] & selected["use_echo"]).any():
        raise ValueError("Standard: CXR/Echo 组合不是互斥的")
    return selected


def run_exports_standard(out_root: Path, mimiciv_root: str) -> Dict[str, object]:
    py = os.environ.get("PYTHON", "/Users/fandong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
    script_dir = Path(__file__).resolve().parent
    commands = [
        [py, str(script_dir / "export_test100_ds_notes.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-notes", "100"],
        [py, str(script_dir / "export_test100_cxr_reports.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-cxr-reports", "60"],
        [py, str(script_dir / "export_test100_echo_measurements.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-echo-studies", "40"],
        [py, str(script_dir / "export_test100_ecg_measurements.py"), "--dataset-root", str(out_root), "--mimiciv-root", mimiciv_root, "--expected-ecg-studies", "100"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)

    stats = {}
    for name, rel in [
        ("ecg_measurements", "ecg_measurements/ecg_measurements_stats.json"),
        ("echo_measurements", "echo_measurements/echo_measurements_stats.json"),
        ("ds_notes", "ds_notes/ds_notes_stats.json"),
        ("cxr_reports", "cxr_reports/cxr_reports_stats.json"),
    ]:
        path = out_root / rel
        if path.exists():
            stats[name] = json.loads(path.read_text())
    return stats


def validate_standard_outputs(out_root: Path) -> Dict[str, object]:
    labels = pd.read_csv(out_root / "labels" / "cohort_labels_test100.csv")
    seq = pd.read_csv(out_root / "labels" / "cohort_labels_test100_seqseverity.csv")
    details = pd.read_csv(out_root / "labels" / "cvd_diagnosis_seq_details_test100.csv")
    audit = pd.read_csv(out_root / "modality_diagnosis_time_audit.csv")
    manifest = pd.read_csv(out_root / "origin_manifest_test100.csv")

    event_time = pd.to_datetime(audit["selected_event_time"], errors="coerce")
    diagnostic_time = pd.to_datetime(audit["diagnostic_note_time"], errors="coerce")
    missing = zero = 0
    for target in manifest["target_path"]:
        path = out_root / str(target)
        if not path.exists():
            missing += 1
        elif path.stat().st_size <= 0:
            zero += 1

    checks = {
        "subject_count": int(labels["subject_id"].nunique()),
        "hadm_count": int(labels["hadm_id"].nunique()),
        "one_hadm_per_subject": bool(labels.groupby("subject_id")["hadm_id"].nunique().max() == 1),
        "has_ecg_sum": int(labels["has_ecg"].sum()),
        "has_cxr_sum": int(labels["has_cxr"].sum()),
        "has_echo_sum": int(labels["has_echo"].sum()),
        "cvd_unique_count_is_3": bool(labels["cvd_unique_count"].astype(int).eq(3).all()),
        "seq_primary_count_is_1": bool(seq["primary_cvd_count"].astype(int).eq(1).all()),
        "seq_comorbidity_count_is_2": bool(seq["comorbidity_cvd_count"].astype(int).eq(2).all()),
        "detail_roles": {str(k): int(v) for k, v in details["diagnosis_role"].value_counts().items()},
        "all_selected_events_before_diagnosis": bool((event_time < diagnostic_time).all()),
        "one_selected_study_per_hadm_modality": bool(audit.groupby(["hadm_id", "modality"])["selected_study_id"].nunique().max() == 1),
        "manifest_rows": int(len(manifest)),
        "manifest_missing_files": int(missing),
        "manifest_zero_size_files": int(zero),
    }
    expected = {
        "subject_count": 100,
        "hadm_count": 100,
        "one_hadm_per_subject": True,
        "has_ecg_sum": 100,
        "has_cxr_sum": 60,
        "has_echo_sum": 40,
        "cvd_unique_count_is_3": True,
        "seq_primary_count_is_1": True,
        "seq_comorbidity_count_is_2": True,
        "all_selected_events_before_diagnosis": True,
        "one_selected_study_per_hadm_modality": True,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise ValueError(f"{key} 校验失败: {checks[key]} != {value}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--splits-root", default=DEFAULT_SPLITS_ROOT)
    ap.add_argument("--hosp-root", default=DEFAULT_HOSP_ROOT)
    ap.add_argument("--mapping-root", default=DEFAULT_MAPPING_ROOT)
    ap.add_argument("--mimiciv-root", default=DEFAULT_MIMICIV_ROOT)
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
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    labels, master, details = load_source(args.source_root, "step_all")
    allowed = test_subjects(Path(args.splits_root))
    diagnostic_time = build_ds_cutoff(details["notes"])
    selected_modality_details, audit = build_selected_modality_tables(details, diagnostic_time)
    base = add_pre_ds_availability(labels, audit, allowed)
    facts, disease_details, gems_meta = build_disease_pool(
        base[["subject_id", "hadm_id", "cvd_unique_count", "CVD_coarse_category", "seq_num", "icd_code"]],
        Path(args.hosp_root),
        Path(args.mapping_root),
    )
    facts = add_cvd_role_counts(facts, disease_details)
    standard_pool = make_standard_pool(base, facts)
    selected = select_standard(standard_pool)

    out_root = Path(args.output_root) / args.dataset_name
    if out_root.exists() and args.overwrite:
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
        spec=STANDARD_SPEC,
    )
    export_stats = run_exports_standard(out_root, args.mimiciv_root)
    validation = validate_standard_outputs(out_root)

    stats["selection_scope"] = "test_subjects_all"
    stats["target_ecg_hadm"] = 100
    stats["target_cxr_hadm"] = 60
    stats["target_echo_hadm"] = 40
    stats["candidate_count"] = int(len(standard_pool))
    stats["candidate_subject_count"] = int(standard_pool["subject_id"].nunique())
    stats["auxiliary_exports"] = export_stats
    stats["standard_validation"] = validation
    (out_root / "stats_test100.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
