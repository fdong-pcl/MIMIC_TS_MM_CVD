import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from build_test100_multimodal_compact import (
    _build_balanced_cvd_targets,
    _collect_hadm_modality_paths,
    _estimate_reuse_score_for_hadm,
    _get_primary_cvd,
    build_manifest,
    ensure_dirs,
    hadm_sets,
    load_source,
)


DEFAULT_SOURCE_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10"
DEFAULT_OUTPUT_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact"
DEFAULT_DATASET_NAME = "subset2_test100_subject100_balanced"
DEFAULT_EXISTING_DATASET_ROOT = (
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/"
    "subset2_test100_balanced"
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


def read_subjects(path: str) -> Set[int]:
    with open(path) as f:
        return {int(x.strip()) for x in f if x.strip()}


def existing_filename_index(origin_root: Optional[str]) -> Dict[str, Set[str]]:
    idx = {"cxr": set(), "ecg": set(), "echo": set()}
    if not origin_root:
        return idx
    root = Path(origin_root)
    if root.name != "origin":
        root = root / "origin"
    for mod in idx:
        mod_root = root / mod
        if not mod_root.exists():
            continue
        for p in mod_root.rglob("*"):
            if p.is_file() and p.stat().st_size > 0:
                idx[mod].add(p.name)
    return idx


def copy_reusable_files(manifest: pd.DataFrame, existing_dataset_root: Optional[str], out_root: str) -> Dict[str, int]:
    stats = {"linked": 0, "copied": 0, "exists": 0, "missing": 0}
    if manifest.empty:
        return stats

    source_index: Dict[Tuple[str, str], Path] = {}
    if existing_dataset_root:
        old_origin = Path(existing_dataset_root)
        if old_origin.name != "origin":
            old_origin = old_origin / "origin"
        for mod in ["cxr", "ecg", "echo"]:
            mod_root = old_origin / mod
            if not mod_root.exists():
                continue
            for p in mod_root.rglob("*"):
                if p.is_file() and p.stat().st_size > 0:
                    source_index.setdefault((mod, p.name), p)

    for _, row in manifest.iterrows():
        dst = Path(out_root) / str(row["target_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.is_file() and dst.stat().st_size > 0:
            stats["exists"] += 1
            continue

        src = source_index.get((str(row["modality"]), dst.name))
        if src is None:
            stats["missing"] += 1
            continue

        try:
            os.link(src, dst)
            stats["linked"] += 1
        except Exception:
            shutil.copy2(src, dst)
            stats["copied"] += 1
    return stats


def download_missing_or_empty_manifest(
    manifest: pd.DataFrame,
    user: str,
    password: str,
    out_root: str,
    download_workers: int,
    download_retries: int,
    download_timeout_sec: int,
) -> None:
    succ = os.path.join(out_root, "download_success.txt")
    fail = os.path.join(out_root, "download_failed.txt")
    open(succ, "w").close()
    open(fail, "w").close()

    rows = []
    for row in manifest.to_dict("records"):
        dst = Path(out_root) / str(row["target_path"])
        if dst.exists() and dst.is_file() and dst.stat().st_size > 0:
            continue
        rows.append(row)

    total = len(rows)
    done = success = failed = 0
    lock = threading.Lock()
    last_print_ts = 0.0
    start_ts = time.time()

    def append_line(path: str, text: str) -> None:
        with open(path, "a") as f:
            f.write(text)

    def run_one(row: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        dst = Path(out_root) / str(row["target_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.is_file() and dst.stat().st_size == 0:
            dst.unlink()

        cmd = [
            "wget",
            "-c",
            "-np",
            "-O",
            str(dst),
            "--user",
            str(user),
            "--password",
            str(password),
            str(row["url"]),
        ]
        tries = max(1, int(download_retries) + 1)
        for _ in range(tries):
            ret = subprocess.run(cmd, timeout=int(download_timeout_sec)).returncode
            if ret == 0 and dst.exists() and dst.stat().st_size > 0:
                return "success", row
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink()
        return "failed", row

    def print_progress(force: bool = False) -> None:
        nonlocal last_print_ts
        now = time.time()
        if force or now - last_print_ts >= 5.0:
            last_print_ts = now
            print(f"[download-missing] done={done} total={total} success={success} failed={failed} remaining={total-done}")

    if total == 0:
        print("[download-missing] done=0 total=0 success=0 failed=0 remaining=0")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(download_workers))) as executor:
        futures = [executor.submit(run_one, row) for row in rows]
        for fut in concurrent.futures.as_completed(futures):
            status, row = fut.result()
            line = f"{row['modality']}\t{row['url']}\n"
            with lock:
                done += 1
                if status == "success":
                    success += 1
                    append_line(succ, line)
                else:
                    failed += 1
                    append_line(fail, line)
                print_progress()

    print_progress(force=True)
    elapsed = max(time.time() - start_ts, 1e-6)
    print(f"[download-missing-summary] elapsed_sec={elapsed:.1f} throughput={total/elapsed:.2f} items/sec")


def prepare_one_hadm_per_subject_pool(
    labels: pd.DataFrame,
    details: Dict[str, Optional[pd.DataFrame]],
    allowed_subjects: Set[int],
    existing_origin_root: Optional[str],
    cxr_format: str,
    ecg_expand_pairs: bool,
) -> pd.DataFrame:
    base = labels[labels["subject_id"].astype(int).isin(allowed_subjects)].copy()
    for col in ["has_ecg", "has_cxr", "has_echo", "modality_score"]:
        if col not in base.columns:
            hs = hadm_sets(details)
            base["has_ecg"] = base["hadm_id"].astype(int).isin(hs["ecg"]).astype(int)
            base["has_cxr"] = base["hadm_id"].astype(int).isin(hs["cxr"]).astype(int)
            base["has_echo"] = base["hadm_id"].astype(int).isin(hs["echo"]).astype(int)
            base["modality_score"] = base[["has_ecg", "has_cxr", "has_echo"]].sum(axis=1)
            break

    base = base[base["has_ecg"].astype(int) == 1].copy()
    base["subject_id"] = base["subject_id"].astype(int)
    base["hadm_id"] = base["hadm_id"].astype(int)
    base["has_cxr"] = base["has_cxr"].fillna(0).astype(int)
    base["has_echo"] = base["has_echo"].fillna(0).astype(int)
    base["modality_score"] = base["modality_score"].fillna(0).astype(int)
    base["primary_cvd"] = base["CVD_coarse_category"].apply(_get_primary_cvd)
    if "cvd_unique_count" not in base.columns:
        base["cvd_unique_count"] = base["CVD_coarse_category"].apply(lambda x: len(set(str(x).split(","))))
    base["cvd_unique_count"] = base["cvd_unique_count"].astype(int)
    base["dischtime_dt"] = pd.to_datetime(base["dischtime"], errors="coerce")

    hadm_paths = {
        "cxr": _collect_hadm_modality_paths(details, "cxr", "cxr_path_list"),
        "ecg": _collect_hadm_modality_paths(details, "ecg", "ecg_path_list"),
        "echo": _collect_hadm_modality_paths(details, "echo", "echo_path_list"),
    }
    existing_idx = existing_filename_index(existing_origin_root)
    base["reuse_score"] = base["hadm_id"].map(
        lambda hid: _estimate_reuse_score_for_hadm(
            int(hid),
            hadm_paths,
            existing_idx,
            cxr_format=cxr_format,
            ecg_expand_pairs=ecg_expand_pairs,
        )
    )

    base = base.sort_values(
        ["modality_score", "has_echo", "has_cxr", "reuse_score", "dischtime_dt", "hadm_id"],
        ascending=[False, False, False, False, False, True],
    )
    return base.drop_duplicates(subset=["subject_id"], keep="first").copy()


def score_rows_for_cvd_balance(pool: pd.DataFrame, cvd_targets: Dict[str, int], cvd_counts: Dict[str, int]) -> pd.DataFrame:
    work = pool.copy()
    work["cvd_gap"] = work["primary_cvd"].map(lambda c: max(int(cvd_targets.get(c, 0)) - int(cvd_counts.get(c, 0)), 0))
    work["cvd_need"] = (work["cvd_gap"] > 0).astype(int)
    return work.sort_values(
        ["cvd_need", "cvd_gap", "cvd_unique_count", "modality_score", "reuse_score", "dischtime_dt", "hadm_id"],
        ascending=[False, False, True, False, False, False, True],
    )


def choose_from_group(
    group: pd.DataFrame,
    n: int,
    cvd_targets: Dict[str, int],
    cvd_counts: Dict[str, int],
    used_subjects: Set[int],
) -> pd.DataFrame:
    chosen = []
    remaining = group[~group["subject_id"].isin(used_subjects)].copy()
    while len(chosen) < n:
        available_classes = set(remaining["primary_cvd"].astype(str))
        class_gaps = {
            c: max(int(cvd_targets.get(c, 0)) - int(cvd_counts.get(c, 0)), 0)
            for c in available_classes
        }
        positive_gaps = {c: gap for c, gap in class_gaps.items() if gap > 0}
        if positive_gaps:
            target_class = sorted(positive_gaps.items(), key=lambda item: (-item[1], item[0]))[0][0]
            rank_pool = remaining[remaining["primary_cvd"].astype(str) == target_class]
        else:
            rank_pool = remaining
        ranked = score_rows_for_cvd_balance(rank_pool, cvd_targets, cvd_counts)
        if ranked.empty:
            raise ValueError(f"候选不足: 还需要 {n - len(chosen)} 条")
        row = ranked.iloc[0]
        chosen.append(row)
        sid = int(row["subject_id"])
        used_subjects.add(sid)
        cvd = str(row["primary_cvd"])
        cvd_counts[cvd] = int(cvd_counts.get(cvd, 0)) + 1
        remaining = remaining[remaining["subject_id"].astype(int) != sid]
    return pd.DataFrame(chosen)


def pick_subject100(
    pool: pd.DataFrame,
    target_subjects: int,
    target_cxr: int,
    target_echo: int,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    cvd_targets = _build_balanced_cvd_targets(pool, target_subjects)
    cvd_counts = {k: 0 for k in cvd_targets}
    used_subjects: Set[int] = set()

    selected_parts = []
    for cvd_len in [3, 4]:
        if sum(len(x) for x in selected_parts) >= target_subjects:
            break
        len_pool = pool[pool["cvd_unique_count"].astype(int) == cvd_len].copy()
        echo_need = target_echo - sum(int(part["has_echo"].sum()) for part in selected_parts)
        cxr_need = target_cxr - sum(int(part["has_cxr"].sum()) for part in selected_parts)
        total_need = target_subjects - sum(len(part) for part in selected_parts)
        if total_need <= 0:
            break

        echo_group = len_pool[(len_pool["has_echo"] == 1) & (~len_pool["subject_id"].isin(used_subjects))]
        cxr_group = len_pool[(len_pool["has_echo"] == 0) & (len_pool["has_cxr"] == 1) & (~len_pool["subject_id"].isin(used_subjects))]
        neither_group = len_pool[(len_pool["has_echo"] == 0) & (len_pool["has_cxr"] == 0) & (~len_pool["subject_id"].isin(used_subjects))]

        take_echo = min(max(echo_need, 0), len(echo_group), total_need)
        if take_echo:
            selected_parts.append(choose_from_group(echo_group, take_echo, cvd_targets, cvd_counts, used_subjects))

        cxr_need = target_cxr - sum(int(part["has_cxr"].sum()) for part in selected_parts)
        total_need = target_subjects - sum(len(part) for part in selected_parts)
        take_cxr = min(max(cxr_need, 0), len(cxr_group), total_need)
        if take_cxr:
            selected_parts.append(choose_from_group(cxr_group, take_cxr, cvd_targets, cvd_counts, used_subjects))

        total_need = target_subjects - sum(len(part) for part in selected_parts)
        if total_need > 0 and len(neither_group) > 0:
            selected_parts.append(choose_from_group(neither_group, min(total_need, len(neither_group)), cvd_targets, cvd_counts, used_subjects))

    if not selected_parts:
        raise ValueError("未选出任何 subject")

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates(subset=["subject_id"], keep="first").copy()
    if len(selected) != target_subjects:
        raise ValueError(f"subject数量异常: {len(selected)} != {target_subjects}")
    if selected["hadm_id"].nunique() != target_subjects:
        raise ValueError(f"hadm数量异常: {selected['hadm_id'].nunique()} != {target_subjects}")
    if int(selected["has_echo"].sum()) != target_echo:
        raise ValueError(f"echo数量异常: {int(selected['has_echo'].sum())} != {target_echo}")
    if int(selected["has_cxr"].sum()) != target_cxr:
        raise ValueError(f"cxr数量异常: {int(selected['has_cxr'].sum())} != {target_cxr}")

    actual_cvd = selected["primary_cvd"].value_counts().sort_index().astype(int).to_dict()
    return selected, cvd_targets, actual_cvd


def write_subset_outputs(
    out_root: str,
    selected_hadms: Set[int],
    selected_labels: pd.DataFrame,
    master: pd.DataFrame,
    details: Dict[str, Optional[pd.DataFrame]],
    selected_meta: pd.DataFrame,
) -> None:
    dirs = ensure_dirs(out_root)
    selected_labels[FULL_LABEL_COLUMNS].to_csv(os.path.join(dirs["labels"], "cohort_labels_test100.csv"), index=False)
    selected_labels[FULL_LABEL_COLUMNS].to_csv(
        os.path.join(dirs["labels"], "cohort_labels_test100.csv.gz"),
        index=False,
        compression="gzip",
    )

    master[master["hadm_id"].astype(int).isin(selected_hadms)].copy().to_csv(
        os.path.join(dirs["timeline"], "master_timeline_test100.csv.gz"),
        index=False,
        compression="gzip",
    )

    for mod, df in details.items():
        if df is None:
            continue
        if "hadm_id" in df.columns:
            out = df[df["hadm_id"].astype(int).isin(selected_hadms)].copy()
        else:
            out = df.copy()
        out.to_csv(os.path.join(dirs["details"], f"details_{mod}_grp_test100.csv.gz"), index=False, compression="gzip")

    pair_df = selected_labels[["subject_id", "hadm_id"]].drop_duplicates().copy()
    pair_df["subject_id"] = pair_df["subject_id"].astype(int)
    pair_df["hadm_id"] = pair_df["hadm_id"].astype(int)
    meta = selected_meta.set_index("hadm_id")[["cvd_unique_count"]].to_dict()["cvd_unique_count"]
    pair_df["cvd_label_count"] = pair_df["hadm_id"].map(lambda x: int(meta.get(int(x), -1)))
    pair_df["selection_tier"] = pair_df["cvd_label_count"].map(lambda x: "len3_primary" if int(x) == 3 else "len4_fallback")
    pair_df = pair_df.sort_values(["subject_id", "hadm_id"])
    pair_df.to_csv(os.path.join(out_root, "test_subject_hadm.csv"), index=False)

    with open(os.path.join(out_root, "test_subjects.txt"), "w") as f:
        for sid in sorted(pair_df["subject_id"].astype(int)):
            f.write(f"{sid}\n")
    with open(os.path.join(out_root, "test_hadms.txt"), "w") as f:
        for hid in sorted(pair_df["hadm_id"].astype(int)):
            f.write(f"{hid}\n")


def validate_dataset(out_root: str, labels: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    stats["subject_count"] = int(labels["subject_id"].nunique())
    stats["hadm_count"] = int(labels["hadm_id"].nunique())
    stats["subject_hadm_pairs_count"] = int(labels[["subject_id", "hadm_id"]].drop_duplicates().shape[0])
    stats["subjects_with_multi_hadm"] = int((labels.groupby("subject_id")["hadm_id"].nunique() > 1).sum())
    stats["actual_ecg_hadm"] = int(labels.loc[labels["has_ecg"].astype(int) == 1, "hadm_id"].nunique())
    stats["actual_cxr_hadm"] = int(labels.loc[labels["has_cxr"].astype(int) == 1, "hadm_id"].nunique())
    stats["actual_echo_hadm"] = int(labels.loc[labels["has_echo"].astype(int) == 1, "hadm_id"].nunique())
    stats["modality_score_check_passed"] = bool(
        (labels["has_ecg"] + labels["has_cxr"] + labels["has_echo"]).equals(labels["modality_score"])
    )
    stats["label_constraints_passed"] = bool(
        stats["subject_count"] == 100
        and stats["hadm_count"] == 100
        and stats["subject_hadm_pairs_count"] == 100
        and stats["subjects_with_multi_hadm"] == 0
        and stats["actual_ecg_hadm"] == 100
        and stats["actual_cxr_hadm"] == 60
        and stats["actual_echo_hadm"] == 40
        and stats["modality_score_check_passed"]
    )

    manifest_stats = {}
    for mod in ["ecg", "cxr", "echo"]:
        sub = manifest[manifest["modality"] == mod].copy() if not manifest.empty else pd.DataFrame()
        missing = 0
        for _, row in sub.iterrows():
            path = Path(out_root) / str(row["target_path"])
            if not (path.exists() and path.is_file() and path.stat().st_size > 0):
                missing += 1
        manifest_stats[mod] = {
            "manifest_rows": int(len(sub)),
            "manifest_subjects": int(sub["subject_id"].nunique()) if not sub.empty else 0,
            "manifest_hadms": int(sub["hadm_id"].nunique()) if not sub.empty else 0,
            "missing_or_empty": int(missing),
            "complete": bool(missing == 0),
        }
    stats["manifest_validation"] = manifest_stats
    stats["download_complete"] = bool(all(v["complete"] for v in manifest_stats.values()))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--new-output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--restrict-subjects-file", default=DEFAULT_RESTRICT_SUBJECTS)
    ap.add_argument("--existing-dataset-root", default=DEFAULT_EXISTING_DATASET_ROOT)
    ap.add_argument("--target-subjects", type=int, default=100)
    ap.add_argument("--target-cxr", type=int, default=60)
    ap.add_argument("--target-echo", type=int, default=40)
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
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--download-retries", type=int, default=2)
    ap.add_argument("--download-timeout-sec", type=int, default=180)
    args = ap.parse_args()

    out_root = os.path.join(args.new_output_root, args.dataset_name)
    labels, master, details = load_source(args.source_root, "step_all")
    allowed_subjects = read_subjects(args.restrict_subjects_file)

    pool = prepare_one_hadm_per_subject_pool(
        labels=labels,
        details=details,
        allowed_subjects=allowed_subjects,
        existing_origin_root=args.existing_dataset_root,
        cxr_format=args.cxr_format,
        ecg_expand_pairs=args.ecg_expand_pairs,
    )
    selected_meta, cvd_targets, actual_cvd = pick_subject100(
        pool=pool,
        target_subjects=args.target_subjects,
        target_cxr=args.target_cxr,
        target_echo=args.target_echo,
    )
    selected_hadms = set(selected_meta["hadm_id"].astype(int))
    selected_labels = labels[labels["hadm_id"].astype(int).isin(selected_hadms)].copy()
    selected_labels = selected_labels.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)

    write_subset_outputs(out_root, selected_hadms, selected_labels, master, details, selected_meta)
    manifest = build_manifest(
        selected_labels=selected_labels,
        details=details,
        cxr_base_url=args.cxr_base_url,
        ecg_base_url=args.ecg_base_url,
        echo_base_url=args.echo_base_url,
        cxr_format=args.cxr_format,
        ecg_expand_pairs=args.ecg_expand_pairs,
        echo_expand_all=args.echo_expand_all_dcm_by_study,
        echo_record_list_path=args.echo_record_list_path,
        out_root=out_root,
    )
    manifest.to_csv(os.path.join(out_root, "origin_manifest_test100.csv"), index=False)
    manifest.to_csv(os.path.join(out_root, "origin_manifest_test100.csv.gz"), index=False, compression="gzip")

    reuse_stats = copy_reusable_files(manifest, args.existing_dataset_root, out_root)
    if args.download_origin_data:
        if not args.physionet_user or not args.physionet_password:
            raise ValueError("下载需要设置 PHYSIONET_USER/PHYSIONET_PASSWORD 或传入 --physionet-user/--physionet-password")
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
        open(os.path.join(out_root, "download_success.txt"), "w").close()
        open(os.path.join(out_root, "download_failed.txt"), "w").close()

    stats = validate_dataset(out_root, selected_labels, manifest)
    stats.update(
        {
            "target_subjects": int(args.target_subjects),
            "target_hadm": int(args.target_subjects),
            "target_ecg_hadm": int(args.target_subjects),
            "target_cxr_hadm": int(args.target_cxr),
            "target_echo_hadm": int(args.target_echo),
            "target_cvd_distribution": cvd_targets,
            "actual_cvd_distribution": actual_cvd,
            "manifest_rows": int(len(manifest)),
            "reuse": reuse_stats,
            "existing_dataset_root": args.existing_dataset_root,
            "source_root": args.source_root,
        }
    )
    with open(os.path.join(out_root, "stats_test100.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
