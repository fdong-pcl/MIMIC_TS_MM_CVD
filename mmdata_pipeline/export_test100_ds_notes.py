import argparse
import json
import os
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


def collect_ds_note_ids(dataset_root: Path) -> pd.DataFrame:
    notes_path = dataset_root / "details" / "details_notes_grp_test100.csv.gz"
    if not notes_path.exists():
        raise FileNotFoundError(f"未找到 notes details: {notes_path}")

    notes = pd.read_csv(notes_path)
    rows: List[Dict[str, object]] = []
    for _, row in notes.iterrows():
        note_ids = [str(x) for x in parse_listlike(row.get("note_id_list", "[]"))]
        note_types = [str(x) for x in parse_listlike(row.get("note_type_list", "[]"))]
        for idx, note_id in enumerate(note_ids):
            note_type = note_types[idx] if idx < len(note_types) else ""
            if note_type == "DS" or "-DS-" in note_id:
                rows.append(
                    {
                        "subject_id": int(row["subject_id"]),
                        "hadm_id": int(float(row["hadm_id"])),
                        "note_id": note_id,
                        "details_charttime": row.get("charttime"),
                        "details_reference_id": row.get("reference_id"),
                    }
                )

    out = pd.DataFrame(rows).drop_duplicates(subset=["note_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("当前 test100 details 中没有找到 DS note_id")
    return out


def scan_discharge_notes(discharge_path: Path, wanted_note_ids: Set[str], chunksize: int) -> pd.DataFrame:
    usecols = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]
    parts = []
    for chunk in pd.read_csv(discharge_path, usecols=usecols, chunksize=chunksize):
        hit = chunk[chunk["note_id"].astype(str).isin(wanted_note_ids)].copy()
        if not hit.empty:
            parts.append(hit)
        found = sum(len(p) for p in parts)
        if found >= len(wanted_note_ids):
            break
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["note_id"]).reset_index(drop=True)


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def write_note_text_files(notes: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    notes_dir = output_root / "notes"
    for _, row in notes.iterrows():
        sid = int(row["subject_id"])
        hid = int(row["hadm_id"])
        note_id = str(row["note_id"])
        rel_path = Path("notes") / str(sid) / str(hid) / f"{safe_filename(note_id)}.txt"
        dst = output_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = "" if pd.isna(row.get("text")) else str(row.get("text"))
        dst.write_text(text, encoding="utf-8")
        rows.append(
            {
                "note_id": note_id,
                "subject_id": sid,
                "hadm_id": hid,
                "note_type": row.get("note_type"),
                "note_seq": row.get("note_seq"),
                "charttime": row.get("charttime"),
                "storetime": row.get("storetime"),
                "target_path": str(rel_path),
                "text_chars": len(text),
                "text_bytes": dst.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def export_discharge_detail(mimiciv_root: Path, note_ids: Set[str], output_root: Path) -> int:
    detail_path = choose_existing(
        [
            mimiciv_root / "note" / "discharge_detail.csv",
            mimiciv_root / "note" / "discharge_detail.csv.gz",
        ]
    )
    detail = pd.read_csv(detail_path)
    detail = detail[detail["note_id"].astype(str).isin(note_ids)].copy()
    detail.to_csv(output_root / "discharge_detail_test100.csv.gz", index=False, compression="gzip")
    return int(len(detail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--mimiciv-root", default=DEFAULT_MIMICIV_ROOT)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--expected-notes", type=int, default=100)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    mimiciv_root = Path(args.mimiciv_root)
    output_root = Path(args.output_dir) if args.output_dir else dataset_root / "ds_notes"
    output_root.mkdir(parents=True, exist_ok=True)

    target = collect_ds_note_ids(dataset_root)
    discharge_path = choose_existing(
        [
            mimiciv_root / "note" / "discharge.csv",
            mimiciv_root / "note" / "discharge.csv.gz",
        ]
    )

    wanted = set(target["note_id"].astype(str))
    notes = scan_discharge_notes(discharge_path, wanted, args.chunksize)
    missing = sorted(wanted - set(notes["note_id"].astype(str)))
    if missing:
        raise ValueError(f"有 {len(missing)} 个 DS note_id 未在原始 discharge 文件中找到: {missing[:10]}")
    if len(notes) != int(args.expected_notes):
        raise ValueError(f"导出的 DS notes 数量异常: {len(notes)} != {args.expected_notes}")

    notes = notes.merge(
        target[["note_id", "details_charttime", "details_reference_id"]],
        on="note_id",
        how="left",
    )
    notes = notes.sort_values(["subject_id", "hadm_id", "note_id"]).reset_index(drop=True)
    notes.to_csv(output_root / "discharge_notes_test100.csv.gz", index=False, compression="gzip")

    manifest = write_note_text_files(notes, output_root)
    manifest = manifest.merge(
        target[["note_id", "details_charttime", "details_reference_id"]],
        on="note_id",
        how="left",
    )
    manifest["source_file"] = str(discharge_path)
    manifest = manifest[
        [
            "note_id",
            "subject_id",
            "hadm_id",
            "note_type",
            "note_seq",
            "charttime",
            "storetime",
            "details_charttime",
            "details_reference_id",
            "source_file",
            "target_path",
            "text_chars",
            "text_bytes",
        ]
    ]
    manifest.to_csv(output_root / "ds_notes_manifest.csv", index=False)
    manifest.to_csv(output_root / "ds_notes_manifest.csv.gz", index=False, compression="gzip")

    detail_rows = export_discharge_detail(mimiciv_root, wanted, output_root)
    stats = {
        "dataset_root": str(dataset_root),
        "mimiciv_root": str(mimiciv_root),
        "source_discharge_file": str(discharge_path),
        "output_root": str(output_root),
        "ds_note_count": int(len(notes)),
        "subject_count": int(notes["subject_id"].nunique()),
        "hadm_count": int(notes["hadm_id"].nunique()),
        "txt_file_count": int(len(manifest)),
        "discharge_detail_rows": detail_rows,
        "missing_note_count": int(len(missing)),
    }
    with open(output_root / "ds_notes_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
