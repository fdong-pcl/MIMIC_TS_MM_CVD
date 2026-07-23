import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_DATASET_ROOT = (
    "/Users/fandong/Desktop/pcl/Data/CVD_MMData/temporal_output_norm_icd10_test100_compact/"
    "subset2_test100_subject100_pre_ds_onestudy_cvd3_fullstudy_backup"
)
DEFAULT_HOSP_ROOT = "/Users/fandong/Desktop/pcl/Data/mimiciv/3.1/hosp"
DEFAULT_MAPPING_ROOT = "/Users/fandong/Desktop/pcl/Data/CVD_MMData/icd9_10/icd9_10_cvd_mapping"
DEFAULT_REPO_GEMS_PATH = Path(__file__).resolve().parent / "cvd_category" / "icd9toicd10cmgem.csv"


def choose_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_single_icd_range(range_str: object) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(range_str, str) or not range_str.strip():
        return None, None
    value = range_str.strip()
    if "–" in value:
        parts = value.split("–")
    elif "-" in value:
        parts = value.split("-")
    else:
        return value, value
    return parts[0].strip(), parts[-1].strip()


def is_icd_in_range(icd_code: object, icd_version: object, range_min: object, range_max: object) -> bool:
    if pd.isna(icd_code) or pd.isna(icd_version) or not range_min:
        return False
    clean_code = str(icd_code).replace(".", "").upper()
    clean_min = str(range_min).replace(".", "").upper()
    clean_max = str(range_max).replace(".", "").upper()
    if len(clean_code) < 3:
        return False

    try:
        version = int(icd_version)
    except Exception:
        return False

    if version == 9:
        if len(clean_min) != 3 or len(clean_max) != 3:
            code_prefix = clean_code[: len(clean_min)]
            return clean_min <= code_prefix <= clean_max
        code_prefix = clean_code[:3]
        return clean_min <= code_prefix <= clean_max
    if version == 10:
        code_prefix = clean_code[: len(clean_min)]
        return clean_min <= code_prefix <= clean_max
    return False


def build_match_map(df: pd.DataFrame) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        internal_code = row["InternalCode"]
        for version, col in [(10, "ICD10_Code"), (9, "ICD9_Code")]:
            if pd.isna(row.get(col)):
                continue
            for part in str(row[col]).split("/"):
                min_code, max_code = parse_single_icd_range(part.strip())
                if min_code:
                    out.append({"code": internal_code, "version": version, "min": min_code, "max": max_code})
    return out


def find_gems_path(mapping_root: Path) -> Optional[Path]:
    gems_path = choose_existing(
        [
            mapping_root / "icd9toicd10cmgem.csv",
            mapping_root / "icd9toicd10cmgem.csv.gz",
            mapping_root / "icd9toicd10cm_gem.csv",
            mapping_root / "icd9toicd10cm_gem.csv.gz",
            DEFAULT_REPO_GEMS_PATH,
        ]
    )
    return gems_path


def load_gems_mapping(mapping_root: Path) -> Tuple[Dict[str, str], Optional[Path]]:
    gems_path = find_gems_path(mapping_root)
    if gems_path is None:
        return {}, None
    df = pd.read_csv(gems_path, sep=None, engine="python", dtype=str)
    lower_cols = {c.lower(): c for c in df.columns}
    icd9_col = lower_cols.get("icd9cm") or lower_cols.get("icd9")
    icd10_col = lower_cols.get("icd10cm") or lower_cols.get("icd10")
    if not icd9_col or not icd10_col:
        return {}, gems_path
    df[icd9_col] = df[icd9_col].astype(str).str.strip().str.upper()
    df[icd10_col] = df[icd10_col].astype(str).str.strip().str.upper()
    return df.drop_duplicates(icd9_col).set_index(icd9_col)[icd10_col].to_dict(), gems_path


def diagnosis_role(seq_num: object) -> str:
    seq = int(seq_num)
    if seq == 1:
        return "primary_diagnosis"
    return "comorbidity_diagnosis"


def parse_listlike(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return list(ast.literal_eval(text))
    except Exception:
        return [x.strip().strip("'\"") for x in text.strip("[]").split(",") if x.strip()]


def match_first(code: object, version: object, match_map: List[Dict[str, object]]) -> Optional[str]:
    for item in match_map:
        if int(item["version"]) == int(version) and is_icd_in_range(code, version, item["min"], item["max"]):
            return str(item["code"])
    return None


def build_cvd_details(labels: pd.DataFrame, hosp_root: Path, mapping_root: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    hadms = set(labels["hadm_id"].astype(int))
    diag = pd.read_csv(
        hosp_root / "diagnoses_icd.csv.gz",
        dtype={"subject_id": "int64", "hadm_id": "int64", "seq_num": "int64", "icd_code": str, "icd_version": "int64"},
    )
    diag = diag[diag["hadm_id"].isin(hadms)].copy()

    dictionary = pd.read_csv(hosp_root / "d_icd_diagnoses.csv.gz", dtype={"icd_code": str, "icd_version": "int64"})
    diag = diag.merge(dictionary[["icd_code", "icd_version", "long_title"]], on=["icd_code", "icd_version"], how="left")

    coarse_map = build_match_map(pd.read_csv(mapping_root / "CVD_coarse_category.csv"))
    fine_map = build_match_map(pd.read_csv(mapping_root / "CVD_fine_category.csv"))
    gems, gems_path = load_gems_mapping(mapping_root)
    v10_titles = dictionary[dictionary["icd_version"].eq(10)].set_index("icd_code")["long_title"].to_dict()

    rows = []
    matched_keys = set()
    for _, row in diag.iterrows():
        code = str(row["icd_code"]).replace(".", "").upper()
        version = int(row["icd_version"])
        coarse = match_first(code, version, coarse_map)
        fine = match_first(code, version, fine_map)
        if coarse is None and fine:
            coarse = str(fine)[:5]
        if coarse is None:
            continue
        if version == 10:
            norm_code = code
            norm_method = "Original_V10"
        elif code in gems:
            norm_code = gems[code]
            norm_method = "GEMs_Official"
        else:
            norm_code = pd.NA
            norm_method = "Failed_Mapping"
        rows.append(
            {
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(row["hadm_id"]),
                "seq_num": int(row["seq_num"]),
                "diagnosis_role": diagnosis_role(row["seq_num"]),
                "icd_code": code,
                "icd_version": version,
                "long_title": row.get("long_title"),
                "CVD_coarse_category": coarse,
                "CVD_fine_category": fine,
                "norm_icd10_code": norm_code,
                "norm_icd10_title": v10_titles.get(norm_code) if pd.notna(norm_code) else pd.NA,
                "norm_method": norm_method,
            }
        )
        matched_keys.add((int(row["hadm_id"]), code))
    details = pd.DataFrame(rows)
    if details.empty:
        raise ValueError("未重建出任何 CVD 诊断明细")
    details = supplement_from_existing_labels(details, diag, labels, v10_titles, gems)
    gems_meta = {
        "gems_path": str(gems_path) if gems_path else None,
        "gems_loaded": bool(gems),
        "gems_mapping_count": int(len(gems)),
    }
    return details.sort_values(["subject_id", "hadm_id", "seq_num", "icd_code"]).reset_index(drop=True), gems_meta


def supplement_from_existing_labels(
    details: pd.DataFrame,
    diag: pd.DataFrame,
    labels: pd.DataFrame,
    v10_titles: Dict[str, str],
    gems: Dict[str, str],
) -> pd.DataFrame:
    extra_rows = []
    for _, label in labels.iterrows():
        hid = int(label["hadm_id"])
        existing = set(details.loc[details["hadm_id"].eq(hid), "CVD_coarse_category"].astype(str))
        label_coarse = [str(x) for x in parse_listlike(label.get("cvd_list", label.get("CVD_coarse_category")))]
        missing_coarse = [c for c in label_coarse if c not in existing]
        if not missing_coarse:
            continue

        label_codes = {str(x).replace(".", "").upper() for x in parse_listlike(label.get("icd_code"))}
        label_fine = [str(x) for x in parse_listlike(label.get("CVD_fine_category"))]
        used_codes = set(details.loc[details["hadm_id"].eq(hid), "icd_code"].astype(str))
        candidates = diag[
            diag["hadm_id"].astype(int).eq(hid)
            & diag["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper().isin(label_codes - used_codes)
        ].copy()
        candidates = candidates.sort_values(["seq_num", "icd_code"])

        for coarse in missing_coarse:
            if candidates.empty:
                continue
            row = candidates.iloc[0]
            candidates = candidates.iloc[1:].copy()
            code = str(row["icd_code"]).replace(".", "").upper()
            version = int(row["icd_version"])
            if version == 10:
                norm_code = code
                norm_method = "Original_V10"
            elif code in gems:
                norm_code = gems[code]
                norm_method = "GEMs_Official"
            else:
                norm_code = pd.NA
                norm_method = "Failed_Mapping"
            fine_candidates = [f for f in label_fine if f.startswith(coarse)]
            extra_rows.append(
                {
                    "subject_id": int(row["subject_id"]),
                    "hadm_id": hid,
                    "seq_num": int(row["seq_num"]),
                    "diagnosis_role": diagnosis_role(row["seq_num"]),
                    "icd_code": code,
                    "icd_version": version,
                    "long_title": row.get("long_title"),
                    "CVD_coarse_category": coarse,
                    "CVD_fine_category": fine_candidates[0] if fine_candidates else pd.NA,
                    "norm_icd10_code": norm_code,
                    "norm_icd10_title": v10_titles.get(norm_code) if pd.notna(norm_code) else pd.NA,
                    "norm_method": norm_method,
                }
            )
    if extra_rows:
        details = pd.concat([details, pd.DataFrame(extra_rows)], ignore_index=True)
    return details


def dumps_list(values: List[object]) -> str:
    return json.dumps(values, ensure_ascii=False)


def make_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detail_cols = [
        "seq_num",
        "diagnosis_role",
        "icd_code",
        "icd_version",
        "long_title",
        "CVD_coarse_category",
        "CVD_fine_category",
        "norm_icd10_code",
        "norm_icd10_title",
        "norm_method",
    ]
    for (subject_id, hadm_id), group in details.groupby(["subject_id", "hadm_id"], sort=False):
        group = group.sort_values(["seq_num", "icd_code"]).copy()
        category_rows = []
        for coarse, sub in group.groupby("CVD_coarse_category"):
            first = sub.sort_values(["seq_num", "icd_code"]).iloc[0]
            category_rows.append(
                {
                    "CVD_coarse_category": str(coarse),
                    "min_seq_num": int(first["seq_num"]),
                    "diagnosis_role": diagnosis_role(first["seq_num"]),
                }
            )
        category_rows = sorted(category_rows, key=lambda x: (x["min_seq_num"], x["CVD_coarse_category"]))
        coarse_sorted = [x["CVD_coarse_category"] for x in category_rows]
        seq_sorted = [x["min_seq_num"] for x in category_rows]
        role_sorted = [x["diagnosis_role"] for x in category_rows]
        rows.append(
            {
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "cvd_diagnosis_seq_sorted_json": dumps_list(group[detail_cols].where(pd.notna(group[detail_cols]), None).to_dict("records")),
                "cvd_coarse_seq_sorted": dumps_list(coarse_sorted),
                "cvd_coarse_min_seq_sorted": dumps_list(seq_sorted),
                "cvd_coarse_role_sorted": dumps_list(role_sorted),
                "primary_cvd_coarse_categories": dumps_list([c for c, r in zip(coarse_sorted, role_sorted) if r == "primary_diagnosis"]),
                "comorbidity_cvd_coarse_categories": dumps_list([c for c, r in zip(coarse_sorted, role_sorted) if r == "comorbidity_diagnosis"]),
                "primary_cvd_count": int(sum(r == "primary_diagnosis" for r in role_sorted)),
                "comorbidity_cvd_count": int(sum(r == "comorbidity_diagnosis" for r in role_sorted)),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    labels: pd.DataFrame,
    enriched: pd.DataFrame,
    details: pd.DataFrame,
    expected_cvd_unique_count: Optional[int] = 3,
) -> Dict[str, object]:
    expected_count = int(expected_cvd_unique_count) if expected_cvd_unique_count is not None else None
    checks = {
        "label_rows": int(len(enriched)),
        "subject_count": int(enriched["subject_id"].nunique()),
        "hadm_count": int(enriched["hadm_id"].nunique()),
        "expected_cvd_unique_count": expected_count,
        "cvd_unique_count_matches_expected": bool(
            True if expected_count is None else enriched["cvd_unique_count"].astype(int).eq(expected_count).all()
        ),
        "coarse_sorted_len_matches_expected": bool(
            True if expected_count is None else enriched["cvd_coarse_seq_sorted"].apply(lambda x: len(json.loads(x)) == expected_count).all()
        ),
        "role_counts_sum_matches_expected": bool(
            (
                enriched["primary_cvd_count"].astype(int)
                + enriched["comorbidity_cvd_count"].astype(int)
            )
            .eq(expected_count)
            .all()
            if expected_count is not None
            else True
        ),
        "detail_hadm_count": int(details["hadm_id"].nunique()),
        "detail_role_rule_ok": bool(
            details.apply(
                lambda r: (
                    (int(r["seq_num"]) == 1 and r["diagnosis_role"] == "primary_diagnosis")
                    or (int(r["seq_num"]) != 1 and r["diagnosis_role"] == "comorbidity_diagnosis")
                ),
                axis=1,
            ).all()
        ),
        "original_label_columns_preserved": bool(list(labels.columns) == list(enriched.columns[: len(labels.columns)])),
    }
    expected = {
        "label_rows": 100,
        "subject_count": 100,
        "hadm_count": 100,
        "detail_hadm_count": 100,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise ValueError(f"{key} 校验失败: {checks[key]} != {value}")
    for key in [
        "cvd_unique_count_matches_expected",
        "coarse_sorted_len_matches_expected",
        "role_counts_sum_matches_expected",
        "detail_role_rule_ok",
        "original_label_columns_preserved",
    ]:
        if not checks[key]:
            raise ValueError(f"{key} 校验失败")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--hosp-root", default=DEFAULT_HOSP_ROOT)
    ap.add_argument("--mapping-root", default=DEFAULT_MAPPING_ROOT)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_dir = dataset_root / "labels"
    labels_path = labels_dir / "cohort_labels_test100.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")

    labels = pd.read_csv(labels_path)
    if len(labels) != 100:
        raise ValueError(f"标签行数异常: {len(labels)} != 100")

    details, gems_meta = build_cvd_details(labels, Path(args.hosp_root), Path(args.mapping_root))
    details = details[details["hadm_id"].isin(set(labels["hadm_id"].astype(int)))].copy()
    summary = make_summary(details)
    enriched = labels.merge(summary, on=["subject_id", "hadm_id"], how="left")
    if enriched["cvd_coarse_seq_sorted"].isna().any():
        missing = enriched[enriched["cvd_coarse_seq_sorted"].isna()][["subject_id", "hadm_id"]]
        raise ValueError(f"部分住院缺少 seq severity 汇总: {missing.to_dict('records')[:10]}")

    checks = validate_outputs(labels, enriched, details)

    details_out = labels_dir / "cvd_diagnosis_seq_details_test100.csv"
    enriched_out = labels_dir / "cohort_labels_test100_seqseverity.csv"
    details.to_csv(details_out, index=False)
    details.to_csv(labels_dir / "cvd_diagnosis_seq_details_test100.csv.gz", index=False, compression="gzip")
    enriched.to_csv(enriched_out, index=False)
    enriched.to_csv(labels_dir / "cohort_labels_test100_seqseverity.csv.gz", index=False, compression="gzip")

    sample = details[["subject_id", "hadm_id", "seq_num", "icd_code", "CVD_coarse_category", "diagnosis_role"]].head(20)
    stats = {
        "dataset_root": str(dataset_root),
        "enriched_labels": str(enriched_out),
        "details": str(details_out),
        "checks": checks,
        **gems_meta,
        "detail_rows": int(len(details)),
        "norm_method_distribution": {str(k): int(v) for k, v in details["norm_method"].value_counts(dropna=False).items()},
        "role_distribution_detail_rows": {str(k): int(v) for k, v in details["diagnosis_role"].value_counts().items()},
        "coarse_category_distribution_detail_rows": {str(k): int(v) for k, v in details["CVD_coarse_category"].value_counts().items()},
        "sample_first_20": sample.to_dict("records"),
    }
    with open(labels_dir / "seqseverity_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
