import argparse
import ast
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def parse_listlike(val) -> List[str]:
    if pd.isna(val):
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if str(x)]
    s = str(val).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            return [str(x) for x in obj if str(x)]
    except Exception:
        pass
    return [s]


def normalize_rel_path(p: str) -> str:
    p = str(p).strip().replace("\\", "/")
    while p.startswith("/"):
        p = p[1:]
    return p


def build_full_url(base_url: str, rel_path: str) -> str:
    return base_url.rstrip("/") + "/" + rel_path.lstrip("/")


def ensure_dirs(root: str) -> Dict[str, str]:
    d = {
        "root": root,
        "labels": os.path.join(root, "labels"),
        "details": os.path.join(root, "details"),
        "timeline": os.path.join(root, "timeline"),
        "origin": os.path.join(root, "origin"),
    }
    for x in d.values():
        os.makedirs(x, exist_ok=True)
    for mod in ["cxr", "ecg", "echo"]:
        os.makedirs(os.path.join(d["origin"], mod), exist_ok=True)
    return d


def load_source(source_root: str, source_mode: str):
    details = {}
    if source_mode == "subset2":
        labels = pd.read_csv(os.path.join(source_root, "labels", "cohort_labels_subset2_all.csv.gz"))
        master = pd.read_csv(os.path.join(source_root, "timeline", "master_timeline_subset2_all.csv.gz"))
        for mod in ["notes", "cxr", "ecg", "echo", "lab", "rx", "proc"]:
            p = os.path.join(source_root, "details", f"details_{mod}_grp_subset2_all.csv.gz")
            details[mod] = pd.read_csv(p) if os.path.exists(p) else None
        return labels, master, details

    if source_mode == "step_all":
        labels_p = os.path.join(source_root, "step4_labels_splits", "labels", "cohort_labels_all.csv")
        master_p = os.path.join(source_root, "step3_temporal_timeline", "master_timeline_all.csv.gz")
        if not os.path.exists(labels_p):
            raise FileNotFoundError(f"未找到labels文件: {labels_p}")
        if not os.path.exists(master_p):
            raise FileNotFoundError(f"未找到timeline文件: {master_p}")
        labels = pd.read_csv(labels_p)
        master = pd.read_csv(master_p)
        for mod in ["notes", "cxr", "ecg", "echo", "lab", "rx", "proc"]:
            p = os.path.join(source_root, "step3_temporal_timeline", f"details_{mod}_grp_all.csv.gz")
            details[mod] = pd.read_csv(p) if os.path.exists(p) else None
        return labels, master, details

    raise ValueError(f"不支持的source_mode: {source_mode}")


def hadm_sets(details: Dict[str, Optional[pd.DataFrame]]):
    out = {}
    for mod in ["cxr", "ecg", "echo"]:
        df = details.get(mod)
        if df is None or df.empty:
            out[mod] = set()
        else:
            out[mod] = set(df["hadm_id"].dropna().astype(int).tolist())
    return out


def _extract_prefix_tag(rel_path: str) -> Optional[str]:
    parts = normalize_rel_path(rel_path).split("/")
    for p in parts:
        if re.fullmatch(r"p\\d+", p):
            return p
    return None


def _collect_hadm_modality_paths(details: Dict[str, Optional[pd.DataFrame]], mod: str, col: str) -> Dict[int, List[str]]:
    out: Dict[int, Set[str]] = {}
    df = details.get(mod)
    if df is None or df.empty:
        return {}
    for _, r in df.iterrows():
        if pd.isna(r.get("hadm_id")):
            continue
        hid = int(r["hadm_id"])
        for p in parse_listlike(r.get(col, "[]")):
            rp = normalize_rel_path(p)
            if rp:
                out.setdefault(hid, set()).add(rp)
    return {k: sorted(v) for k, v in out.items()}


def _build_existing_filename_index(existing_origin_root: Optional[str]) -> Dict[str, Set[str]]:
    idx = {"cxr": set(), "ecg": set(), "echo": set()}
    if not existing_origin_root or not os.path.exists(existing_origin_root):
        return idx
    for mod in ["cxr", "ecg", "echo"]:
        mod_root = os.path.join(existing_origin_root, mod)
        if not os.path.exists(mod_root):
            continue
        for root, _, files in os.walk(mod_root):
            for f in files:
                idx[mod].add(f)
    return idx


def _get_primary_cvd(val) -> str:
    lst = parse_listlike(val)
    if not lst:
        return "UNKNOWN"
    # 保持稳定：取排序后的第一个类别作为主类
    return sorted(set(lst))[0]


def _get_cvd_label_count(val) -> int:
    """计算一次住院的 CVD 粗分类标签数（去重后）。"""
    lst = parse_listlike(val)
    return len(set([x for x in lst if str(x).strip()]))


def _build_balanced_cvd_targets(base: pd.DataFrame, target_hadm: int) -> Dict[str, int]:
    """
    为 CVD 主类构造尽量均衡的目标配额。
    若存在 8 个类别，则每类大约 12/13；若少于 8，则在可用类别上均分。
    """
    cvd_order = [f"CVD_{x}" for x in "ABCDEFGH"]
    avail = base["primary_cvd"].value_counts().to_dict()
    active = [c for c in cvd_order if avail.get(c, 0) > 0]
    if not active:
        return {}
    n = len(active)
    base_n = target_hadm // n
    rem = target_hadm % n
    # 余数优先给样本更充足的类别，降低不可达风险
    active_by_capacity = sorted(active, key=lambda c: (-avail.get(c, 0), c))
    targets = {c: base_n for c in active}
    for c in active_by_capacity[:rem]:
        targets[c] += 1
    # 不超过类别可用上限
    for c in active:
        targets[c] = min(targets[c], int(avail.get(c, 0)))
    return targets


def _estimate_reuse_score_for_hadm(
    hadm_id: int,
    hadm_paths: Dict[str, Dict[int, List[str]]],
    existing_idx: Dict[str, Set[str]],
    cxr_format: str,
    ecg_expand_pairs: bool,
) -> int:
    score = 0
    # CXR
    for rp in hadm_paths["cxr"].get(hadm_id, []):
        x = rp
        if cxr_format == "jpg":
            x = x[:-4] + ".jpg" if x.lower().endswith(".dcm") else (x if x.lower().endswith(".jpg") else x + ".jpg")
        fn = "cxr_" + os.path.basename(x)
        if fn in existing_idx["cxr"]:
            score += 1
    # ECG
    ecg_paths = hadm_paths["ecg"].get(hadm_id, [])
    if ecg_expand_pairs:
        ecg_paths = expand_ecg_relpaths(ecg_paths)
    for rp in ecg_paths:
        fn = "ecg_" + os.path.basename(rp)
        if fn in existing_idx["ecg"]:
            score += 1
    # Echo
    for rp in hadm_paths["echo"].get(hadm_id, []):
        fn = "echo_" + os.path.basename(rp)
        if fn in existing_idx["echo"]:
            score += 1
    return score


def _greedy_select_with_constraints(
    base: pd.DataFrame,
    target_hadm: int,
    need_echo: int,
    need_cxr: int,
    cvd_targets: Dict[str, int],
    cvd_counts_init: Dict[str, int],
) -> Tuple[List[int], int, int, Dict[str, int]]:
    """
    在给定候选池上执行贪心选样。
    返回: (chosen_hadms, remain_need_echo, remain_need_cxr, cvd_counts)
    """
    cvd_counts = dict(cvd_counts_init)
    chosen: List[int] = []
    remaining = base.to_dict("records")
    rem_echo_total = int(sum(1 for r in remaining if bool(r["has_echo"])))
    rem_cxr_total = int(sum(1 for r in remaining if bool(r["has_cxr"])))

    while len(chosen) < target_hadm and remaining:
        best_idx = None
        best_score = None
        for idx, r in enumerate(remaining):
            he = 1 if bool(r["has_echo"]) else 0
            hc = 1 if bool(r["has_cxr"]) else 0
            next_need_echo = max(need_echo - he, 0)
            next_need_cxr = max(need_cxr - hc, 0)
            rem_echo_after = rem_echo_total - he
            rem_cxr_after = rem_cxr_total - hc
            # 可行性硬校验：选了它之后仍能满足剩余模态约束
            if next_need_echo > rem_echo_after or next_need_cxr > rem_cxr_after:
                continue

            cvd = r["primary_cvd"]
            cvd_gap = 0
            if cvd in cvd_targets:
                cvd_gap = max(cvd_targets[cvd] - cvd_counts.get(cvd, 0), 0)

            mod_bonus = (1 if (need_echo > 0 and he == 1) else 0) + (1 if (need_cxr > 0 and hc == 1) else 0)
            score = (
                mod_bonus,                            # 先补模态缺口
                1 if cvd_gap > 0 else 0,             # 再补 CVD 缺口
                cvd_gap,
                int(r["prefix_hit"]),                # 再看 p10~p15 命中
                int(r["reuse_score"]),               # 再看复用得分
                int(r["dischtime_ord"]),             # 再看时间新旧
                -int(r["hadm_id"]),                  # 稳定tie-break
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        pick = remaining.pop(best_idx)
        hid = int(pick["hadm_id"])
        chosen.append(hid)
        if bool(pick["has_echo"]):
            rem_echo_total -= 1
            need_echo = max(need_echo - 1, 0)
        if bool(pick["has_cxr"]):
            rem_cxr_total -= 1
            need_cxr = max(need_cxr - 1, 0)
        cvd = pick["primary_cvd"]
        if cvd in cvd_counts:
            cvd_counts[cvd] += 1

    return chosen, need_echo, need_cxr, cvd_counts


def pick_hadms(
    labels: pd.DataFrame,
    details: Dict[str, Optional[pd.DataFrame]],
    target_hadm: int,
    target_echo: int,
    target_cxr: int,
    target_ecg: int,
    prefix_priority: List[str],
    prefix_rule: str,
    maximize_reuse: bool,
    existing_origin_root: Optional[str],
    cxr_format: str,
    ecg_expand_pairs: bool,
    disable_prefix_priority: bool,
) -> Tuple[List[int], Dict[str, int], Dict[str, int], Dict[int, Dict[str, int]], Dict[str, object]]:
    base_cols = ["subject_id", "hadm_id", "dischtime"]
    has_cvd_col = "CVD_coarse_category" in labels.columns
    if has_cvd_col:
        base_cols.append("CVD_coarse_category")
    base = labels[base_cols].drop_duplicates().copy()
    base["hadm_id"] = base["hadm_id"].astype(int)
    hs = hadm_sets(details)
    base["has_ecg"] = base["hadm_id"].isin(hs["ecg"])
    base["has_echo"] = base["hadm_id"].isin(hs["echo"])
    base["has_cxr"] = base["hadm_id"].isin(hs["cxr"])
    if has_cvd_col:
        base["primary_cvd"] = base["CVD_coarse_category"].apply(_get_primary_cvd)
        base["cvd_label_count"] = base["CVD_coarse_category"].apply(_get_cvd_label_count)
    else:
        base["primary_cvd"] = "UNKNOWN"
        base["cvd_label_count"] = 0
    base = base[base["has_ecg"]].copy()  # ECG 硬约束
    if len(base) < target_ecg:
        raise ValueError(f"ECG候选不足: {len(base)} < {target_ecg}")
    base["dischtime"] = pd.to_datetime(base["dischtime"], errors="coerce")
    available_echo = int(base["has_echo"].sum())
    available_cxr = int(base["has_cxr"].sum())
    # echo 使用硬约束：必须达到 target_echo，不允许回退
    if available_echo < int(target_echo):
        raise ValueError(f"echo候选不足: 可用{available_echo} < 目标{target_echo}")
    effective_target_echo = int(target_echo)
    effective_target_cxr = min(int(target_cxr), available_cxr)

    hadm_paths = {
        "cxr": _collect_hadm_modality_paths(details, "cxr", "cxr_path_list"),
        "ecg": _collect_hadm_modality_paths(details, "ecg", "ecg_path_list"),
        "echo": _collect_hadm_modality_paths(details, "echo", "echo_path_list"),
    }
    existing_idx = _build_existing_filename_index(existing_origin_root) if maximize_reuse else {"cxr": set(), "ecg": set(), "echo": set()}

    prefix_set = set(prefix_priority)
    prefix_hits = {}
    reuse_score = {}
    for hid in base["hadm_id"].tolist():
        paths = hadm_paths["cxr"].get(hid, []) + hadm_paths["ecg"].get(hid, []) + hadm_paths["echo"].get(hid, [])
        if disable_prefix_priority:
            prefix_hits[hid] = 0
        else:
            tag_hit = False
            for rp in paths:
                t = _extract_prefix_tag(rp)
                if t and t in prefix_set:
                    tag_hit = True
                    break
            prefix_hits[hid] = int(tag_hit)
        reuse_score[hid] = _estimate_reuse_score_for_hadm(hid, hadm_paths, existing_idx, cxr_format, ecg_expand_pairs) if maximize_reuse else 0

    base["prefix_hit"] = base["hadm_id"].map(prefix_hits).fillna(0).astype(int)
    base["reuse_score"] = base["hadm_id"].map(reuse_score).fillna(0).astype(int)
    base["dischtime_ord"] = base["dischtime"].fillna(pd.Timestamp("1970-01-01")).astype("int64")
    base = base.sort_values(["dischtime", "hadm_id"], ascending=[False, True]).reset_index(drop=True)

    # CVD 均衡目标（软约束优先级高于 prefix/reuse，但低于模态硬约束可行性）
    cvd_targets = _build_balanced_cvd_targets(base, target_hadm)
    cvd_counts = {k: 0 for k in cvd_targets.keys()}

    # 第一层：严格使用“恰好3类CVD”
    pool_len3 = base[base["cvd_label_count"] == 3].copy()
    chosen3, need_echo, need_cxr, cvd_counts = _greedy_select_with_constraints(
        base=pool_len3,
        target_hadm=target_hadm,
        need_echo=effective_target_echo,
        need_cxr=effective_target_cxr,
        cvd_targets=cvd_targets,
        cvd_counts_init=cvd_counts,
    )

    # 第二层：若不足，使用“恰好4类CVD”最小增量补齐
    chosen = list(chosen3)
    cvd_fallback_used = False
    if len(chosen) < target_hadm or need_echo > 0 or need_cxr > 0:
        cvd_fallback_used = True
        pool_len4 = base[base["cvd_label_count"] == 4].copy()
        if not pool_len4.empty:
            chosen4, need_echo, need_cxr, cvd_counts = _greedy_select_with_constraints(
                base=pool_len4,
                target_hadm=(target_hadm - len(chosen)),
                need_echo=need_echo,
                need_cxr=need_cxr,
                cvd_targets=cvd_targets,
                cvd_counts_init=cvd_counts,
            )
            chosen.extend(chosen4)

    if len(chosen) != target_hadm:
        raise ValueError(f"最终hadm数异常: {len(chosen)} != {target_hadm}（len3+len4候选不足）")
    if need_echo > 0 or need_cxr > 0:
        raise ValueError(
            f"模态约束未满足: need_echo={need_echo}, need_cxr={need_cxr}（目标echo={effective_target_echo}, cxr={effective_target_cxr}）"
        )

    # prefix_rule='priority'：不足时自动补其它前缀，无需报错；统计在下游输出。
    if prefix_rule not in {"priority"}:
        raise ValueError(f"不支持的prefix_rule: {prefix_rule}")
    actual_cvd = {}
    hadm_meta: Dict[int, Dict[str, int]] = {}
    for hid in chosen:
        row = base[base["hadm_id"] == hid]
        if not row.empty:
            c = str(row.iloc[0]["primary_cvd"])
            actual_cvd[c] = actual_cvd.get(c, 0) + 1
            hadm_meta[int(hid)] = {
                "cvd_label_count": int(row.iloc[0]["cvd_label_count"]),
                "selection_tier": 3 if int(row.iloc[0]["cvd_label_count"]) == 3 else 4,
            }
    selection_meta = {
        "cvd_len3_count_selected": int(sum(1 for h in chosen if hadm_meta.get(int(h), {}).get("selection_tier") == 3)),
        "cvd_len4_count_selected": int(sum(1 for h in chosen if hadm_meta.get(int(h), {}).get("selection_tier") == 4)),
        "cvd_len4_fallback_used": bool(cvd_fallback_used),
        "fallback_reason": "len3池无法同时满足hadm数量或模态约束，补充len4" if cvd_fallback_used else "",
        "requested_target_echo_hadm": int(target_echo),
        "requested_target_cxr_hadm": int(target_cxr),
        "available_echo_hadm": int(available_echo),
        "available_cxr_hadm": int(available_cxr),
        "min_echo_hadm": int(target_echo),
        "effective_target_echo_hadm": int(effective_target_echo),
        "effective_target_cxr_hadm": int(effective_target_cxr),
        "echo_min_constraint_satisfied": bool(available_echo >= int(target_echo)),
        "echo_hard_constraint_passed": bool(available_echo >= int(target_echo)),
        "echo_target_fallback_used": False,
        "echo_fallback_reason": "",
        "cxr_target_fallback_used": bool(int(effective_target_cxr) < int(target_cxr)),
        "cxr_fallback_reason": (
            f"cxr目标从{int(target_cxr)}回退到{int(effective_target_cxr)}，受候选池可用量限制"
            if int(effective_target_cxr) < int(target_cxr)
            else ""
        ),
        "prefix_priority_enabled": bool(not disable_prefix_priority),
    }
    return chosen, cvd_targets, actual_cvd, hadm_meta, selection_meta


def guess_echo_record_list(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    cands = [
        "/Users/fandong/Desktop/pcl/Data/CVD_MMData/mimiciv/echo/echo-record-list.csv",
        "/Users/fandong/Desktop/pcl/Data/CVD_MMData/mimiciv/echo/echo-record-list.csv.gz",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def expand_echo_paths(details_echo: Optional[pd.DataFrame], selected_hadms: Set[int], echo_record_list_path: Optional[str], enable_expand: bool) -> Dict[int, List[str]]:
    out: Dict[int, Set[str]] = {}
    if details_echo is None or details_echo.empty:
        return {}
    d = details_echo[details_echo["hadm_id"].astype(int).isin(selected_hadms)].copy()
    study_to_hadm = {}
    for _, r in d.iterrows():
        hid = int(r["hadm_id"])
        for sid in parse_listlike(r.get("study_id_list", "[]")):
            s = str(sid).strip()
            if s.isdigit():
                study_to_hadm.setdefault(int(s), set()).add(hid)
        for p in parse_listlike(r.get("echo_path_list", "[]")):
            out.setdefault(hid, set()).add(normalize_rel_path(p))

    if enable_expand:
        rl = guess_echo_record_list(echo_record_list_path)
        if rl and study_to_hadm:
            rec = pd.read_csv(rl, usecols=["study_id", "dicom_filepath"])
            rec = rec[rec["study_id"].isin(list(study_to_hadm.keys()))]
            for _, rr in rec.iterrows():
                sid = int(rr["study_id"])
                rp = normalize_rel_path(rr["dicom_filepath"])
                for hid in study_to_hadm.get(sid, set()):
                    out.setdefault(hid, set()).add(rp)

    return {k: sorted(v) for k, v in out.items()}


def expand_ecg_relpaths(paths: List[str]) -> List[str]:
    out = set()
    for p in paths:
        r = normalize_rel_path(p)
        low = r.lower()
        if low.endswith(".dat"):
            out.add(r)
            out.add(r[:-4] + ".hea")
        elif low.endswith(".hea"):
            out.add(r)
            out.add(r[:-4] + ".dat")
        else:
            out.add(r + ".dat")
            out.add(r + ".hea")
    return sorted(out)


def build_manifest(selected_labels: pd.DataFrame, details: Dict[str, Optional[pd.DataFrame]],
                   cxr_base_url: str, ecg_base_url: str, echo_base_url: str,
                   cxr_format: str, ecg_expand_pairs: bool, echo_expand_all: bool,
                   echo_record_list_path: Optional[str], out_root: str) -> pd.DataFrame:
    hadm2subj = dict(zip(selected_labels["hadm_id"].astype(int), selected_labels["subject_id"].astype(int)))
    selected_hadms = set(hadm2subj.keys())
    rows = []

    # CXR / ECG from details path lists
    for mod, col, base_url in [("cxr", "cxr_path_list", cxr_base_url), ("ecg", "ecg_path_list", ecg_base_url)]:
        df = details.get(mod)
        if df is None or df.empty:
            continue
        df = df[df["hadm_id"].astype(int).isin(selected_hadms)]
        for _, r in df.iterrows():
            hid = int(r["hadm_id"])
            sid = hadm2subj[hid]
            rels = [normalize_rel_path(x) for x in parse_listlike(r.get(col, "[]"))]
            if mod == "cxr" and cxr_format == "jpg":
                rels = [x[:-4] + ".jpg" if x.lower().endswith(".dcm") else (x if x.lower().endswith('.jpg') else x + '.jpg') for x in rels]
            if mod == "ecg" and ecg_expand_pairs:
                rels = expand_ecg_relpaths(rels)
            for rp in rels:
                if not rp:
                    continue
                url = build_full_url(base_url, rp)
                fname = f"{mod}_" + os.path.basename(rp)
                target = os.path.join("origin", mod, str(sid), str(hid), fname)
                rows.append({"modality": mod, "subject_id": sid, "hadm_id": hid, "relative_path": rp, "url": url, "target_path": target})

    # ECHO with study expansion
    echo_map = expand_echo_paths(details.get("echo"), selected_hadms, echo_record_list_path, echo_expand_all)
    for hid, rels in echo_map.items():
        sid = hadm2subj[hid]
        for rp in rels:
            url = build_full_url(echo_base_url, rp)
            fname = "echo_" + os.path.basename(rp)
            target = os.path.join("origin", "echo", str(sid), str(hid), fname)
            rows.append({"modality": "echo", "subject_id": sid, "hadm_id": hid, "relative_path": rp, "url": url, "target_path": target})

    m = pd.DataFrame(rows)
    if not m.empty:
        m = m.drop_duplicates(subset=["modality", "subject_id", "hadm_id", "target_path"]).reset_index(drop=True)
    return m


def reuse_existing(manifest: pd.DataFrame, copy_from_existing_origin_root: Optional[str], out_root: str) -> Dict[str, int]:
    stat = {"linked": 0, "copied": 0, "missing": 0, "exists": 0}
    if manifest.empty:
        return stat
    for _, r in manifest.iterrows():
        dst_rel = str(r["target_path"])
        dst = os.path.join(out_root, dst_rel) if not os.path.isabs(dst_rel) else dst_rel
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            stat["exists"] += 1
            continue
        src = None
        if copy_from_existing_origin_root:
            cand = os.path.join(copy_from_existing_origin_root, r["modality"], os.path.basename(dst).split("_",1)[1])
            if os.path.exists(cand):
                src = cand
        if src is None:
            stat["missing"] += 1
            continue
        try:
            os.link(src, dst)
            stat["linked"] += 1
        except Exception:
            shutil.copy2(src, dst)
            stat["copied"] += 1
    return stat


def download_manifest(manifest: pd.DataFrame, user: Optional[str], password: Optional[str],
                      skip_existing: bool, retry_failed_only: bool, out_root: str,
                      download_workers: int, download_retries: int, download_timeout_sec: int):
    succ = os.path.join(out_root, "download_success.txt")
    fail = os.path.join(out_root, "download_failed.txt")
    if not retry_failed_only:
        open(succ, "w").close()
        open(fail, "w").close()
    else:
        if not os.path.exists(fail):
            open(fail, "w").close()

    rows = manifest.to_dict("records")
    if retry_failed_only and os.path.exists(fail):
        failed_urls = set()
        with open(fail) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    failed_urls.add(parts[1])
        rows = [r for r in rows if r["url"] in failed_urls]

    total = len(rows)
    done = success = failed = skipped = 0
    lock = threading.Lock()
    last_print_ts = 0.0
    start_ts = time.time()

    def _append_line(path: str, text: str) -> None:
        with open(path, "a") as f:
            f.write(text)

    def _run_one(r: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        dst_rel = str(r["target_path"])
        dst = os.path.join(out_root, dst_rel) if not os.path.isabs(dst_rel) else dst_rel
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if skip_existing and os.path.exists(dst):
            return "skipped", r

        # -N 与 -O 冲突，保留 -c 断点续传。
        cmd = ["wget", "-c", "-np", "-O", dst]
        if user:
            cmd += ["--user", str(user)]
        if password:
            cmd += ["--password", str(password)]
        else:
            cmd += ["--ask-password"]
        cmd += [str(r["url"])]

        tries = max(1, int(download_retries) + 1)
        for _ in range(tries):
            ret = subprocess.run(cmd, timeout=int(download_timeout_sec)).returncode
            if ret == 0:
                return "success", r
        return "failed", r

    def _print_progress(force: bool = False) -> None:
        nonlocal last_print_ts
        now = time.time()
        if force or (now - last_print_ts >= 1.0):
            last_print_ts = now
            print(
                f"[download] done={done} total={total} success={success} failed={failed} "
                f"skipped={skipped} remaining={total-done}"
            )

    if total == 0:
        print("[download] done=0 total=0 success=0 failed=0 skipped=0 remaining=0")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(download_workers))) as ex:
        futures = [ex.submit(_run_one, r) for r in rows]
        for fut in concurrent.futures.as_completed(futures):
            status, rec = fut.result()
            line = f"{rec['modality']}\t{rec['url']}\n"
            with lock:
                done += 1
                if status == "success":
                    success += 1
                    _append_line(succ, line)
                elif status == "failed":
                    failed += 1
                    _append_line(fail, line)
                else:
                    skipped += 1
                    _append_line(succ, line)
                _print_progress(force=False)

    _print_progress(force=True)
    elapsed = max(time.time() - start_ts, 1e-6)
    print(f"[download-summary] elapsed_sec={elapsed:.1f} throughput={total/elapsed:.2f} items/sec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--source-mode", choices=["subset2", "step_all"], default="subset2")
    ap.add_argument("--new-output-root", required=True)
    ap.add_argument("--dataset-name", default="subset2_test100_balanced")
    ap.add_argument("--target-hadm", type=int, default=100)
    ap.add_argument("--target-echo-hadm", type=int, default=40)
    ap.add_argument("--min-echo-hadm", type=int, default=40)
    ap.add_argument("--target-cxr-hadm", type=int, default=60)
    ap.add_argument("--target-ecg-hadm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-from-existing-origin-root", default=None)
    ap.add_argument("--existing-origin-root", default=None, help="用于选样复用评分的已下载根目录")
    ap.add_argument("--download-origin-data", action="store_true")
    ap.add_argument("--physionet-user", default=os.environ.get("PHYSIONET_USER"))
    ap.add_argument("--physionet-password", default=os.environ.get("PHYSIONET_PASSWORD"))
    ap.add_argument("--cxr-format", choices=["dcm", "jpg"], default="jpg")
    ap.add_argument("--cxr-base-url", default="https://physionet.org/files/mimic-cxr-jpg/2.1.0")
    ap.add_argument("--ecg-base-url", default="https://physionet.org/files/mimic-iv-ecg/1.0")
    ap.add_argument("--echo-base-url", default="https://physionet.org/files/mimic-iv-echo/0.1")
    ap.add_argument("--ecg-expand-pairs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-expand-all-dcm-by-study", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--echo-record-list-path", default=None)
    ap.add_argument("--skip-existing-before-wget", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retry-failed-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--download-retries", type=int, default=2)
    ap.add_argument("--download-timeout-sec", type=int, default=180)
    ap.add_argument("--prefix-priority", default="p10,p11,p12,p13,p14,p15")
    ap.add_argument("--prefix-rule", choices=["priority"], default="priority")
    ap.add_argument("--disable-prefix-priority", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--maximize-reuse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--restrict-subjects-file", default=None, help="仅从该文件中的subject中筛选（每行一个subject_id）")
    args = ap.parse_args()

    out_root = os.path.join(args.new_output_root, args.dataset_name)
    dirs = ensure_dirs(out_root)

    labels, master, details = load_source(args.source_root, args.source_mode)
    if args.restrict_subjects_file:
        with open(args.restrict_subjects_file) as f:
            allow_subjects = set(int(x.strip()) for x in f if str(x).strip())
        labels = labels[labels["subject_id"].astype(int).isin(allow_subjects)].copy()
    prefix_priority = [x.strip() for x in str(args.prefix_priority).split(",") if x.strip()]
    existing_origin_root = args.existing_origin_root if args.existing_origin_root else args.copy_from_existing_origin_root
    selected_hadms, cvd_target_dist, cvd_actual_dist, hadm_meta, selection_meta = pick_hadms(
        labels=labels,
        details=details,
        target_hadm=args.target_hadm,
        target_echo=args.target_echo_hadm,
        target_cxr=args.target_cxr_hadm,
        target_ecg=args.target_ecg_hadm,
        prefix_priority=prefix_priority,
        prefix_rule=args.prefix_rule,
        maximize_reuse=args.maximize_reuse,
        existing_origin_root=existing_origin_root,
        cxr_format=args.cxr_format,
        ecg_expand_pairs=args.ecg_expand_pairs,
        disable_prefix_priority=args.disable_prefix_priority,
    )
    selected_set = set(selected_hadms)

    selected_labels = labels[labels["hadm_id"].astype(int).isin(selected_set)].copy()
    selected_labels.to_csv(os.path.join(dirs["labels"], "cohort_labels_test100.csv.gz"), index=False, compression="gzip")

    master_f = master[master["hadm_id"].astype(int).isin(selected_set)].copy()
    master_f.to_csv(os.path.join(dirs["timeline"], "master_timeline_test100.csv.gz"), index=False, compression="gzip")

    for mod, df in details.items():
        if df is None:
            continue
        if "hadm_id" in df.columns:
            df2 = df[df["hadm_id"].astype(int).isin(selected_set)].copy()
        else:
            df2 = df.copy()
        df2.to_csv(os.path.join(dirs["details"], f"details_{mod}_grp_test100.csv.gz"), index=False, compression="gzip")

    # 以 subject-hadm 对齐表作为主基准，避免 subjects/hadms 独立列表错位。
    pair_df = (
        selected_labels[["subject_id", "hadm_id"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    pair_df["subject_id"] = pair_df["subject_id"].astype(int)
    pair_df["hadm_id"] = pair_df["hadm_id"].astype(int)
    pair_df["cvd_label_count"] = pair_df["hadm_id"].map(lambda x: int(hadm_meta.get(int(x), {}).get("cvd_label_count", -1)))
    pair_df["selection_tier"] = pair_df["hadm_id"].map(
        lambda x: "len3_primary" if int(hadm_meta.get(int(x), {}).get("selection_tier", 0)) == 3 else "len4_fallback"
    )
    pair_df = pair_df.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)
    pair_df.to_csv(os.path.join(out_root, "test_subject_hadm.csv"), index=False)

    subj = sorted(pair_df["subject_id"].unique().tolist())
    hadm_sorted = sorted(pair_df["hadm_id"].unique().tolist())
    with open(os.path.join(out_root, "test_subjects.txt"), "w") as f:
        for s in subj:
            f.write(f"{s}\n")
    with open(os.path.join(out_root, "test_hadms.txt"), "w") as f:
        for h in hadm_sorted:
            f.write(f"{h}\n")

    # 一致性校验
    alignment_check_passed = True
    if len(pair_df) != args.target_hadm:
        alignment_check_passed = False
    if pair_df["hadm_id"].nunique() != args.target_hadm:
        alignment_check_passed = False
    if set(hadm_sorted) != selected_set:
        alignment_check_passed = False
    if args.restrict_subjects_file:
        if not set(pair_df["subject_id"].astype(int).tolist()).issubset(allow_subjects):
            alignment_check_passed = False

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

    # target_path 使用相对路径，并执行格式校验
    rel_check = True
    if not manifest.empty:
        for tp in manifest["target_path"].astype(str):
            if tp.startswith("/") or (not tp.startswith("origin/")):
                rel_check = False
                break
    manifest.to_csv(os.path.join(out_root, "origin_manifest_test100.csv.gz"), index=False, compression="gzip")

    reuse_stat = reuse_existing(manifest, args.copy_from_existing_origin_root, out_root=out_root)

    # stats
    hs = hadm_sets(details)
    actual_ecg = sum(1 for h in selected_set if h in hs["ecg"])
    actual_echo = sum(1 for h in selected_set if h in hs["echo"])
    actual_cxr = sum(1 for h in selected_set if h in hs["cxr"])
    prefix_set = set(prefix_priority)
    hit_rows = 0
    total_rows = 0
    if not manifest.empty and "relative_path" in manifest.columns:
        for rp in manifest["relative_path"].astype(str):
            total_rows += 1
            tag = _extract_prefix_tag(rp)
            if tag in prefix_set:
                hit_rows += 1
    prefix_ratio = (hit_rows / total_rows) if total_rows > 0 else 0.0
    fallback_outside = total_rows - hit_rows
    reused_count = int(reuse_stat.get("linked", 0) + reuse_stat.get("copied", 0) + reuse_stat.get("exists", 0))
    new_download_estimate = max(int(len(manifest) - reused_count), 0)
    dist = {
        "hadm_count": len(selected_set),
        "subject_count": len(subj),
        "subject_hadm_pairs_count": int(len(pair_df)),
        "subject_unique_count": int(pair_df["subject_id"].nunique()),
        "hadm_unique_count": int(pair_df["hadm_id"].nunique()),
        "alignment_check_passed": bool(alignment_check_passed),
        "target_ecg_hadm": int(args.target_ecg_hadm),
        "target_echo_hadm": int(args.target_echo_hadm),
        "target_cxr_hadm": int(args.target_cxr_hadm),
        "requested_target_echo_hadm": int(selection_meta.get("requested_target_echo_hadm", args.target_echo_hadm)),
        "requested_target_cxr_hadm": int(selection_meta.get("requested_target_cxr_hadm", args.target_cxr_hadm)),
        "source_mode": args.source_mode,
        "min_echo_hadm": int(selection_meta.get("min_echo_hadm", args.target_echo_hadm)),
        "available_echo_hadm": int(selection_meta.get("available_echo_hadm", actual_echo)),
        "available_cxr_hadm": int(selection_meta.get("available_cxr_hadm", actual_cxr)),
        "effective_target_echo_hadm": int(selection_meta.get("effective_target_echo_hadm", args.target_echo_hadm)),
        "effective_target_cxr_hadm": int(selection_meta.get("effective_target_cxr_hadm", args.target_cxr_hadm)),
        "echo_min_constraint_satisfied": bool(selection_meta.get("echo_min_constraint_satisfied", True)),
        "echo_hard_constraint_passed": bool(selection_meta.get("echo_hard_constraint_passed", True)),
        "echo_target_fallback_used": bool(selection_meta.get("echo_target_fallback_used", False)),
        "echo_fallback_reason": str(selection_meta.get("echo_fallback_reason", "")),
        "cxr_target_fallback_used": bool(selection_meta.get("cxr_target_fallback_used", False)),
        "cxr_fallback_reason": str(selection_meta.get("cxr_fallback_reason", "")),
        "actual_ecg_hadm": int(actual_ecg),
        "actual_echo_hadm": int(actual_echo),
        "actual_cxr_hadm": int(actual_cxr),
        "prefix_priority": prefix_priority,
        "prefix_priority_enabled": bool(selection_meta.get("prefix_priority_enabled", not args.disable_prefix_priority)),
        "prefix_rule": args.prefix_rule,
        "prefix_priority_hit_ratio": float(prefix_ratio),
        "fallback_outside_p10_p15_count": int(fallback_outside),
        "reused_file_count": int(reused_count),
        "new_download_estimate": int(new_download_estimate),
        "target_cvd_distribution": cvd_target_dist,
        "actual_cvd_distribution": cvd_actual_dist,
        "cvd_len3_count_selected": int(selection_meta.get("cvd_len3_count_selected", 0)),
        "cvd_len4_count_selected": int(selection_meta.get("cvd_len4_count_selected", 0)),
        "cvd_len4_fallback_used": bool(selection_meta.get("cvd_len4_fallback_used", False)),
        "fallback_reason": str(selection_meta.get("fallback_reason", "")),
        "manifest_rows": int(len(manifest)),
        "target_path_mode": "relative",
        "manifest_relative_path_check_passed": bool(rel_check),
        "reuse": reuse_stat,
    }
    with open(os.path.join(out_root, "stats_test100.json"), "w") as f:
        json.dump(dist, f, indent=2, ensure_ascii=False)

    print(dist)

    if args.download_origin_data:
        download_manifest(
            manifest=manifest,
            user=args.physionet_user,
            password=args.physionet_password,
            skip_existing=args.skip_existing_before_wget,
            retry_failed_only=args.retry_failed_only,
            out_root=out_root,
            download_workers=args.download_workers,
            download_retries=args.download_retries,
            download_timeout_sec=args.download_timeout_sec,
        )


if __name__ == "__main__":
    main()
