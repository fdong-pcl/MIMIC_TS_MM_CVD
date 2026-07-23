import argparse
import ast
import json
import os
import shlex
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_DATA_DIR = "/Users/fandong/Desktop/pcl/Data/CVD_MMData"
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT", os.path.join(BASE_DATA_DIR, "temporal_output_norm_icd10")
)

DETAIL_MODALITIES = ["notes", "cxr", "ecg", "echo", "lab", "rx", "proc"]


# =============================
# 基础工具函数
# =============================
def parse_cvd_list(val) -> List[str]:
    """
    输入:
      - val: Step1 输出中的 CVD_coarse_category 字段，通常是字符串化 list
    输出:
      - 标准化后的字符串列表（去空值）
    关键约束:
      - 必须兼容 list / tuple / set / 字符串化 list / 单值字符串。
    """
    if pd.isna(val):
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]

    text = str(val).strip()
    if not text:
        return []

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple, set)):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    return [text]


def setup_subset_dirs(output_root: str, subset_name: str) -> Dict[str, str]:
    """
    输入:
      - output_root: 总输出目录（默认 temporal_output_norm_icd10）
      - subset_name: subset1 / subset2
    输出:
      - 各子目录路径字典
    关键约束:
      - 目录结构固定，便于下游读取和复用。
    """
    base = os.path.join(output_root, subset_name)
    dirs = {
        "base": base,
        "labels": os.path.join(base, "labels"),
        "splits": os.path.join(base, "splits"),
        "timeline": os.path.join(base, "timeline"),
        "details": os.path.join(base, "details"),
        "origin_data": os.path.join(base, "origin_data"),
        "stats": os.path.join(base, "stats"),
        "log": os.path.join(base, "log"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def parse_listlike(val) -> List[str]:
    """把字符串化 list / 单值字符串统一解析为字符串列表。"""
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


# =============================
# 数据加载函数
# =============================
def load_inputs(mode: str, output_root: str) -> Dict[str, object]:
    """
    输入:
      - mode: DEBUG / ALL
      - output_root: Step0-4 输出根目录
    输出:
      - 包含 step1 cohort / master timeline / details 字典
    关键约束:
      - 不重跑 Step0-4，只读取既有产物。
      - 如果核心输入缺失，直接抛错终止。
    """
    suffix = mode.lower()

    step1_path = os.path.join(output_root, "step1_cvd_filter", f"step_1_cvd_cohort_{suffix}.csv.gz")
    master_path = os.path.join(output_root, "step3_temporal_timeline", f"master_timeline_{suffix}.csv.gz")

    if not os.path.exists(step1_path):
        raise FileNotFoundError(f"找不到 Step1 结果: {step1_path}")
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"找不到 Step3 master timeline: {master_path}")

    df_cvd = pd.read_csv(step1_path)
    df_master = pd.read_csv(master_path)

    details = {}
    for mod in DETAIL_MODALITIES:
        p = os.path.join(output_root, "step3_temporal_timeline", f"details_{mod}_grp_{suffix}.csv.gz")
        details[mod] = _read_csv_if_exists(p)

    return {
        "suffix": suffix,
        "df_cvd": df_cvd,
        "df_master": df_master,
        "details": details,
        "input_step1": step1_path,
        "input_master": master_path,
    }


# =============================
# 切分与校验函数
# =============================
def subject_level_split(
    subject_ids: List[int],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, Set[int]]:
    """
    输入:
      - subject_ids: 待切分 subject 列表
      - test_ratio/val_ratio: 比例参数，train 比例由剩余自动决定
      - seed: 随机种子
    输出:
      - {'train': set, 'val': set, 'test': set}
    关键约束:
      - 这是“先 subject 后 hadm”的核心，避免患者跨集合泄露。
    """
    subjects = np.array(sorted(set(subject_ids)))
    if len(subjects) == 0:
        return {"train": set(), "val": set(), "test": set()}
    if len(subjects) < 3:
        # 小样本兜底: 按顺序尽量分配
        train, val, test = set(subjects), set(), set()
        if len(subjects) >= 2:
            val.add(subjects[-1])
            train.remove(subjects[-1])
        if len(subjects) >= 3:
            test.add(subjects[-2])
            train.remove(subjects[-2])
        return {"train": train, "val": val, "test": test}

    train_val, test = train_test_split(subjects, test_size=test_ratio, random_state=seed)
    # val_ratio 是相对全集；换算到 train_val 内部比例
    internal_val_ratio = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(train_val, test_size=internal_val_ratio, random_state=seed)

    return {
        "train": set(train.tolist()),
        "val": set(val.tolist()),
        "test": set(test.tolist()),
    }


def validate_splits(
    split_subjects: Dict[str, Set[int]],
    hadm_to_subject: Dict[int, int],
    split_hadms: Dict[str, Set[int]],
    test_target: Optional[int] = None,
    tolerance: Optional[int] = None,
    min_test_cases: Optional[int] = None,
) -> None:
    """
    输入:
      - split_subjects: 三个集合的 subject
      - hadm_to_subject: hadm -> subject 映射
      - split_hadms: 三个集合的 hadm
      - test_target/tolerance: 兼容旧逻辑的测试集目标区间校验参数
      - min_test_cases: 新逻辑，要求 test 病例数至少达到该值
    输出:
      - 无返回，失败时抛异常
    关键约束:
      - 校验失败必须中断写出，防止产生错误数据集。
    """
    train_s, val_s, test_s = split_subjects["train"], split_subjects["val"], split_subjects["test"]
    if train_s & val_s:
        raise ValueError("切分失败: train/val 存在 subject 重叠")
    if train_s & test_s:
        raise ValueError("切分失败: train/test 存在 subject 重叠")
    if val_s & test_s:
        raise ValueError("切分失败: val/test 存在 subject 重叠")

    # 校验 hadm 只能归属一个集合
    union_all = set()
    for name in ["train", "val", "test"]:
        overlap = union_all & split_hadms[name]
        if overlap:
            raise ValueError(f"切分失败: hadm 跨集合重叠，示例: {list(sorted(overlap))[:5]}")
        union_all.update(split_hadms[name])

    # 校验 hadm 的 subject 归属与 split_subjects 一致
    for name in ["train", "val", "test"]:
        allowed_subjects = split_subjects[name]
        for hid in split_hadms[name]:
            sid = hadm_to_subject.get(int(hid))
            if sid not in allowed_subjects:
                raise ValueError(f"切分失败: {name} 中 hadm={hid} 的 subject={sid} 不在该集合")

    if min_test_cases is not None:
        n_test = len(split_hadms["test"])
        if n_test < min_test_cases:
            raise ValueError(
                f"切分失败: test 病例数 {n_test} 小于最低要求 {min_test_cases}"
            )

    # 兼容旧校验逻辑（若外部仍显式传入 target+tolerance）
    if min_test_cases is None and test_target is not None and tolerance is not None:
        n_test = len(split_hadms["test"])
        if not (test_target - tolerance <= n_test <= test_target + tolerance):
            raise ValueError(
                f"切分失败: test 病例数 {n_test} 不在目标区间 [{test_target-tolerance}, {test_target+tolerance}]"
            )


# =============================
# subset1 构建
# =============================
def build_subset1(df_cvd: pd.DataFrame, seed: int) -> Dict[str, object]:
    """
    输入:
      - df_cvd: Step1 hadm 聚合结果
    输出:
      - subset1 的基础数据结构（hadm池 + split +统计）
    关键约束:
      - 严格筛选“恰好3类 CVD_coarse_category”。
      - 切分必须基于 subject，防止同一患者跨集合。
    """
    work = df_cvd.copy()
    work["cvd_list"] = work["CVD_coarse_category"].apply(parse_cvd_list)
    work["cvd_unique_count"] = work["cvd_list"].apply(lambda x: len(set(x)))

    subset = work[work["cvd_unique_count"] == 3].copy()
    subset = subset.drop_duplicates(subset=["subject_id", "hadm_id"])

    subject_ids = subset["subject_id"].astype(int).tolist()
    split_subjects = subject_level_split(subject_ids, test_ratio=0.1, val_ratio=0.2, seed=seed)

    split_hadms: Dict[str, Set[int]] = {"train": set(), "val": set(), "test": set()}
    hadm_to_subject: Dict[int, int] = {}
    for _, r in subset[["subject_id", "hadm_id"]].iterrows():
        sid = int(r["subject_id"])
        hid = int(r["hadm_id"])
        hadm_to_subject[hid] = sid
        if sid in split_subjects["train"]:
            split_hadms["train"].add(hid)
        elif sid in split_subjects["val"]:
            split_hadms["val"].add(hid)
        elif sid in split_subjects["test"]:
            split_hadms["test"].add(hid)

    validate_splits(split_subjects, hadm_to_subject, split_hadms)

    stats = {
        "subject_count": int(subset["subject_id"].nunique()),
        "hadm_count": int(subset["hadm_id"].nunique()),
        "train_subject_count": int(len(split_subjects["train"])),
        "val_subject_count": int(len(split_subjects["val"])),
        "test_subject_count": int(len(split_subjects["test"])),
        "train_hadm_count": int(len(split_hadms["train"])),
        "val_hadm_count": int(len(split_hadms["val"])),
        "test_hadm_count": int(len(split_hadms["test"])),
    }

    return {
        "labels": subset,
        "split_subjects": split_subjects,
        "split_hadms": split_hadms,
        "hadm_to_subject": hadm_to_subject,
        "stats": stats,
    }


# =============================
# subset2 构建
# =============================
def compute_modality_coverage(details: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    """
    输入:
      - details: Step3 各模态分组明细表
    输出:
      - hadm 级模态覆盖表（has_cxr/has_ecg/has_echo/modality_score）
    关键约束:
      - 模态优先规则只看 CXR/ECG/ECHO 三类。
    """
    frames = []
    for mod, flag_col in [("cxr", "has_cxr"), ("ecg", "has_ecg"), ("echo", "has_echo")]:
        df = details.get(mod)
        if df is None or df.empty or "hadm_id" not in df.columns:
            continue
        tmp = df[["hadm_id"]].drop_duplicates().copy()
        tmp[flag_col] = 1
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["hadm_id", "has_cxr", "has_ecg", "has_echo", "modality_score"])

    cov = frames[0]
    for f in frames[1:]:
        cov = cov.merge(f, on="hadm_id", how="outer")

    for c in ["has_cxr", "has_ecg", "has_echo"]:
        if c not in cov.columns:
            cov[c] = 0
    cov[["has_cxr", "has_ecg", "has_echo"]] = cov[["has_cxr", "has_ecg", "has_echo"]].fillna(0).astype(int)
    cov["modality_score"] = cov[["has_cxr", "has_ecg", "has_echo"]].sum(axis=1)
    cov["hadm_id"] = cov["hadm_id"].astype(int)
    return cov


def _greedy_pick_subjects_for_target(
    subject_to_hadm: Dict[int, List[int]],
    subject_rank: List[int],
    target_hadm: int,
) -> Set[int]:
    """
    输入:
      - subject_to_hadm: subject -> hadm 列表
      - subject_rank: 已按优先级排序的 subject 顺序
      - target_hadm: 目标病例数
    输出:
      - 被选中的 subject 集合
    关键约束:
      - 这里是“subject隔离 + hadm数量目标”折中核心。
      - 使用贪心使累计 hadm 更接近目标。
    """
    chosen: Set[int] = set()
    current = 0

    for sid in subject_rank:
        cnt = len(subject_to_hadm.get(sid, []))
        if cnt <= 0:
            continue
        if current < target_hadm:
            # 若尚未到目标，优先吸纳；若会超，比较“吸纳后”和“不吸纳”哪个更接近目标
            if abs((current + cnt) - target_hadm) <= abs(current - target_hadm):
                chosen.add(sid)
                current += cnt

    # 若一次遍历不足目标，则继续从未选subject补齐
    if current < target_hadm:
        for sid in subject_rank:
            if sid in chosen:
                continue
            cnt = len(subject_to_hadm.get(sid, []))
            if cnt <= 0:
                continue
            chosen.add(sid)
            current += cnt
            if current >= target_hadm:
                break

    return chosen


def _subject_split_with_test_target(
    candidate_subjects: List[int],
    subject_to_hadm: Dict[int, List[int]],
    seed: int,
    test_target: int,
) -> Dict[str, Set[int]]:
    """
    输入:
      - candidate_subjects: 候选 subject
      - subject_to_hadm: subject -> hadm 列表
      - test_target: 目标测试病例数（hadm）
    输出:
      - subject 级 train/val/test 分配
    关键约束:
      - subject 不可跨集合；
      - test 集在 hadm 维度尽量靠近目标值。
    """
    rng = np.random.default_rng(seed)
    subjects = list(sorted(set(candidate_subjects)))
    rng.shuffle(subjects)

    # 先按 hadm 数降序、模态优先外部已排序，这里维持输入顺序为主，仅稳定排序
    rank = subjects
    test_subjects = _greedy_pick_subjects_for_target(subject_to_hadm, rank, test_target)

    # 达到目标后，继续做“最小超出”优化：
    # 在不低于 test_target 的前提下，尽量减少 test hadm 数量。
    def _test_hadm_count(subj_set: Set[int]) -> int:
        return sum(len(subject_to_hadm.get(s, [])) for s in subj_set)

    current_test = _test_hadm_count(test_subjects)
    if current_test >= test_target and len(test_subjects) > 0:
        improved = True
        while improved:
            improved = False
            # 1) 先尝试直接移除某个 subject（如果仍 >= target 且更小）
            for sid in list(test_subjects):
                new_set = set(test_subjects)
                new_set.remove(sid)
                new_cnt = _test_hadm_count(new_set)
                if test_target <= new_cnt < current_test:
                    test_subjects = new_set
                    current_test = new_cnt
                    improved = True
                    break

            if improved:
                continue

            # 2) 再尝试交换: test 内一个 subject 换成外部一个 subject，降低超出但仍 >= target
            outside = [s for s in rank if s not in test_subjects]
            for s_in in list(test_subjects):
                for s_out in outside:
                    new_set = set(test_subjects)
                    new_set.remove(s_in)
                    new_set.add(s_out)
                    new_cnt = _test_hadm_count(new_set)
                    if test_target <= new_cnt < current_test:
                        test_subjects = new_set
                        current_test = new_cnt
                        improved = True
                        break
                if improved:
                    break

    remain = [s for s in subjects if s not in test_subjects]
    # 对剩余集合按 7:2 分 train/val（对应总7:2:1）
    if len(remain) <= 1:
        train_subjects = set(remain)
        val_subjects = set()
    else:
        train_arr, val_arr = train_test_split(np.array(remain), test_size=2 / 9, random_state=seed)
        train_subjects = set(train_arr.tolist())
        val_subjects = set(val_arr.tolist())

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": set(test_subjects),
    }


def build_subset2_from_subset1(
    subset1_labels: pd.DataFrame,
    details_subset1: Dict[str, Optional[pd.DataFrame]],
    seed: int,
    target_hadm: int,
    target_test_hadm: int,
    test_tolerance: int,
    one_hadm_per_subject: bool = True,
) -> Dict[str, object]:
    """
    输入:
      - subset1_labels: subset1 hadm 标签表
      - details_subset1: subset1 各模态 details 表
      - target_hadm: subset2 总目标病例数（hadm）
      - target_test_hadm: 测试集目标病例数（hadm）
      - test_tolerance: 测试集容忍区间
    输出:
      - subset2 的 labels/split/stats 等结构
    关键约束:
      - 先在 hadm 级别硬过滤: modality_score >= 2（至少两种模态）。
      - 默认每个 subject 只保留一次最佳住院（one_hadm_per_subject=True）。
      - 再按模态覆盖优先选样，并做 subject 级隔离切分。
      - test 集要求 hadm 数至少达到目标值，且尽量“最小超出”。
    """
    labels = subset1_labels.copy().drop_duplicates(subset=["subject_id", "hadm_id"])

    coverage = compute_modality_coverage(details_subset1)
    labels = labels.merge(coverage, on="hadm_id", how="left")
    for c in ["has_cxr", "has_ecg", "has_echo", "modality_score"]:
        if c not in labels.columns:
            labels[c] = 0
    labels[["has_cxr", "has_ecg", "has_echo", "modality_score"]] = labels[
        ["has_cxr", "has_ecg", "has_echo", "modality_score"]
    ].fillna(0).astype(int)

    # 硬约束: subset2 只保留至少 2 模态的住院记录（hadm）。
    # 这一步直接满足你提出的条件:
    # - 若同一患者 N 次住院中仅 A 次 >=2 模态，则仅保留这 A 次；
    # - 后续 train/test/val 里的 hadm 全部都会来自这个过滤结果。
    total_before_filter = int(labels["hadm_id"].nunique())
    labels = labels[labels["modality_score"] >= 2].copy()
    labels = labels.drop_duplicates(subset=["subject_id", "hadm_id"])
    total_after_filter = int(labels["hadm_id"].nunique())

    if total_after_filter == 0:
        raise ValueError("subset2 构建失败: 满足 >=2 模态条件的 hadm 数量为 0。")
    if total_after_filter < target_test_hadm:
        raise ValueError(
            f"subset2 构建失败: >=2 模态 hadm 总数仅 {total_after_filter}，小于 test 目标 {target_test_hadm}。"
        )

    # 多模态优先 + 时间优先（越新越前）+ hadm_id 稳定排序
    if "dischtime" in labels.columns:
        labels["dischtime_dt"] = pd.to_datetime(labels["dischtime"], errors="coerce")
    else:
        labels["dischtime_dt"] = pd.NaT

    labels = labels.sort_values(
        by=["modality_score", "dischtime_dt", "hadm_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # 新策略: 每个 subject 仅保留一次住院记录（优先最高模态覆盖，若并列优先最新出院时间）。
    # 这样 train/val/test 内部与之间都能保持 subject 隔离，同时 hadm 数可直接由 subject 数控制。
    if one_hadm_per_subject:
        labels = labels.drop_duplicates(subset=["subject_id"], keep="first").copy()

    # 先构造 subject 的优先级（使用该 subject 最高模态覆盖 + 最新出院时间）
    subj_rank_df = (
        labels.groupby("subject_id", as_index=False)
        .agg(best_score=("modality_score", "max"), last_dischtime=("dischtime_dt", "max"), hadm_count=("hadm_id", "nunique"))
        .sort_values(by=["best_score", "last_dischtime", "hadm_count", "subject_id"], ascending=[False, False, False, True])
    )

    subject_rank = subj_rank_df["subject_id"].astype(int).tolist()

    subject_to_hadm: Dict[int, List[int]] = (
        labels.groupby("subject_id")["hadm_id"].apply(lambda x: sorted(set(map(int, x.tolist())))).to_dict()
    )

    # 先从高优先级 subject 中选出接近 target_hadm 的池
    selected_subjects = _greedy_pick_subjects_for_target(subject_to_hadm, subject_rank, target_hadm)
    selected_labels = labels[labels["subject_id"].isin(selected_subjects)].copy()

    # 若超出目标，从低优先级 hadm 开始裁剪（但保持 subject 隔离: 只能整 subject 裁剪）
    # 因为 subject 粒度约束，最终 hadm 可能不完全等于目标，这里允许接近。
    # 若远超目标，尝试删除“优先级最低且病例数较多”的 subject 直到更接近目标。
    def total_hadm(sset: Set[int]) -> int:
        return sum(len(subject_to_hadm.get(s, [])) for s in sset)

    current_total = total_hadm(selected_subjects)
    if current_total > target_hadm:
        # 反向序: 优先移除优先级低的 subject
        reverse_rank = list(reversed(subject_rank))
        improved = True
        while improved:
            improved = False
            for sid in reverse_rank:
                if sid not in selected_subjects:
                    continue
                new_set = set(selected_subjects)
                new_set.remove(sid)
                if not new_set:
                    continue
                new_total = total_hadm(new_set)
                if abs(new_total - target_hadm) <= abs(current_total - target_hadm):
                    selected_subjects = new_set
                    current_total = new_total
                    improved = True

    selected_labels = labels[labels["subject_id"].isin(selected_subjects)].copy()

    # 在已选 subject 上做 7:2:1 的 subject 级切分，并尽量让 test_hadm 接近 target_test_hadm
    split_subjects = _subject_split_with_test_target(
        candidate_subjects=list(selected_subjects),
        subject_to_hadm=subject_to_hadm,
        seed=seed,
        test_target=target_test_hadm,
    )

    split_hadms: Dict[str, Set[int]] = {"train": set(), "val": set(), "test": set()}
    hadm_to_subject: Dict[int, int] = {}
    for _, r in selected_labels[["subject_id", "hadm_id"]].drop_duplicates().iterrows():
        sid = int(r["subject_id"])
        hid = int(r["hadm_id"])
        hadm_to_subject[hid] = sid
        if sid in split_subjects["train"]:
            split_hadms["train"].add(hid)
        elif sid in split_subjects["val"]:
            split_hadms["val"].add(hid)
        elif sid in split_subjects["test"]:
            split_hadms["test"].add(hid)

    validate_splits(
        split_subjects,
        hadm_to_subject,
        split_hadms,
        min_test_cases=target_test_hadm,
    )

    stats = {
        "subject_count": int(selected_labels["subject_id"].nunique()),
        "hadm_count": int(selected_labels["hadm_id"].nunique()),
        "train_subject_count": int(len(split_subjects["train"])),
        "val_subject_count": int(len(split_subjects["val"])),
        "test_subject_count": int(len(split_subjects["test"])),
        "train_hadm_count": int(len(split_hadms["train"])),
        "val_hadm_count": int(len(split_hadms["val"])),
        "test_hadm_count": int(len(split_hadms["test"])),
        "target_hadm": int(target_hadm),
        "target_test_hadm": int(target_test_hadm),
        "test_tolerance": int(test_tolerance),
        "hadm_total_before_modality_filter": total_before_filter,
        "hadm_total_after_modality_filter_ge2": total_after_filter,
        "one_hadm_per_subject": bool(one_hadm_per_subject),
        "modality_score_distribution": selected_labels["modality_score"].value_counts().sort_index().to_dict(),
    }

    selected_labels = selected_labels.drop(columns=["dischtime_dt"], errors="ignore")

    return {
        "labels": selected_labels,
        "split_subjects": split_subjects,
        "split_hadms": split_hadms,
        "hadm_to_subject": hadm_to_subject,
        "stats": stats,
    }


# =============================
# 写出函数
# =============================
def filter_by_hadm(df: Optional[pd.DataFrame], hadm_set: Set[int]) -> Optional[pd.DataFrame]:
    """按 hadm 过滤任意明细表。若表为空或无 hadm_id 列则原样返回。"""
    if df is None:
        return None
    if df.empty:
        return df.copy()
    if "hadm_id" not in df.columns:
        return df.copy()
    return df[df["hadm_id"].astype(int).isin(hadm_set)].copy()


def _normalize_rel_path(p: str) -> str:
    p = str(p).strip()
    p = p.replace("\\", "/")
    while p.startswith("/"):
        p = p[1:]
    return p


def _collect_modality_relpaths(details_subset2: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, List[str]]:
    """
    从 subset2 details 表提取三种模态的相对路径列表。
    兼容 *_path_list 为字符串化 list 的情况。
    """
    relpaths = {"cxr": set(), "ecg": set(), "echo": set()}
    mapping = {
        "cxr": ("cxr", "cxr_path_list"),
        "ecg": ("ecg", "ecg_path_list"),
        "echo": ("echo", "echo_path_list"),
    }
    for mod, (key, col) in mapping.items():
        df = details_subset2.get(key)
        if df is None or df.empty or col not in df.columns:
            continue
        for v in df[col].tolist():
            for p in parse_listlike(v):
                rp = _normalize_rel_path(p)
                if rp:
                    relpaths[mod].add(rp)
    return {k: sorted(v) for k, v in relpaths.items()}


def _collect_cxr_dicom_ids(details_subset2: Dict[str, Optional[pd.DataFrame]]) -> List[str]:
    """从 subset2 的 CXR details 提取 dicom_id 列表。"""
    df = details_subset2.get("cxr")
    if df is None or df.empty or "dicom_id_list" not in df.columns:
        return []
    ids = set()
    for v in df["dicom_id_list"].tolist():
        for d in parse_listlike(v):
            ds = str(d).strip()
            if ds:
                ids.add(ds)
    return sorted(ids)


def _guess_cxr_index_path(explicit_path: Optional[str]) -> Optional[str]:
    """猜测本地可用的 CXR 路径索引文件。"""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    candidates = [
        os.path.join(BASE_DATA_DIR, "mimiciv/cxr/cxr-record-list.csv.gz"),
        os.path.join(BASE_DATA_DIR, "mimiciv/cxr/mimic-cxr-2.1.0-record-list.csv.gz"),
        os.path.join(BASE_DATA_DIR, "mimiciv/cxr/mimic-cxr-2.0.0-record-list.csv.gz"),
        os.path.join(BASE_DATA_DIR, "mimiciv/cxr/record_list.csv.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _build_cxr_paths_from_index(dicom_ids: List[str], cxr_index_path: str, cxr_format: str) -> List[str]:
    """
    使用本地索引文件把 dicom_id 映射到相对路径。
    - cxr_format=dcm: 直接用索引路径
    - cxr_format=jpg: 优先使用索引中的 jpg 路径，否则做后缀替换
    """
    idx = pd.read_csv(cxr_index_path, usecols=["dicom_id", "path"], dtype={"dicom_id": str, "path": str})
    idx = idx.dropna(subset=["dicom_id", "path"]).drop_duplicates(subset=["dicom_id"], keep="first")
    idx_map = dict(zip(idx["dicom_id"].astype(str), idx["path"].astype(str)))

    rels = []
    for did in dicom_ids:
        p = idx_map.get(str(did))
        if not p:
            continue
        rp = _normalize_rel_path(p)
        if cxr_format == "jpg":
            if rp.lower().endswith(".dcm"):
                rp = rp[:-4] + ".jpg"
            elif not rp.lower().endswith(".jpg"):
                rp = rp + ".jpg"
        rels.append(rp)
    return sorted(set(rels))


def _build_full_url(base_url: str, rel_path: str) -> str:
    return base_url.rstrip("/") + "/" + rel_path.lstrip("/")


def prepare_origin_data_download(
    subset2_dirs: Dict[str, str],
    details_subset2: Dict[str, Optional[pd.DataFrame]],
    cxr_base_url: str,
    ecg_base_url: str,
    echo_base_url: str,
    cxr_format: str = "dcm",
    cxr_index_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    生成 subset2 原始数据下载清单（manifest）与 wget 脚本，不立即下载。
    """
    rels = _collect_modality_relpaths(details_subset2)

    # CXR: 按目标格式分支。jpg 模式优先用 dicom_id + 本地官方索引构建路径。
    if cxr_format not in {"dcm", "jpg"}:
        raise ValueError(f"不支持的 cxr_format: {cxr_format}")
    if cxr_format == "jpg":
        dicom_ids = _collect_cxr_dicom_ids(details_subset2)
        guessed = _guess_cxr_index_path(cxr_index_path)
        if guessed:
            rels["cxr"] = _build_cxr_paths_from_index(dicom_ids, guessed, cxr_format="jpg")
        else:
            # 回退: 基于现有路径做后缀替换（准确率可能低于官方索引）
            rels["cxr"] = [rp[:-4] + ".jpg" if rp.lower().endswith(".dcm") else rp for rp in rels["cxr"]]

    rows = []
    base_map = {"cxr": cxr_base_url, "ecg": ecg_base_url, "echo": echo_base_url}
    for mod in ["cxr", "ecg", "echo"]:
        for rp in rels[mod]:
            rows.append(
                {
                    "modality": mod,
                    "relative_path": rp,
                    "url": _build_full_url(base_map[mod], rp),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(subset2_dirs["origin_data"], "origin_download_manifest.csv.gz")
    manifest.to_csv(manifest_path, index=False, compression="gzip")
    return manifest


def write_wget_script(
    script_path: str,
    manifest: pd.DataFrame,
    origin_data_dir: str,
    physionet_user: Optional[str],
    physionet_password: Optional[str],
    wget_bin: str = "wget",
) -> None:
    """
    生成可执行 wget 脚本。
    - 推荐使用 --ask-password 或环境变量，避免明文密码写入脚本。
    """
    lines_tsv = os.path.join(origin_data_dir, "origin_download_manifest.tsv")
    success_file = os.path.join(origin_data_dir, "download_success.txt")
    failed_file = os.path.join(origin_data_dir, "download_failed.txt")
    manifest[["modality", "url"]].to_csv(lines_tsv, sep="\t", index=False, header=False)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -u\n")
        f.write(f'ORIGIN_DIR={shlex.quote(origin_data_dir)}\n')
        f.write(f'MANIFEST_TSV={shlex.quote(lines_tsv)}\n')
        f.write(f'SUCCESS_FILE={shlex.quote(success_file)}\n')
        f.write(f'FAILED_FILE={shlex.quote(failed_file)}\n')
        f.write("mkdir -p \"$ORIGIN_DIR/cxr\" \"$ORIGIN_DIR/ecg\" \"$ORIGIN_DIR/echo\"\n")
        f.write(": > \"$SUCCESS_FILE\"\n: > \"$FAILED_FILE\"\n")
        f.write("TOTAL=$(wc -l < \"$MANIFEST_TSV\" | tr -d ' ')\n")
        f.write("DONE=0\nFAILED=0\n")
        f.write("while IFS=$'\\t' read -r MOD URL; do\n")
        f.write("  [ -z \"$MOD\" ] && continue\n")
        f.write("  DONE=$((DONE+1))\n")
        f.write("  TARGET_DIR=\"$ORIGIN_DIR/$MOD\"\n")
        base_cmd = f"{shlex.quote(wget_bin)} -N -c -np"
        if physionet_user:
            base_cmd += f" --user {shlex.quote(physionet_user)}"
        if physionet_password:
            base_cmd += f" --password {shlex.quote(physionet_password)}"
        else:
            base_cmd += " --ask-password"
        f.write(f"  if {base_cmd} -P \"$TARGET_DIR\" \"$URL\"; then\n")
        f.write("    echo \"$MOD\\t$URL\" >> \"$SUCCESS_FILE\"\n")
        f.write("  else\n")
        f.write("    FAILED=$((FAILED+1))\n")
        f.write("    echo \"$MOD\\t$URL\" >> \"$FAILED_FILE\"\n")
        f.write("  fi\n")
        f.write("  REMAIN=$((TOTAL-DONE))\n")
        f.write("  printf '[download] done=%d total=%d failed=%d remaining=%d\\n' \"$DONE\" \"$TOTAL\" \"$FAILED\" \"$REMAIN\"\n")
        f.write("done < \"$MANIFEST_TSV\"\n")
        f.write("echo \"finished: total=$TOTAL success=$(wc -l < \\\"$SUCCESS_FILE\\\") failed=$(wc -l < \\\"$FAILED_FILE\\\")\"\n")
    os.chmod(script_path, 0o755)


def run_wget_script(script_path: str) -> None:
    subprocess.run(["bash", script_path], check=True)


def save_splits_txt(split_subjects: Dict[str, Set[int]], split_hadms: Dict[str, Set[int]], split_dir: str, suffix: str) -> None:
    for name in ["train", "val", "test"]:
        subj_path = os.path.join(split_dir, f"{name}_subjects_{suffix}.txt")
        hadm_path = os.path.join(split_dir, f"{name}_hadms_{suffix}.txt")
        with open(subj_path, "w", encoding="utf-8") as f:
            for sid in sorted(split_subjects[name]):
                f.write(f"{sid}\n")
        with open(hadm_path, "w", encoding="utf-8") as f:
            for hid in sorted(split_hadms[name]):
                f.write(f"{hid}\n")


def save_outputs(
    subset_name: str,
    suffix: str,
    output_root: str,
    labels: pd.DataFrame,
    master: pd.DataFrame,
    details: Dict[str, Optional[pd.DataFrame]],
    split_subjects: Dict[str, Set[int]],
    split_hadms: Dict[str, Set[int]],
    stats: Dict[str, object],
) -> None:
    """
    输入:
      - subset 全部中间结果
    输出:
      - 统一落盘 labels/splits/timeline/details/stats/log
    关键约束:
      - 所有输出均在 subset 目录下，避免污染原 Step0-4 产物。
    """
    dirs = setup_subset_dirs(output_root, subset_name)

    labels_path = os.path.join(dirs["labels"], f"cohort_labels_{subset_name}_{suffix}.csv.gz")
    labels.to_csv(labels_path, index=False, compression="gzip")

    master_path = os.path.join(dirs["timeline"], f"master_timeline_{subset_name}_{suffix}.csv.gz")
    master.to_csv(master_path, index=False, compression="gzip")

    for mod, df in details.items():
        if df is None:
            continue
        out_path = os.path.join(dirs["details"], f"details_{mod}_grp_{subset_name}_{suffix}.csv.gz")
        df.to_csv(out_path, index=False, compression="gzip")

    save_splits_txt(split_subjects, split_hadms, dirs["splits"], f"{subset_name}_{suffix}")

    stats_path = os.path.join(dirs["stats"], f"stats_{subset_name}_{suffix}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(dirs["log"], f"summary_{subset_name}_{suffix}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"subset={subset_name}\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


# =============================
# 主流程
# =============================
def main():
    parser = argparse.ArgumentParser(description="后处理构建 subset1/subset2（不重跑 Step0-4）")
    parser.add_argument("--mode", type=str, default="ALL", choices=["DEBUG", "ALL", "debug", "all"])
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset2-target-hadm", type=int, default=1000)
    parser.add_argument("--subset2-target-test-hadm", type=int, default=100)
    parser.add_argument("--subset2-test-tolerance", type=int, default=5)
    parser.add_argument("--subset2-one-hadm-per-subject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-origin-data", action="store_true")
    parser.add_argument("--wget-bin", type=str, default="wget")
    parser.add_argument("--physionet-user", type=str, default=os.environ.get("PHYSIONET_USER"))
    parser.add_argument("--physionet-password", type=str, default=os.environ.get("PHYSIONET_PASSWORD"))
    parser.add_argument("--cxr-format", type=str, choices=["dcm", "jpg"], default="dcm")
    parser.add_argument("--cxr-index-path", type=str, default=None, help="本地 CXR 路径索引(csv.gz)，用于 dicom_id -> path")
    parser.add_argument("--cxr-base-url", type=str, default="https://physionet.org/files/mimic-cxr/2.1.0")
    parser.add_argument("--ecg-base-url", type=str, default="https://physionet.org/files/mimic-iv-ecg/1.0")
    parser.add_argument("--echo-base-url", type=str, default="https://physionet.org/files/mimic-iv-echo/0.1")
    args = parser.parse_args()

    mode = args.mode.upper()

    loaded = load_inputs(mode=mode, output_root=args.output_root)
    suffix = loaded["suffix"]
    df_cvd = loaded["df_cvd"]
    df_master = loaded["df_master"]
    details_all = loaded["details"]

    # Step A: subset1
    subset1 = build_subset1(df_cvd=df_cvd, seed=args.seed)
    subset1_hadms = set(map(int, subset1["labels"]["hadm_id"].drop_duplicates().tolist()))

    subset1_master = filter_by_hadm(df_master, subset1_hadms)
    subset1_details = {k: filter_by_hadm(v, subset1_hadms) for k, v in details_all.items()}

    save_outputs(
        subset_name="subset1",
        suffix=suffix,
        output_root=args.output_root,
        labels=subset1["labels"],
        master=subset1_master,
        details=subset1_details,
        split_subjects=subset1["split_subjects"],
        split_hadms=subset1["split_hadms"],
        stats=subset1["stats"],
    )

    print("[subset1] 完成")
    print(subset1["stats"])

    # Step B: subset2（基于 subset1）
    subset2 = build_subset2_from_subset1(
        subset1_labels=subset1["labels"],
        details_subset1=subset1_details,
        seed=args.seed,
        target_hadm=args.subset2_target_hadm,
        target_test_hadm=args.subset2_target_test_hadm,
        test_tolerance=args.subset2_test_tolerance,
        one_hadm_per_subject=args.subset2_one_hadm_per_subject,
    )

    subset2_hadms = set(map(int, subset2["labels"]["hadm_id"].drop_duplicates().tolist()))
    subset2_master = filter_by_hadm(subset1_master, subset2_hadms)
    subset2_details = {k: filter_by_hadm(v, subset2_hadms) for k, v in subset1_details.items()}

    save_outputs(
        subset_name="subset2",
        suffix=suffix,
        output_root=args.output_root,
        labels=subset2["labels"],
        master=subset2_master,
        details=subset2_details,
        split_subjects=subset2["split_subjects"],
        split_hadms=subset2["split_hadms"],
        stats=subset2["stats"],
    )

    print("[subset2] 完成")
    print(subset2["stats"])

    # Step C: subset2 原始模态文件下载清单与脚本（可选执行下载）
    subset2_dirs = setup_subset_dirs(args.output_root, "subset2")
    manifest = prepare_origin_data_download(
        subset2_dirs=subset2_dirs,
        details_subset2=subset2_details,
        cxr_base_url=args.cxr_base_url,
        ecg_base_url=args.ecg_base_url,
        echo_base_url=args.echo_base_url,
        cxr_format=args.cxr_format,
        cxr_index_path=args.cxr_index_path,
    )
    script_path = os.path.join(subset2_dirs["origin_data"], "download_origin_data.sh")
    write_wget_script(
        script_path=script_path,
        manifest=manifest,
        origin_data_dir=subset2_dirs["origin_data"],
        physionet_user=args.physionet_user,
        physionet_password=args.physionet_password,
        wget_bin=args.wget_bin,
    )
    print(f"[subset2] origin data manifest 已生成: {os.path.join(subset2_dirs['origin_data'], 'origin_download_manifest.csv.gz')}")
    print(f"[subset2] wget 脚本已生成: {script_path}")
    if args.download_origin_data:
        print("[subset2] 开始执行 wget 下载 ...")
        run_wget_script(script_path)
        print("[subset2] origin data 下载完成。")


if __name__ == "__main__":
    main()
