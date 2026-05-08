import pandas as pd
import numpy as np
import os
import logging
import argparse
import ast
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

# 启用 tqdm 与 Pandas 的集成
tqdm.pandas() 

# ================= 配置区域 (Configuration) =================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Defaults target the checked-in project layout:
#   <project_root>/mimiciv
#   <project_root>/mmdata_pipeline/cvd_category
# Override with BASE_DATA_DIR / OUTPUT_ROOT when running against another dataset.
BASE_DATA_DIR = os.environ.get('BASE_DATA_DIR', PROJECT_ROOT)
OUTPUT_ROOT = os.environ.get(
    'OUTPUT_ROOT',
    os.path.join(BASE_DATA_DIR, 'temporal_output_norm_icd10')
)

PATHS = {
    'hosp': os.path.join(BASE_DATA_DIR, 'mimiciv/3.1/hosp'),
    'icu': os.path.join(BASE_DATA_DIR, 'mimiciv/3.1/icu'),
    'note_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/note'),
    'cxr_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/cxr'),
    'ecg_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/ecg'),
    'echo_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/echo'),
    'CVD_CATEGORY_PATH': os.environ.get(
        'CVD_CATEGORY_PATH',
        os.path.join(SCRIPT_DIR, 'cvd_category')
    ),

    'step0_output_dir': os.path.join(OUTPUT_ROOT, 'step0_death_admissionlabel/'),
    'step1_output_dir': os.path.join(OUTPUT_ROOT, 'step1_cvd_filter/'), 
    'step2_output_dir': os.path.join(OUTPUT_ROOT, 'step2_multimodal_matching/'),
    'step3_output_dir': os.path.join(OUTPUT_ROOT, 'step3_temporal_timeline/'),
    'step4_output_dir': os.path.join(OUTPUT_ROOT, 'step4_labels_splits/'),
    'labels_dir': os.path.join(OUTPUT_ROOT, 'step4_labels_splits/labels/'),
    'splits_dir': os.path.join(OUTPUT_ROOT, 'step4_labels_splits/splits/'),
}

NOTE_FILES = {'radiology': 'radiology.csv.gz', 'discharge': 'discharge.csv.gz'}
CXR_FILES = {'metadata': 'mimic-cxr-2.0.0-metadata.csv.gz', 'chexpert': 'mimic-cxr-2.0.0-chexpert.csv.gz', 'record_list': 'cxr-record-list.csv.gz'}

# MIMIC-CXR CheXpert labels are study-level labels.
# They will be merged into df_cxr by ['subject_id', 'study_id'].
CXR_CHEXPERT_LABEL_COLS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices',
]
ECG_FILES = {'machine_measurements': 'machine_measurements.csv', 'record_list': 'record_list.csv'}
ECHO_FILES = {'study_list': 'echo-study-list.csv', 'record_list': 'echo-record-list.csv'}

# ================= 辅助函数 =================
def setup_logger(output_dir, log_filename, module_name):
    LOG_DIR = os.path.join(output_dir, 'log')
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    log_filepath = os.path.join(LOG_DIR, log_filename)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_filepath, mode='w')
    ch = logging.StreamHandler()
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def calc_time_offset(event_time, admittime):
    return (pd.to_datetime(event_time) - pd.to_datetime(admittime)).dt.total_seconds() / 3600.0

def parse_single_icd_range(range_str):
    """解析单个 ICD 范围字符串 (e.g., "410–414" 或 "I5A") 为 min, max"""
    if not isinstance(range_str, str) or not range_str:
        return None, None
    range_str = range_str.strip()
    if '–' in range_str: # 使用 en-dash
        parts = range_str.split('–')
    elif '-' in range_str: # 使用 hyphen
        parts = range_str.split('-')
    else:
        return range_str, range_str
    return parts[0].strip(), parts[-1].strip()

def is_icd_in_range(icd_code, icd_version, range_min, range_max):
    """判断单个 ICD 代码是否落在 min/max 范围内"""
    if pd.isna(icd_code) or pd.isna(icd_version) or not range_min:
        return False
    # 预处理：移除 ICD-9/10 的小数点，只比较有效字符
    clean_code = str(icd_code).replace('.', '').upper()
    clean_min = str(range_min).replace('.', '').upper()
    clean_max = str(range_max).replace('.', '').upper()
    
    if len(clean_code) < 3: return False
    code_prefix = clean_code[:3] 
    
    if icd_version == 9:
        if len(clean_min) != 3 or len(clean_max) != 3:
            code_prefix_match = clean_code[:len(clean_min)] 
            return code_prefix_match >= clean_min and code_prefix_match <= clean_max
        else:
            return code_prefix >= clean_min and code_prefix <= clean_max
    elif icd_version == 10:
        code_prefix_icd10 = clean_code[:len(clean_min)] 
        return code_prefix_icd10 >= clean_min and code_prefix_icd10 <= clean_max
    return False

def build_match_map(df):
    """将分类 DataFrame 转换为易于查找的匹配字典列表"""
    match_map = []
    for index, row in df.iterrows():
        internal_code = row['InternalCode']
        
        icd10_code = row['ICD10_Code']
        if pd.notna(icd10_code):
            for part in str(icd10_code).split('/'):
                min_code, max_code = parse_single_icd_range(part.strip())
                if min_code: match_map.append({'code': internal_code, 'version': 10, 'min': min_code, 'max': max_code})
        
        icd9_code = row['ICD9_Code']
        if pd.notna(icd9_code): 
            for part in str(icd9_code).split('/'):
                min_code, max_code = parse_single_icd_range(part.strip())
                if min_code: match_map.append({'code': internal_code, 'version': 9, 'min': min_code, 'max': max_code})
    return match_map

# ================= Step 0: 结局标签生成 =================
def step_0_generate_labels(run_mode, debug_subjects=None):
    out_dir = PATHS['step0_output_dir']
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logger = setup_logger(out_dir, f'step0_{run_mode.lower()}.log', 'Step0')
    logger.info("-> Step 0: 计算死亡与再入院标签")
    
    df_adm = pd.read_csv(os.path.join(PATHS['hosp'], 'admissions.csv.gz'), usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
    if run_mode == 'DEBUG' and debug_subjects is not None: df_adm = df_adm[df_adm['subject_id'].isin(debug_subjects)].copy()
    for col in ['admittime', 'dischtime', 'deathtime']: df_adm[col] = pd.to_datetime(df_adm[col], errors='coerce')

    df_pat = pd.read_csv(os.path.join(PATHS['hosp'], 'patients.csv.gz'), usecols=['subject_id', 'dod'])
    df_pat['dod'] = pd.to_datetime(df_pat['dod'], errors='coerce')
    
    df_icu = pd.read_csv(os.path.join(PATHS['icu'], 'icustays.csv.gz'), usecols=['subject_id', 'hadm_id', 'stay_id', 'intime'])
    df_icu['intime'] = pd.to_datetime(df_icu['intime'], errors='coerce')
    df_icu_first = df_icu.sort_values(['subject_id', 'intime']).groupby('hadm_id').first().reset_index()

    df_label = pd.merge(pd.merge(df_adm, df_pat, on='subject_id', how='left'), df_icu_first[['hadm_id', 'stay_id']].rename(columns={'stay_id': 'curr_stay_id'}), on='hadm_id', how='left')
    df_label.rename(columns={'hospital_expire_flag': 'mortality_in_hospital'}, inplace=True)
    df_label['final_death_date'] = df_label['deathtime'].combine_first(df_label['dod'])
    df_label['diff_death_days_raw'] = (df_label['final_death_date'] - df_label['dischtime']).dt.total_seconds() / (24*3600)
    df_label['mortality_30d'] = ((df_label['diff_death_days_raw'] > 0) & (df_label['diff_death_days_raw'] <= 30)).astype(int)
    
    df_label = df_label.sort_values(by=['subject_id', 'admittime'])
    df_label['next_hadm_id'] = df_label.groupby('subject_id')['hadm_id'].shift(-1)
    df_label['next_admittime'] = df_label.groupby('subject_id')['admittime'].shift(-1)
    df_label = pd.merge(df_label, df_icu_first[['hadm_id', 'stay_id']].rename(columns={'hadm_id': 'next_hadm_id', 'stay_id': 'next_stay_id'}), on='next_hadm_id', how='left')
    df_label['diff_next_adm_days_raw'] = (df_label['next_admittime'] - df_label['dischtime']).dt.total_seconds() / (24*3600)
    df_label['readmission_30d_hosp'] = ((df_label['diff_next_adm_days_raw'] <= 30) & (df_label['diff_next_adm_days_raw'] > 0)).astype(int)
    df_label['readmission_30d_icu'] = ((df_label['readmission_30d_hosp'] == 1) & (df_label['next_stay_id'].notna())).astype(int)

    df_diag = pd.read_csv(os.path.join(PATHS['hosp'], 'diagnoses_icd.csv.gz'), dtype={'icd_code': str, 'icd_version': 'Int64'})
    if run_mode == 'DEBUG' and debug_subjects is not None: df_diag = df_diag[df_diag['subject_id'].isin(debug_subjects)]
    df_dict = pd.read_csv(os.path.join(PATHS['hosp'], 'd_icd_diagnoses.csv.gz'), usecols=['icd_code', 'icd_version', 'long_title'])

    return pd.merge(pd.merge(df_diag, df_label, on=['subject_id', 'hadm_id'], how='left'), df_dict, on=['icd_code', 'icd_version'], how='left')

# ================= Step 1: CVD 队列提取与 List 聚合 =================
# ================= Step 1: 精准过滤与归一化 (增强版) =================

def load_gems_mapping(gems_path):
    """加载官方 GEMs 映射表"""
    if not os.path.exists(gems_path):
        print(f"Warning: GEMs file not found at {gems_path}")
        return {}
    try:
        # 兼容 NBER csv 格式
        df = pd.read_csv(gems_path, sep=None, engine='python', dtype={'icd9cm': str, 'icd10cm': str})
        df['icd9cm'] = df['icd9cm'].str.strip().str.upper()
        df['icd10cm'] = df['icd10cm'].str.strip().str.upper()
        return df.drop_duplicates('icd9cm').set_index('icd9cm')['icd10cm'].to_dict()
    except Exception as e:
        print(f"Error loading GEMs: {e}")
        return {}

def step_1_cvd_labeling(run_mode, df_input, debug_seed=42, debug_cvd_subjects=20):
    out_dir = PATHS['step1_output_dir']
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logger = setup_logger(out_dir, f'step1_{run_mode.lower()}.log', 'Step1')
    logger.info("-> Step 1: 精准过滤 CVD 且执行 ICD-9 到 ICD-10 的归一化映射")
    
    # 1. 加载所有分类规则与映射表
    df_coarse = pd.read_csv(os.path.join(PATHS['CVD_CATEGORY_PATH'], 'CVD_coarse_category.csv'))
    df_fine = pd.read_csv(os.path.join(PATHS['CVD_CATEGORY_PATH'], 'CVD_fine_category.csv'))
    coarse_map, fine_map = build_match_map(df_coarse), build_match_map(df_fine)
    
    gems_path = os.path.join(PATHS['CVD_CATEGORY_PATH'], 'icd9toicd10cmgem.csv')
    gems_dict = load_gems_mapping(gems_path)
    
    # 加载字典用于 norm_title
    df_dict = pd.read_csv(os.path.join(PATHS['hosp'], 'd_icd_diagnoses.csv.gz'))
    v10_titles = df_dict[df_dict['icd_version'] == 10].set_index('icd_code')['long_title'].to_dict()

    # 2. 预处理归一化逻辑 (向量化优化)
    logger.info("提取唯一 ICD 代码进行归一化逻辑预计算...")
    unique_codes = df_input[['icd_code', 'icd_version']].drop_duplicates().copy()
    
    def get_norm_info(row):
        code, ver = str(row['icd_code']).replace('.', '').upper(), row['icd_version']
        c_cat, f_cat = pd.NA, pd.NA
        norm_code, norm_method = pd.NA, 'None'
        
        # 匹配细分亚型
        for item in fine_map:
            if item['version'] == ver and is_icd_in_range(code, ver, item['min'], item['max']): 
                f_cat = item['code']
                break
        
        # 匹配大类
        for item in coarse_map:
            if item['version'] == ver and is_icd_in_range(code, ver, item['min'], item['max']): 
                c_cat = item['code']; break
        
        # 严格的归一化映射逻辑
        if ver == 10:
            norm_code, norm_method = code, 'Original_V10'
        elif ver == 9:
            if code in gems_dict:
                norm_code, norm_method = gems_dict[code], 'GEMs_Official'
            else:
                # 舍弃原有的 fallback 兜底逻辑，无法映射直接判空
                norm_code, norm_method = pd.NA, 'Failed_Mapping'
        
        return pd.Series([c_cat, f_cat, norm_code, norm_method])

    # 执行映射并合并回主表
    mapping_meta = unique_codes.apply(get_norm_info, axis=1)
    mapping_meta.columns = ['CVD_coarse_category', 'CVD_fine_category', 'norm_icd10_code', 'norm_method']
    unique_codes = pd.concat([unique_codes, mapping_meta], axis=1)
    
    # 增加归一化标题
    unique_codes['norm_icd10_title'] = unique_codes['norm_icd10_code'].map(v10_titles)
    
    logger.info("同步映射结果至主数据流...")
    df_input = pd.merge(df_input, unique_codes, on=['icd_code', 'icd_version'], how='left')

    # 3. 筛选 CVD 记录并执行统计与强过滤
    df_cvd_matched = df_input[df_input['CVD_coarse_category'].notna()].copy()
    
    # 新增过滤：如果 norm_icd10_code 是 NaN（即 ICD-9 映射失败），则丢弃这条诊断记录
    initial_len = len(df_cvd_matched)
    df_cvd_matched = df_cvd_matched[df_cvd_matched['norm_icd10_code'].notna()].copy()
    dropped_len = initial_len - len(df_cvd_matched)
    logger.info(f"由于 ICD-9 无法根据规则映射至 ICD-10，已丢弃 {dropped_len} 条诊断记录。")
    
    if df_cvd_matched.empty:
        logger.warning("未找到任何合规的 CVD 匹配记录。")
        return pd.DataFrame()
        
    # 统计映射成功率
    v9_stats = df_cvd_matched[df_cvd_matched['icd_version'] == 9]['norm_method'].value_counts()
    logger.info(f"CVD ICD-9 映射统计: {v9_stats.to_dict()}")
    v9_stats.to_csv(os.path.join(out_dir, 'step1_mapping_stats.csv'))

    if run_mode == 'DEBUG':
        unique_subj = df_cvd_matched['subject_id'].drop_duplicates().to_numpy()
        sample_size = min(debug_cvd_subjects, len(unique_subj))
        rng = np.random.default_rng(debug_seed)
        unique_subj = rng.choice(unique_subj, size=sample_size, replace=False)
        df_cvd_matched = df_cvd_matched[df_cvd_matched['subject_id'].isin(unique_subj)].copy()
        logger.info(f"[DEBUG] seed={debug_seed}，约束至 {len(unique_subj)} 名 CVD 患者。")

    # 4. 执行 List 聚合 (保留所有原始溯源字段 + 新增归一化字段)
    logger.info("执行多维度 List 聚合...")
    
    agg_funcs = {
        # --- 原始诊断详情 ---
        'icd_code': lambda x: list(set(x.dropna())),
        'icd_version': lambda x: list(set(x.dropna())),
        'seq_num': lambda x: list(x.dropna()),
        'long_title': lambda x: list(set(x.dropna())),
        'CVD_coarse_category': lambda x: list(set(x.dropna())),
        'CVD_fine_category': lambda x: list(set(x.dropna())),
        
        # --- 新增：归一化信息 ---
        'norm_icd10_code': lambda x: list(set(x.dropna())),
        'norm_icd10_title': lambda x: list(set(x.dropna())),
        'norm_method': lambda x: list(set(x.dropna())),
        
        # --- 基础时间与 ID ---
        'admittime': 'first',
        'dischtime': 'first',
        'curr_stay_id': 'first',
        
        # --- 死亡结局审计 ---
        'deathtime': 'first',
        'dod': 'first',
        'final_death_date': 'first',
        'diff_death_days_raw': 'first',
        
        # --- 再入院审计 ---
        'next_hadm_id': 'first',
        'next_admittime': 'first',
        'next_stay_id': 'first',
        'diff_next_adm_days_raw': 'first',
        
        # --- 最终标签 ---
        'mortality_in_hospital': 'first',
        'mortality_30d': 'first',
        'readmission_30d_hosp': 'first',
        'readmission_30d_icu': 'first'
    }
    
    df_cvd_cohort = df_cvd_matched.groupby(['subject_id', 'hadm_id']).agg(agg_funcs).reset_index()
    
    # 排除没有任何 CVD 分类的极端空列表情况
    df_cvd_cohort = df_cvd_cohort[df_cvd_cohort['CVD_coarse_category'].apply(len) > 0]
    
    logger.info(f"Step 1 完成。总计提取 {len(df_cvd_cohort)} 次 CVD 住院记录。")
    df_cvd_cohort.to_csv(os.path.join(out_dir, f'step_1_cvd_cohort_{run_mode.lower()}.csv.gz'), index=False, compression='gzip')
    
    return df_cvd_cohort


def _safe_timedelta_hours(later, earlier):
    """Return absolute time difference in hours for pandas Series."""
    return (pd.to_datetime(later) - pd.to_datetime(earlier)).abs().dt.total_seconds() / 3600.0


def resolve_duplicate_exam_matches(df, exam_id_col, event_time_col, modality_name, out_dir, run_mode, logger=None):
    """
    检查并解决同一条检查匹配到多个 hadm_id 的情况。

    当前 CXR/ECG/Echo 没有可靠 hadm_id，代码先按 subject_id 连接候选住院，
    再用 [admittime - 24h, dischtime] 时间窗过滤。若同一检查落入多个候选住院，
    这里按下面规则保留唯一匹配：
      1) 如果检查发生在某次住院内，比较它距离该次住院 dischtime 的距离；
      2) 如果检查发生在入院前 24h 窗口，比较它距离该次住院 admittime 的距离；
      3) 保留距离更近的一条；若完全相同，优先保留真正住院期间的记录。

    返回去重后的 df。重复匹配明细会保存到 step2 输出目录，便于人工核查。
    """
    if df is None or df.empty:
        msg = f"[{modality_name}] 空表，跳过重复匹配统计。"
        print(msg)
        if logger: logger.info(msg)
        return df

    required_cols = {exam_id_col, event_time_col, 'subject_id', 'hadm_id', 'admittime', 'dischtime'}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"[{modality_name}] 缺少必要列 {missing}，跳过重复匹配处理。"
        print(msg)
        if logger: logger.warning(msg)
        return df

    suffix = run_mode.lower()
    df = df.copy()
    df[event_time_col] = pd.to_datetime(df[event_time_col], errors='coerce')
    df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
    df['dischtime'] = pd.to_datetime(df['dischtime'], errors='coerce')

    # 统计一条检查对应多少个不同 hadm_id。
    hadm_per_exam = df.groupby(exam_id_col)['hadm_id'].nunique().sort_values(ascending=False)
    dist_before = hadm_per_exam.value_counts().sort_index()
    dup_exam_ids = hadm_per_exam[hadm_per_exam > 1].index

    msg = (
        f"\n[{modality_name}] 重复匹配统计（处理前）\n"
        f"检查唯一 ID: {exam_id_col}\n"
        f"总检查数: {len(hadm_per_exam)}\n"
        f"每条检查匹配 hadm_id 数量分布:\n{dist_before.to_string()}\n"
        f"重复匹配检查数: {len(dup_exam_ids)}"
    )
    print(msg)
    if logger: logger.info(msg)

    if len(dup_exam_ids) == 0:
        return df

    dup_detail = (
        df[df[exam_id_col].isin(dup_exam_ids)]
        .sort_values([exam_id_col, 'subject_id', 'admittime', 'dischtime'])
        .copy()
    )
    dup_detail.to_csv(
        os.path.join(out_dir, f'duplicate_{modality_name.lower()}_matches_before_{suffix}.csv.gz'),
        index=False,
        compression='gzip'
    )

    # 计算冲突消解分数。
    # in_stay：检查真正发生在住院期间；pre_admit：检查发生在入院前 24h 窗口。
    df['_in_stay'] = (df[event_time_col] >= df['admittime']) & (df[event_time_col] <= df['dischtime'])
    df['_pre_admit'] = (df[event_time_col] < df['admittime']) & (df[event_time_col] >= df['admittime'] - pd.Timedelta(hours=24))
    df['_dist_to_discharge_h'] = _safe_timedelta_hours(df['dischtime'], df[event_time_col])
    df['_dist_to_admit_h'] = _safe_timedelta_hours(df[event_time_col], df['admittime'])
    df['_match_distance_h'] = np.where(df['_in_stay'], df['_dist_to_discharge_h'], df['_dist_to_admit_h'])
    # 若距离完全相同，优先保留住院期间记录，再按时间边界距离和 hadm_id 稳定排序。
    df['_match_priority'] = np.where(df['_in_stay'], 0, np.where(df['_pre_admit'], 1, 2))

    # 每个 exam_id 只保留最佳 hadm_id 匹配。
    sort_cols = [exam_id_col, '_match_distance_h', '_match_priority', 'hadm_id']
    df_resolved = (
        df.sort_values(sort_cols)
        .drop_duplicates(subset=[exam_id_col], keep='first')
        .drop(columns=['_in_stay', '_pre_admit', '_dist_to_discharge_h', '_dist_to_admit_h', '_match_distance_h', '_match_priority'])
        .reset_index(drop=True)
    )

    hadm_per_exam_after = df_resolved.groupby(exam_id_col)['hadm_id'].nunique().sort_values(ascending=False)
    dist_after = hadm_per_exam_after.value_counts().sort_index()
    msg = (
        f"\n[{modality_name}] 重复匹配处理完成\n"
        f"处理前行数: {len(df)}\n"
        f"处理后行数: {len(df_resolved)}\n"
        f"每条检查匹配 hadm_id 数量分布（处理后）:\n{dist_after.to_string()}"
    )
    print(msg)
    if logger: logger.info(msg)

    return df_resolved


# ================= Step 2: 恢复完整的多模态映射 =================
def step_2_multimodal_matching(run_mode, df_cvd):
    suffix = run_mode.lower()
    out_dir = PATHS['step2_output_dir']
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logger = setup_logger(out_dir, f'step2_{suffix}.log', 'Step2')
    logger.info("-> Step 2: 多模态匹配、Notes 时间窗修正、CXR CheXpert 融合与重复匹配统计")

    valid_subjects = set(df_cvd['subject_id'].unique())
    df_cohort = df_cvd[['subject_id', 'hadm_id', 'admittime', 'dischtime']].drop_duplicates().copy()
    df_cohort['admittime'] = pd.to_datetime(df_cohort['admittime'], errors='coerce')
    df_cohort['dischtime'] = pd.to_datetime(df_cohort['dischtime'], errors='coerce')
    
    # 2A: Notes 文本匹配
    # 修正点：Notes 也统一使用 [admittime - 24h, dischtime] 时间窗。
    # 即使 note 自带 hadm_id，也需要确认 charttime 在时间窗内，且 hadm_id 对得上。
    matched_notes = []
    subject_stay_map = {}
    for _, row in df_cohort.iterrows():
        sid, hid, start, end = row['subject_id'], row['hadm_id'], row['admittime'], row['dischtime']
        if sid not in subject_stay_map:
            subject_stay_map[sid] = []
        subject_stay_map[sid].append({'hadm_id': hid, 'start': start, 'end': end})

    for ntype, fname in NOTE_FILES.items():
        npath = os.path.join(PATHS['note_dir'], fname)
        if os.path.exists(npath):
            logger.info(f"匹配 Notes: {ntype} -> {fname}")
            for chunk in pd.read_csv(npath, usecols=['note_id', 'subject_id', 'hadm_id', 'note_type', 'charttime'], chunksize=100000):
                chunk = chunk[chunk['subject_id'].isin(valid_subjects)].copy()
                if chunk.empty:
                    continue
                chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
                
                def find_hadm(r):
                    if pd.isna(r['charttime']):
                        return None

                    stays = subject_stay_map.get(r['subject_id'], [])
                    for s in stays:
                        in_window = (
                            s['start'] - pd.Timedelta(hours=24)
                            <= r['charttime']
                            <= s['end']
                        )
                        same_hadm = (
                            pd.isna(r['hadm_id'])
                            or int(r['hadm_id']) == int(s['hadm_id'])
                        )
                        if in_window and same_hadm:
                            return s['hadm_id']
                    return None
                
                chunk['matched_hadm'] = chunk.apply(find_hadm, axis=1)
                valid_notes = chunk[chunk['matched_hadm'].notna()].copy()
                valid_notes['hadm_id'] = valid_notes['matched_hadm']
                if not valid_notes.empty:
                    matched_notes.append(valid_notes.drop(columns=['matched_hadm']))
        else:
            logger.warning(f"Notes 文件不存在，跳过: {npath}")
    
    df_notes = pd.concat(matched_notes, ignore_index=True) if matched_notes else pd.DataFrame()
    logger.info(f"Notes 匹配完成，共 {len(df_notes)} 条 note 记录。")

    # 2B: CXR 影像路径、元数据与 CheXpert label
    cxr_meta = pd.read_csv(
        os.path.join(PATHS['cxr_dir'], CXR_FILES['metadata']),
        usecols=['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'ViewPosition']
    )
    cxr_meta = cxr_meta[cxr_meta['subject_id'].isin(valid_subjects)].copy()
    cxr_meta['StudyTime'] = cxr_meta['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6)
    cxr_meta['cxr_time'] = pd.to_datetime(
        cxr_meta['StudyDate'].astype(str) + ' ' + cxr_meta['StudyTime'],
        format='%Y%m%d %H%M%S',
        errors='coerce'
    )
    df_cxr = pd.merge(cxr_meta, df_cohort, on='subject_id', how='inner')
    df_cxr = df_cxr[
        (df_cxr['cxr_time'] >= df_cxr['admittime'] - pd.Timedelta(hours=24)) &
        (df_cxr['cxr_time'] <= df_cxr['dischtime'])
    ].copy()
    cxr_paths = pd.read_csv(
        os.path.join(PATHS['cxr_dir'], CXR_FILES['record_list']),
        usecols=['dicom_id', 'path']
    ).rename(columns={'path': 'cxr_path'})
    df_cxr = pd.merge(df_cxr, cxr_paths, on='dicom_id', how='left')

    # 新增：真正融合 CheXpert label。CheXpert 是 study-level 标签，用 subject_id + study_id 合并。
    chexpert_path = os.path.join(PATHS['cxr_dir'], CXR_FILES['chexpert'])
    if os.path.exists(chexpert_path):
        cxr_chexpert = pd.read_csv(chexpert_path)
        expected_keys = ['subject_id', 'study_id']
        if all(col in cxr_chexpert.columns for col in expected_keys):
            keep_cols = expected_keys + [col for col in CXR_CHEXPERT_LABEL_COLS if col in cxr_chexpert.columns]
            cxr_chexpert = cxr_chexpert[keep_cols].drop_duplicates(subset=expected_keys)
            df_cxr = pd.merge(df_cxr, cxr_chexpert, on=expected_keys, how='left')
            logger.info(f"CXR CheXpert label 已合并，标签列: {[c for c in CXR_CHEXPERT_LABEL_COLS if c in df_cxr.columns]}")
        else:
            logger.warning(f"CheXpert 文件缺少 subject_id/study_id，跳过合并: {chexpert_path}")
    else:
        logger.warning(f"CheXpert 文件不存在，跳过合并: {chexpert_path}")

    # 新增：统计并处理 CXR 重复匹配。
    df_cxr = resolve_duplicate_exam_matches(
        df_cxr,
        exam_id_col='dicom_id',
        event_time_col='cxr_time',
        modality_name='CXR',
        out_dir=out_dir,
        run_mode=run_mode,
        logger=logger
    )

    # 2C: ECG 波形测量与路径
    df_meas = pd.read_csv(
        os.path.join(PATHS['ecg_dir'], ECG_FILES['machine_measurements']),
        usecols=['subject_id', 'study_id', 'ecg_time', 'rr_interval', 'p_onset', 'qrs_onset']
    )
    df_meas = df_meas[df_meas['subject_id'].isin(valid_subjects)].copy()
    df_meas['ecg_time'] = pd.to_datetime(df_meas['ecg_time'], errors='coerce')
    df_ecg = pd.merge(df_meas, df_cohort, on='subject_id', how='inner')
    df_ecg = df_ecg[
        (df_ecg['ecg_time'] >= df_ecg['admittime'] - pd.Timedelta(hours=24)) &
        (df_ecg['ecg_time'] <= df_ecg['dischtime'])
    ].copy()
    ecg_paths = pd.read_csv(
        os.path.join(PATHS['ecg_dir'], ECG_FILES['record_list']),
        usecols=['study_id', 'path']
    ).rename(columns={'path': 'ecg_path'})
    df_ecg = pd.merge(df_ecg, ecg_paths, on='study_id', how='left')

    # 新增：统计并处理 ECG 重复匹配。
    df_ecg = resolve_duplicate_exam_matches(
        df_ecg,
        exam_id_col='study_id',
        event_time_col='ecg_time',
        modality_name='ECG',
        out_dir=out_dir,
        run_mode=run_mode,
        logger=logger
    )

    # 2D: Echo 字典与路径
    df_study = pd.read_csv(
        os.path.join(PATHS['echo_dir'], ECHO_FILES['study_list']),
        usecols=['subject_id', 'study_id', 'study_datetime']
    )
    df_study = df_study[df_study['subject_id'].isin(valid_subjects)].copy()
    df_study['echo_time'] = pd.to_datetime(df_study['study_datetime'], errors='coerce')
    df_echo = pd.merge(df_study, df_cohort, on='subject_id', how='inner')
    df_echo = df_echo[
        (df_echo['echo_time'] >= df_echo['admittime'] - pd.Timedelta(hours=24)) &
        (df_echo['echo_time'] <= df_echo['dischtime'])
    ].copy()
    echo_paths = pd.read_csv(
        os.path.join(PATHS['echo_dir'], ECHO_FILES['record_list']),
        usecols=['study_id', 'dicom_filepath']
    ).rename(columns={'dicom_filepath': 'echo_path'})
    df_echo = pd.merge(df_echo, echo_paths, on='study_id', how='left')

    # 新增：统计并处理 Echo 重复匹配。
    df_echo = resolve_duplicate_exam_matches(
        df_echo,
        exam_id_col='study_id',
        event_time_col='echo_time',
        modality_name='Echo',
        out_dir=out_dir,
        run_mode=run_mode,
        logger=logger
    )
    
    return df_cohort, df_notes, df_cxr, df_ecg, df_echo

# ================= step 3: 全模态并发事件分组聚合 =================
def step_3_build_grouped_timeline(run_mode, df_cohort, df_notes, df_cxr, df_ecg, df_echo):
    suffix = run_mode.lower()
    out_dir = PATHS['step3_output_dir']
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logger = setup_logger(out_dir, f'step3_{suffix}.log', 'step3')
    logger.info("-> step 3: 执行严谨的同时间点并发事件打包 (Grouping)")
    
    valid_hadms = set(df_cohort['hadm_id'])
    master_events = []

    # 1. 挂载基准点
    for ev_type, t_col, cat in [('Admission', 'admittime', 'Admin'), ('Discharge', 'dischtime', 'Admin')]:
        df_ad = df_cohort.copy()
        df_ad['event_time'], df_ad['time_offset_hours'] = df_ad[t_col], calc_time_offset(df_ad[t_col], df_ad['admittime'])
        df_ad['event_type'], df_ad['event_category'], df_ad['reference_id'] = ev_type, cat, df_ad['hadm_id'].astype(str)
        master_events.append(df_ad[['subject_id', 'hadm_id', 'event_time', 'time_offset_hours', 'event_type', 'event_category', 'reference_id']])

    # 2. 挂载文本 (Notes Grouping)
    if not df_notes.empty:
        df_notes = pd.merge(df_notes, df_cohort[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'])
        df_notes['time_offset_hours'] = calc_time_offset(df_notes['charttime'], df_notes['admittime'])
        n_grp = df_notes.groupby(['subject_id', 'hadm_id', 'charttime', 'time_offset_hours'], as_index=False).agg(
            note_id_list=('note_id', list), note_type_list=('note_type', list)
        )
        n_grp['reference_id'] = 'note_grp_' + n_grp.index.astype(str)
        tmp = n_grp[['subject_id', 'hadm_id', 'charttime', 'time_offset_hours', 'reference_id']].rename(columns={'charttime': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'Clinical_Note', 'Text'
        master_events.append(tmp)
        n_grp.to_csv(os.path.join(out_dir, f'details_notes_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    # 3. 挂载影像 (CXR/ECG/Echo Grouping)
    if not df_cxr.empty:
        df_cxr['time_offset_hours'] = calc_time_offset(df_cxr['cxr_time'], df_cxr['admittime'])
        cxr_agg = {
            'study_id_list': ('study_id', lambda x: list(dict.fromkeys(x))),
            'dicom_id_list': ('dicom_id', list),
            'cxr_path_list': ('cxr_path', list),
            'view_pos_list': ('ViewPosition', list),
        }
        # CheXpert 是 study-level 标签；同一 study 下多张图共享同一组标签，因此分组时取 first。
        for col in CXR_CHEXPERT_LABEL_COLS:
            if col in df_cxr.columns:
                cxr_agg[col] = (col, 'first')
        cxr_grp = df_cxr.groupby(['subject_id', 'hadm_id', 'cxr_time', 'time_offset_hours'], as_index=False).agg(**cxr_agg)
        cxr_grp['reference_id'] = 'cxr_grp_' + cxr_grp.index.astype(str)
        tmp = cxr_grp[['subject_id', 'hadm_id', 'cxr_time', 'time_offset_hours', 'reference_id']].rename(columns={'cxr_time': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'CXR_Imaging', 'Imaging'
        master_events.append(tmp)
        cxr_grp.to_csv(os.path.join(out_dir, f'details_cxr_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    if not df_ecg.empty:
        df_ecg['time_offset_hours'] = calc_time_offset(df_ecg['ecg_time'], df_ecg['admittime'])
        ecg_grp = df_ecg.groupby(['subject_id', 'hadm_id', 'ecg_time', 'time_offset_hours'], as_index=False).agg(
            study_id_list=('study_id', list), ecg_path_list=('ecg_path', list), rr_list=('rr_interval', list)
        )
        ecg_grp['reference_id'] = 'ecg_grp_' + ecg_grp.index.astype(str)
        tmp = ecg_grp[['subject_id', 'hadm_id', 'ecg_time', 'time_offset_hours', 'reference_id']].rename(columns={'ecg_time': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'ECG_Recording', 'Waveform'
        master_events.append(tmp)
        ecg_grp.to_csv(os.path.join(out_dir, f'details_ecg_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    if not df_echo.empty:
        df_echo['time_offset_hours'] = calc_time_offset(df_echo['echo_time'], df_echo['admittime'])
        echo_grp = df_echo.groupby(['subject_id', 'hadm_id', 'echo_time', 'time_offset_hours'], as_index=False).agg(
            study_id_list=('study_id', list), echo_path_list=('echo_path', list)
        )
        echo_grp['reference_id'] = 'echo_grp_' + echo_grp.index.astype(str)
        tmp = echo_grp[['subject_id', 'hadm_id', 'echo_time', 'time_offset_hours', 'reference_id']].rename(columns={'echo_time': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'Echo_Imaging', 'Imaging'
        master_events.append(tmp)
        echo_grp.to_csv(os.path.join(out_dir, f'details_echo_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    # 4. EHR 数据并行加载与打包 (Labs, Rx, Proc)
    lab_chunks = []
    for chunk in pd.read_csv(os.path.join(PATHS['hosp'], 'labevents.csv.gz'), usecols=['subject_id', 'hadm_id', 'labevent_id', 'charttime', 'itemid', 'valuenum'], chunksize=500000):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)]
        if not chunk.empty: lab_chunks.append(pd.merge(chunk, df_cohort[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id']))
    if lab_chunks:
        df_lab = pd.concat(lab_chunks)
        df_lab['time_offset_hours'] = calc_time_offset(df_lab['charttime'], df_lab['admittime'])
        lab_grp = df_lab.groupby(['subject_id', 'hadm_id', 'charttime', 'time_offset_hours'], as_index=False).agg(
            itemid_list=('itemid', list), val_list=('valuenum', list)
        )
        lab_grp['reference_id'] = 'lab_grp_' + lab_grp.index.astype(str)
        tmp = lab_grp[['subject_id', 'hadm_id', 'charttime', 'time_offset_hours', 'reference_id']].rename(columns={'charttime': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'Lab_Test', 'Lab'
        master_events.append(tmp)
        lab_grp.to_csv(os.path.join(out_dir, f'details_lab_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    df_rx = pd.read_csv(os.path.join(PATHS['hosp'], 'prescriptions.csv.gz'), usecols=['subject_id', 'hadm_id', 'starttime', 'drug', 'dose_val_rx'])
    df_rx = df_rx[df_rx['hadm_id'].isin(valid_hadms)]
    if not df_rx.empty:
        df_rx = pd.merge(df_rx, df_cohort[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'])
        df_rx['time_offset_hours'] = calc_time_offset(df_rx['starttime'], df_rx['admittime'])
        rx_grp = df_rx.groupby(['subject_id', 'hadm_id', 'starttime', 'time_offset_hours'], as_index=False).agg(
            drug_list=('drug', list), dose_list=('dose_val_rx', list)
        )
        rx_grp['reference_id'] = 'rx_grp_' + rx_grp.index.astype(str)
        tmp = rx_grp[['subject_id', 'hadm_id', 'starttime', 'time_offset_hours', 'reference_id']].rename(columns={'starttime': 'event_time'})
        tmp['event_type'], tmp['event_category'] = 'Prescribe_Med', 'Medication'
        master_events.append(tmp)
        rx_grp.to_csv(os.path.join(out_dir, f'details_rx_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    if os.path.exists(os.path.join(PATHS['icu'], 'procedureevents.csv.gz')):
        df_proc = pd.read_csv(os.path.join(PATHS['icu'], 'procedureevents.csv.gz'), usecols=['subject_id', 'hadm_id', 'starttime', 'itemid', 'value'])
        df_proc = df_proc[df_proc['hadm_id'].isin(valid_hadms)]
        if not df_proc.empty:
            df_proc = pd.merge(df_proc, df_cohort[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'])
            df_proc['time_offset_hours'] = calc_time_offset(df_proc['starttime'], df_proc['admittime'])
            proc_grp = df_proc.groupby(['subject_id', 'hadm_id', 'starttime', 'time_offset_hours'], as_index=False).agg(
                itemid_list=('itemid', list), val_list=('value', list)
            )
            proc_grp['reference_id'] = 'proc_grp_' + proc_grp.index.astype(str)
            tmp = proc_grp[['subject_id', 'hadm_id', 'starttime', 'time_offset_hours', 'reference_id']].rename(columns={'starttime': 'event_time'})
            tmp['event_type'], tmp['event_category'] = 'ICU_Procedure', 'Procedure'
            master_events.append(tmp)
            proc_grp.to_csv(os.path.join(out_dir, f'details_proc_grp_{suffix}.csv.gz'), index=False, compression='gzip')

    df_master = pd.concat(master_events, ignore_index=True)
    df_master = df_master.sort_values(by=['subject_id', 'hadm_id', 'time_offset_hours']).reset_index(drop=True)
    df_master['temporal_step'] = df_master.groupby(['subject_id', 'hadm_id']).cumcount() + 1
    cols = ['subject_id', 'hadm_id', 'temporal_step', 'event_category', 'event_type', 'time_offset_hours', 'event_time', 'reference_id']
    df_master[cols].to_csv(os.path.join(out_dir, f'master_timeline_{suffix}.csv.gz'), index=False, compression='gzip')
    logger.info("✅ 组装完成：主时间轴与模态细节表映射成功。")

# ================= step 4: 分层切分与防泄露机制 =================
def step_4_labels_and_splits(run_mode):
    suffix = run_mode.lower()
    for d in [PATHS['step4_output_dir'], PATHS['labels_dir'], PATHS['splits_dir']]:
        if not os.path.exists(d): os.makedirs(d)
    logger = setup_logger(PATHS['step4_output_dir'], f'step4_{suffix}.log', 'step4')
    logger.info("-> step 4: 提取净标签，执行患者级别(Subject-Level)分层切分防泄露")

    master_path = os.path.join(PATHS['step3_output_dir'], f'master_timeline_{suffix}.csv.gz')
    if not os.path.exists(master_path): return
    valid_hadms = set(pd.read_csv(master_path, usecols=['hadm_id'])['hadm_id'])
    
    # 抽取纯净标签
    df_cvd = pd.read_csv(os.path.join(PATHS['step1_output_dir'], f'step_1_cvd_cohort_{suffix}.csv.gz'))
    # df_cvd 已经是在 Step 1 做过 hadm_id 级别聚合的了
    df_cohort_labels = df_cvd[df_cvd['hadm_id'].isin(valid_hadms)].copy()
    
    #label_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'CVD_coarse_category', 'CVD_fine_category', 'icd_code', 'long_title', 'mortality_in_hospital', 'mortality_30d', 'readmission_30d_hosp', 'readmission_30d_icu']
    #df_cohort_labels[label_cols].to_csv(os.path.join(PATHS['labels_dir'], f'cohort_labels_{suffix}.csv'), index=False)
# 【修改点】：在 label_cols 列表中新增 'norm_icd10_code'
    label_cols = [
        'subject_id', 'hadm_id', 'admittime', 'dischtime', 
        'CVD_coarse_category', 'CVD_fine_category', 
        'icd_code', 'norm_icd10_code', 'long_title', 
        'mortality_in_hospital', 'mortality_30d', 
        'readmission_30d_hosp', 'readmission_30d_icu'
    ]
    df_cohort_labels[label_cols].to_csv(os.path.join(PATHS['labels_dir'], f'cohort_labels_{suffix}.csv'), index=False)

    # 辅助函数：由于 CVD_coarse_category 现在是保存为类似 "['IHD', 'HF']" 的字符串表示的 List，
    # 为了防止组合过多导致类别破碎，我们解析出其中的第一个类别作为分层依据
    def get_primary_category(val):
        try:
            l = ast.literal_eval(val)
            return str(l[0]) if len(l) > 0 else 'None'
        except:
            return str(val)

    # 隔离切分 (Train: 70%, Val: 10%, Test: 20%)
    df_subj = df_cohort_labels.groupby('subject_id').agg({'mortality_30d': 'max', 'CVD_coarse_category': 'first'}).reset_index()
    df_subj['primary_cvd'] = df_subj['CVD_coarse_category'].apply(get_primary_category)
    df_subj['strat_key'] = df_subj['primary_cvd'] + "_" + df_subj['mortality_30d'].astype(str)
    
    subjects = df_subj['subject_id'].to_numpy()
    strat_labels = df_subj['strat_key'].to_numpy(dtype=object)
    try:
        t_v_subj, test_subj, t_v_strat, _ = train_test_split(subjects, strat_labels, test_size=0.20, random_state=42, stratify=strat_labels)
        train_subj, val_subj = train_test_split(t_v_subj, test_size=0.125, random_state=42, stratify=t_v_strat) 
    except ValueError:
        logger.warning("类别失衡触发兜底：降级为随机物理隔离安全抽样。")
        t_v_subj, test_subj = train_test_split(subjects, test_size=0.20, random_state=42)
        train_subj, val_subj = train_test_split(t_v_subj, test_size=0.125, random_state=42)

    # 写入白名单
    for name, arr in [('train', train_subj), ('val', val_subj), ('test', test_subj)]:
        with open(os.path.join(PATHS['splits_dir'], f'{name}_subjects_{suffix}.txt'), 'w') as f:
            for s in sorted(arr): f.write(f"{s}\n")
    logger.info("✅ 隔离切分完成，科研防泄露机制部署成功。")

# ================= 调度器 =================
def get_debug_subjects(debug_seed=42, debug_subjects=1000):
    cxr = set(pd.read_csv(os.path.join(PATHS['cxr_dir'], CXR_FILES['metadata']), usecols=['subject_id'])['subject_id'])
    ecg = set(pd.read_csv(os.path.join(PATHS['ecg_dir'], ECG_FILES['machine_measurements']), usecols=['subject_id'])['subject_id'])
    echo = set(pd.read_csv(os.path.join(PATHS['echo_dir'], ECHO_FILES['study_list']), usecols=['subject_id'])['subject_id'])
    note = set(pd.read_csv(os.path.join(PATHS['note_dir'], NOTE_FILES['discharge']), usecols=['subject_id'])['subject_id'])
    subjects = np.array(sorted(cxr.intersection(ecg).intersection(echo).intersection(note)))
    sample_size = min(debug_subjects, len(subjects))
    rng = np.random.default_rng(debug_seed)
    return rng.choice(subjects, size=sample_size, replace=False).tolist()

def main_pipeline(run_mode, debug_seed=42, debug_subjects=1000, debug_cvd_subjects=20):
    print(f"========== 启动终极全模态科研级流水线 [{run_mode}] ==========")
    debug_subs = get_debug_subjects(debug_seed, debug_subjects) if run_mode.upper() == 'DEBUG' else None

    df_labels = step_0_generate_labels(run_mode, debug_subs)
    df_cvd = step_1_cvd_labeling(run_mode, df_labels, debug_seed=debug_seed, debug_cvd_subjects=debug_cvd_subjects)
    
    if not df_cvd.empty:
        df_cohort, df_notes, df_cxr, df_ecg, df_echo = step_2_multimodal_matching(run_mode, df_cvd)
        step_3_build_grouped_timeline(run_mode, df_cohort, df_notes, df_cxr, df_ecg, df_echo)
        step_4_labels_and_splits(run_mode)
    else:
        print("无合规 CVD 数据，退出。")
    print("====================== Pipeline Finished ======================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='debug', choices=['debug', 'all', 'DEBUG', 'ALL'])
    parser.add_argument('--debug-seed', type=int, default=42)
    parser.add_argument('--debug-subjects', type=int, default=1000)
    parser.add_argument('--debug-cvd-subjects', type=int, default=20)
    args = parser.parse_args()
    main_pipeline(
        args.mode.upper(),
        debug_seed=args.debug_seed,
        debug_subjects=args.debug_subjects,
        debug_cvd_subjects=args.debug_cvd_subjects
    )
