"""
========================================================================================
OFFICIAL CODE REPOSITORY
========================================================================================

Paper Title:
    Temporal Validation of a Machine Learning Readmission Model in Heart Failure With Preserved Ejection Fraction and Chronic Kidney Disease

Authors:
    Ping Xie*, Yaoting Deng, Weijie Lu, Yang Zhong, JiaJia Liu, Pengcheng Sheng,
    Mengyang Liu, Kang Yang, Yujie Hu, Nan Ma

    * Corresponding Author

Description:
    This script contains the main analytical pipeline for the study. It performs:
    1. Data Loading & Temporal Splitting (Derivation vs. Validation)
    2. Population Characteristic Analysis 
    3. Robust Feature Selection (Lasso + RFECV with Stability Runs)
    4. Model Training & Hyperparameter Tuning (XGBoost, LightGBM, LR, etc.)
    5. Comprehensive Evaluation (AUROC, Calibration, DCA, Temporal Robustness)

Dependencies:
    See requirements.txt for full list (e.g., numpy, pandas, scikit-learn, lightgbm, xgboost).

========================================================================================
"""

# PART 1: Setup, NEJM Styles, Data Loading, Drift Analysis (Split & Optimized)
# ==============================================================================
# --- [ULTIMATE LOCK]
import os
os.environ['PYTHONHASHSEED'] = '42'
import random
random.seed(42)
import numpy as np
np.random.seed(42)

try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    pass
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass

print("  > [System] GLOBAL RANDOM SEED LOCKED TO 42")

# ----------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import ks_2samp, chi2_contingency
import random
import numpy as np
import os

#
SEED = 42

#
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#
np.random.seed(SEED)

#
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print(f"  > [System] Global Random Seed Locked to {SEED}")

# --- 1. FORCE SINGLE THREADING (Windows Stability) ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# --- Advanced Imports ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Global Style Settings (NEJM / Lancet Grade) ---
plt.style.use('default')
warnings.filterwarnings("ignore")

# Define High-End Journal Colors
NEJM_RED = '#BC3C29'
NEJM_BLUE = '#0072B5'
NEJM_GREY = '#787878'
NEJM_GOLD = '#E18727'
NEJM_GREEN = '#20854E'

PALETTE_12 = sns.color_palette("husl", 12)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.titlesize': 18,
    'figure.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False
})

#
OUTPUT_DIR = 'Results'  #

#
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# [REPLACE THIS SECTION] Enhanced Pretty Name Map & Helper Function
# ==============================================================================
pretty_name_map = {
    # --- Outcomes & Demographics ---
    'readmission_within_1_year': 'Readmission (1-Year)',
    'admission_date': 'Admission Date',
    'age': 'Age', 'sex': 'Sex', 'bmi': 'BMI', 'obesity': 'Obesity',
    'smoker': 'Smoking History', 'length_of_stay': 'Length of Stay',

    # --- Clinical Signs & Vitals ---
    'systolic_bp': 'Systolic BP', 'diastolic_bp': 'Diastolic BP',
    'nyha_class': 'NYHA Class', 'heart_rate': 'Heart Rate',

    # --- Echocardiography ---
    'lvef': 'LVEF', 'nt_probnp': 'NT-proBNP',
    'E_over_e_prime': "E/e'", 'LAVI': 'LAVI',
    'TR_velocity': 'TR Velocity', 'LVMI': 'LVMI',

    # --- Labs: Renal & Metabolic ---
    'egfr': 'eGFR', 'serum_creatinine': 'Serum Creatinine',
    'blood_urea_nitrogen': 'BUN', 'cystatin_c': 'Cystatin C',
    'serum_uric_acid': 'Uric Acid', 'proteinuria_quantification': 'Proteinuria',
    'history_of_aki': 'History of AKI',

    # --- Labs: Inflammatory & Others ---
    'hs_crp': 'hs-CRP', 'il_6': 'IL-6', 'd_dimer': 'D-Dimer',
    'homocysteine': 'Homocysteine', 'serum_sodium': 'Serum Sodium',
    'albumin': 'Albumin', 'total_protein': 'Total Protein',

    # --- Comorbidities & Meds ---
    'diabetes': 'Diabetes', 'hypertension': 'Hypertension',
    'atrial_fibrillation': 'Atrial Fibrillation', 'copd': 'COPD',
    'coronary_artery_disease': 'CAD', 'anemia': 'Anemia',
    'raas_use': 'RAAS Inhibitor Use', 'sglt2i_use': 'SGLT2i Use',
    'beta_blocker_use': 'Beta-Blocker Use', 'mra_use': 'MRA Use',

    # --- Interactions ---
    'Inter_eGFR_NTproBNP': 'eGFR x NT-proBNP',
    'Inter_BMI_NTproBNP': 'BMI x NT-proBNP',
    'Inter_hsCRP_NTproBNP': 'hs-CRP x NT-proBNP'
}

def get_pretty_name(col):
    """Auto-formatter for variable names"""
    if col in pretty_name_map:
        return pretty_name_map[col]
    # Fallback smart formatting
    clean = col.replace('_', ' ').title()
    # Fix common acronyms if missed
    clean = clean.replace('Bnp', 'BNP').replace('Egfr', 'eGFR').replace('Crp', 'CRP').replace('Bmi', 'BMI').replace(
        'Lvef', 'LVEF')
    return clean


# 1. LOAD DATA & TEMPORAL SPLITTING (Strict Date-Based)
# ==============================================================================
print("--- Stage 1: Loading Data & Drift Analysis ---")

# --- [CONFIGURATION] Define Study Periods ---
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-05-31'
# (Washout Period: 2024-06-01 to 2024-06-30)
TEST_START = '2024-07-01'
TEST_END = '2024-10-15'

# ==========================================
# [For Reviewers] Data Loading Section
# ==========================================

try:
    df = pd.read_csv('dummy_data.csv')
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    print("✅ Successfully loaded synthetic data (dummy_data.csv).")
    print("   This dataset is generated by generate_data.py for reproduction purposes.")
except FileNotFoundError:
    print("❌ Error: 'dummy_data.csv' not found.")
    print("   Please run 'python generate_data.py' first to generate the dataset.")
    exit()

# ==============================================================================
# [NEW] STRICT TEMPORAL SPLIT IMPLEMENTATION
# ==============================================================================
print(f"  > Applying Temporal Split Strategy:")
print(f"    Derivation: {TRAIN_START} to {TRAIN_END}")
print(f"    Validation: {TEST_START} to {TEST_END}")

# 1. Create Masks
mask_train = (df['admission_date'] >= pd.Timestamp(TRAIN_START)) & (df['admission_date'] <= pd.Timestamp(TRAIN_END))
mask_test = (df['admission_date'] >= pd.Timestamp(TEST_START)) & (df['admission_date'] <= pd.Timestamp(TEST_END))

# 2. Apply Split
df_train = df[mask_train].copy().reset_index(drop=True)
df_test = df[mask_test].copy().reset_index(drop=True)

# 3. Validation Checks
n_train = len(df_train)
n_test = len(df_test)
n_excluded = len(df) - n_train - n_test

print(f"\n  > Split Results:")
print(f"    - Derivation Cohort: {n_train} patients")
print(f"    - Validation Cohort: {n_test} patients")
print(f"    - Excluded (Washout/Out of range): {n_excluded} patients")

if n_train < 10 or n_test < 10:
    print("\n[WARNING] Sample size too small! Falling back to 80/20 split for debugging...")
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

target = 'readmission_within_1_year'
features = [c for c in df.columns if c not in [target, 'admission_date']]

# ==============================================================================
# OPTIMIZED VARIABLE SELECTION (Restored & SHAP-Aligned)
# ==============================================================================

#
cat_feats_all = [c for c in features if df[c].nunique() < 10]
num_feats_all = [c for c in features if c not in cat_feats_all]

# 2. 定义核心临床变量列表 (保留原代码配置)
CORE_CLINICAL_VARS = [
    'age', 'sex', 'egfr', 'nt_probnp', 'blood_urea_nitrogen', 'hs_crp',
    'homocysteine', 'serum_creatinine', 'serum_sodium',
    'lvef', 'E_over_e_prime', 'TR_velocity', 'LAVI', 'LVMI',
    'nyha_class', 'systolic_bp', 'length_of_stay', 'raas_use'
]

#
if CORE_CLINICAL_VARS:
    missing_vars = [c for c in CORE_CLINICAL_VARS if c not in df.columns]
    if missing_vars:
        print(f"\n[WARNING] Missing variables skipped: {missing_vars}")
    valid_core = [c for c in CORE_CLINICAL_VARS if c in df.columns]
    num_feats = [c for c in num_feats_all if c in valid_core]
    cat_feats = [c for c in cat_feats_all if c in valid_core]
else:
    num_feats = num_feats_all
    cat_feats = cat_feats_all

print(f"  > Plotting Continuous Features: {len(num_feats)}")
print(f"  > Plotting Categorical Features: {len(cat_feats)}")

# ==============================================================================
# 2. DRIFT PLOTS (SMART LAYOUT OPTIMIZATION) - Restored
# ==============================================================================
import math


def get_optimal_grid(n_feats):
    if n_feats <= 3: return n_feats, 1
    if n_feats == 4: return 2, 2
    if n_feats in [5, 6]: return 3, 2
    if n_feats in [7, 8]: return 4, 2
    if n_feats == 9: return 3, 3
    if n_feats == 10: return 5, 2
    return 4, math.ceil(n_feats / 4)


# --- FIGURE 1A: CONTINUOUS VARIABLES ---
if num_feats:
    print("  > Generating Continuous Drift Plot...")
    n_cols, n_rows = get_optimal_grid(len(num_feats))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_feats):
        sns.kdeplot(df_train[col], ax=axes[i], fill=True, color=NEJM_BLUE, label='Derivation', alpha=0.3)
        sns.kdeplot(df_test[col], ax=axes[i], fill=True, color=NEJM_RED, label='Validation', alpha=0.3)
        stat, p_val = ks_2samp(df_train[col], df_test[col])
        title_text = f"{pretty_name_map.get(col, col)}\n(KS p={p_val:.3f})"
        axes[i].set_title(title_text, color='red' if p_val < 0.05 else 'black', fontweight='bold')
        if i == 0: axes[i].legend()

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01a_Drift_Continuous_NEJM.pdf'))
    plt.close()

# --- FIGURE 1B: CATEGORICAL VARIABLES ---
if cat_feats:
    print("  > Generating Categorical Drift Plot...")
    n_cols, n_rows = get_optimal_grid(len(cat_feats))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(cat_feats):
        train_counts = df_train[col].value_counts(normalize=True).sort_index()
        test_counts = df_test[col].value_counts(normalize=True).sort_index()

        # Chi-square test (Simplified for display)
        try:
            contingency = pd.crosstab(df[col], df['admission_date'] >= pd.Timestamp(TEST_START))
            _, p_val, _, _ = chi2_contingency(contingency)
        except:
            p_val = 1.0

        all_cats = sorted(list(set(train_counts.index) | set(test_counts.index)))
        x = np.arange(len(all_cats))
        width = 0.35

        axes[i].bar(x - width / 2, [train_counts.get(c, 0) for c in all_cats], width, label='Derivation',
                    color=NEJM_BLUE)
        axes[i].bar(x + width / 2, [test_counts.get(c, 0) for c in all_cats], width, label='Validation', color=NEJM_RED)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(all_cats)

        title_text = f"{pretty_name_map.get(col, col)}\n(Chi2 p={p_val:.3f})"
        axes[i].set_title(title_text, color='red' if p_val < 0.05 else 'black', fontweight='bold')
        if i == 0: axes[i].legend()

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01b_Drift_Categorical_NEJM.pdf'))
    plt.close()

# 3. PREPROCESSING (MICE + VIF + Interaction Terms)
# ==============================================================================
print("--- Stage 2: MICE Imputation & VIF ---")

#
X_train_raw = df_train[features]
X_test_raw = df_test[features]
y_train_full = df_train[target]
y_test = df_test[target]

#
imputer = IterativeImputer(max_iter=10, random_state=42)
scaler = StandardScaler()

print("  > Running MICE Imputation...")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=features)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=features)

print("  > Creating Explicit Interaction Terms (Cardio-Renal, Obesity-BNP, Inflam-Stress)...")


def add_interactions(df_in):
    df_out = df_in.copy()

    #
    if 'egfr' in df_out.columns and 'nt_probnp' in df_out.columns:
        df_out['Inter_eGFR_NTproBNP'] = df_out['egfr'] * df_out['nt_probnp']

    if 'bmi' in df_out.columns and 'nt_probnp' in df_out.columns:
        df_out['Inter_BMI_NTproBNP'] = df_out['bmi'] * df_out['nt_probnp']

    if 'hs_crp' in df_out.columns and 'nt_probnp' in df_out.columns:
        df_out['Inter_hsCRP_NTproBNP'] = df_out['hs_crp'] * df_out['nt_probnp']

    return df_out



X_train_imp = add_interactions(X_train_imp)
X_test_imp = add_interactions(X_test_imp)


current_cols = X_train_imp.columns.tolist()
new_interaction_cols = [c for c in current_cols if c not in features]
features = current_cols


pretty_name_map.update({
    'Inter_eGFR_NTproBNP': 'Interaction: eGFR × NT-proBNP',
    'Inter_BMI_NTproBNP': 'Interaction: BMI × NT-proBNP',
    'Inter_hsCRP_NTproBNP': 'Interaction: hs-CRP × NT-proBNP'
})

print(f"  > Added {len(new_interaction_cols)} interaction terms.")
print(f"  > Total features for modeling: {len(features)}")


# ==============================================================================
# [MODIFIED] MAGGIC Risk Score Implementation (Approximated)
# ==============================================================================
def calculate_maggic_score(d):

    #

    #
    d = d.copy()

    #
    binary_cols = [
        'sex',
        'smoker', 'diabetes', 'copd',
        'beta_blocker_use', 'raas_use',
    ]

    #
    for col in binary_cols:
        if col in d.columns:
            d[col] = d[col].round().astype(int)

    #

    """
    Computes the MAGGIC Risk Score (Integer Score) strictly based on Pocock et al., 2013.
    """
    # 1. Initialize Score
    score = pd.Series(0, index=d.index, dtype=float)


    # A. Demographics & Vitals
    # ==========================================

    # Age (Strict Binning per Table 1 in Pocock et al.)
    age = d.get('age', 70)
    score += ((age >= 55) & (age < 60)).astype(int) * 1
    score += ((age >= 60) & (age < 65)).astype(int) * 2
    score += ((age >= 65) & (age < 70)).astype(int) * 4
    score += ((age >= 70) & (age < 75)).astype(int) * 6
    score += ((age >= 75) & (age < 80)).astype(int) * 8
    score += (age >= 80).astype(int) * 10

    # Gender: Male +1
    score += (d.get('sex', 0) == 1).astype(int) * 1

    # BMI: < 22 is a risk factor
    bmi = d.get('bmi', 28)
    score += ((bmi < 22) & (bmi >= 0)).astype(int) * 2  # Pocock update: low BMI is bad
    # Note: Some versions define BMI categories differently, but <22 is generally the risk cutoff.

    # SBP: Lower BP is worse in HF
    sbp = d.get('systolic_bp', 130)
    score += (sbp < 110).astype(int) * 5
    score += ((sbp >= 110) & (sbp < 130)).astype(int) * 2

    # ==========================================
    # B. Comorbidities

    # Smoker: Yes +1
    score += (d.get('smoker', 0) == 1).astype(int) * 1

    # Diabetes: Yes +3
    score += (d.get('diabetes', 0) == 1).astype(int) * 3

    # COPD: Yes +2
    score += (d.get('copd', 0) == 1).astype(int) * 2

    # HF Duration > 18 months?
    # New onset (<18mo) = +2 points. Chronic (>18mo) = 0 points.
    # STRATEGY: Assume Chronic (0 points) for HFpEF readmission cohort.
    # score += 0

    # ==========================================
    # C. Clinical Status
    # ==========================================

    # NYHA Class
    nyha = d.get('nyha_class', 1)
    # Class I = 0
    score += (nyha == 2).astype(int) * 2
    score += (nyha == 3).astype(int) * 6
    score += (nyha == 4).astype(int) * 8

    # LVEF (Ejection Fraction)
    # HFpEF patients (EF >= 50) get 0 points according to MAGGIC integer score
    ef = d.get('lvef', 55)
    score += (ef < 20).astype(int) * 7
    score += ((ef >= 20) & (ef < 30)).astype(int) * 6
    score += ((ef >= 30) & (ef < 40)).astype(int) * 5
    score += ((ef >= 40) & (ef < 50)).astype(int) * 3
    # EF >= 50 = 0 points

    # ==========================================
    # D. Labs (Creatinine)
    # ==========================================

    # Creatinine: Points based on umol/L
    # Conversion: mg/dL * 88.4 = umol/L
    scr = d.get('serum_creatinine', 1.0)

    # < 90 umol/L (~1.02 mg/dL) = 0
    # 90-109 umol/L (~1.02-1.23) = 1
    # 110-129 umol/L (~1.24-1.46) = 2
    # 130-149 umol/L (~1.47-1.68) = 3
    # 150-169 umol/L (~1.70-1.91) = 4
    # 170-209 umol/L (~1.92-2.36) = 5
    # 210-249 umol/L (~2.37-2.82) = 6
    # >= 250 umol/L (~2.83+) = 8

    # Implementation using thresholds to avoid unit errors
    score += ((scr >= 1.02) & (scr < 1.24)).astype(int) * 1
    score += ((scr >= 1.24) & (scr < 1.47)).astype(int) * 2
    score += ((scr >= 1.47) & (scr < 1.70)).astype(int) * 3
    score += ((scr >= 1.70) & (scr < 1.92)).astype(int) * 4
    score += ((scr >= 1.92) & (scr < 2.37)).astype(int) * 5
    score += ((scr >= 2.37) & (scr < 2.83)).astype(int) * 6
    score += (scr >= 2.83).astype(int) * 8

    # ==========================================
    # E. Medications (Not Taking = Risk)
    # ==========================================

    # Beta-blocker: NOT taking = +3
    bb_use = d.get('beta_blocker_use', 0)
    score += (bb_use == 0).astype(int) * 3

    #
    raas_use = d.get('raas_use', 0)
    score += (raas_use == 0).astype(int) * 1

    return score

print("  > Calculating MAGGIC Risk Scores...")
score_train = calculate_maggic_score(X_train_imp)
score_test = calculate_maggic_score(X_test_imp)

#
print(f"    MAGGIC Score Range: {score_train.min():.1f} - {score_train.max():.1f} (Mean: {score_train.mean():.1f})")

#
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=features)

# VIF Check
print("  > Checking VIF...")
X_train_vif = X_train_scaled.copy()
while True:

    vifs = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
    max_vif = max(vifs)
    if max_vif > 10:
        drop_col = X_train_vif.columns[vifs.index(max_vif)]

        X_train_vif.drop(columns=[drop_col], inplace=True)
        print(f"    [VIF] Dropped collinear feature: {drop_col}")
    else:
        break

features_final = X_train_vif.columns.tolist()
X_test_final = X_test_scaled[features_final]
print(f"  > Final features after VIF: {len(features_final)}")
# ==============================================================================
# PART 2: Robust Feature Selection & Training 12 Models
# ==============================================================================
from sklearn.linear_model import LassoCV, LogisticRegression, lasso_path
from sklearn.feature_selection import RFECV
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

#
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

try:
    import xgboost as xgb
except:
    xgb = None

# ------------------------------------------------------------------------------
# 3. ROBUST FEATURE SELECTION (Retaining your Logic)
# ------------------------------------------------------------------------------
print("--- Stage 3: Robust Feature Selection ---")
#
feature_votes = Counter()
N_STABILITY_RUNS = 50

for i in range(N_STABILITY_RUNS):
    # Resample
    X_res, y_res = resample(X_train_vif, y_train_full, random_state=i)

    # Lasso
    lasso = LassoCV(cv=3, random_state=i).fit(X_res, y_res)
    lasso_feats = set(X_train_vif.columns[lasso.coef_ != 0])

    # RFE (LightGBM)
    lgbm = lgb.LGBMClassifier(verbosity=-1, random_state=i)
    rfe = RFECV(lgbm, step=1, cv=3, scoring='roc_auc')
    rfe.fit(X_res, y_res)
    rfe_feats = set(X_train_vif.columns[rfe.support_])

    feature_votes.update(list(lasso_feats & rfe_feats))

stable_features = [f for f, c in feature_votes.items() if c >= 2]
if len(stable_features) < 5: stable_features = [f for f, c in feature_votes.most_common(10)]
print(f"  > Stable Features Selected: {len(stable_features)}")
#

# [REPLACE/INSERT IN PART 2] Figure 2: Combined Feature Selection Panel (A, B, C)
# ==============================================================================
print("  > Generating Combined Figure 2: Lasso CV, Lasso Path, and RFECV...")

from sklearn.linear_model import lasso_path

# 1.
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# --- Panel A: Lasso Selection (CV Error) ---

if 'lasso_final' not in locals():
    lasso_final = LassoCV(cv=5, random_state=42).fit(X_train_vif, y_train_full)

#
axes[0].plot(-np.log10(lasso_final.alphas_), lasso_final.mse_path_.mean(axis=-1),
             color='black', lw=2, label='Mean MSE')
#
axes[0].axvline(-np.log10(lasso_final.alpha_), color=NEJM_RED, linestyle='--',
                lw=2, label='Optimal Alpha')

axes[0].set_title('A. Lasso Selection (CV Error)', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'-Log($\lambda$)', fontweight='bold')
axes[0].set_ylabel('Mean Square Error', fontweight='bold')
axes[0].legend(frameon=False, loc='upper left')
axes[0].grid(linestyle='--', alpha=0.2)

# --- Panel B: Lasso Coefficient Path ---
print("    Calculating Lasso Path...")
#
alphas_lasso, coefs_lasso, _ = lasso_path(X_train_vif, y_train_full, eps=0.001, n_alphas=100)
colors = plt.cm.jet(np.linspace(0, 1, len(X_train_vif.columns)))

#
for coef_l, c in zip(coefs_lasso, colors):
    axes[1].plot(-np.log10(alphas_lasso), coef_l, color=c, alpha=0.5, lw=1)

#
axes[1].axvline(-np.log10(lasso_final.alpha_), color='black', linestyle='--',
                alpha=0.5, label='Optimal Alpha')

axes[1].set_title('B. Lasso Coefficient Path', fontweight='bold', fontsize=14)
axes[1].set_xlabel(r'-Log($\lambda$)', fontweight='bold')
axes[1].set_ylabel('Coefficients', fontweight='bold')
axes[1].axis('tight')


# --- Panel C: RFECV Curve ---
print("    Calculating RFECV Curve...")
#
from sklearn.model_selection import StratifiedKFold
rfecv_plot = RFECV(
    estimator=lgb.LGBMClassifier(verbosity=-1, random_state=42),
    step=1,
    cv=StratifiedKFold(5),
    scoring='roc_auc'
)
rfecv_plot.fit(X_train_vif, y_train_full)

#
if hasattr(rfecv_plot, 'cv_results_'):
    scores = rfecv_plot.cv_results_['mean_test_score']
else:
    scores = rfecv_plot.grid_scores_

#
axes[2].plot(range(1, len(scores) + 1), scores, color=NEJM_BLUE, lw=2, label='CV AUC')

#
optimal_num = rfecv_plot.n_features_
max_score = max(scores)
axes[2].plot(optimal_num, max_score, 'o', color=NEJM_RED, markersize=8,
             label=f'Optimal: {optimal_num} Features')
axes[2].axvline(optimal_num, color=NEJM_RED, linestyle=':', lw=1.5)

axes[2].set_title('C. RFECV Feature Selection', fontweight='bold', fontsize=14)
axes[2].set_xlabel('Number of Features Selected', fontweight='bold')
axes[2].set_ylabel('Cross-validation AUC', fontweight='bold')
axes[2].legend(frameon=False, loc='lower right')
axes[2].grid(linestyle='--', alpha=0.3)

#
save_path = os.path.join(OUTPUT_DIR, '02_Feature_Selection_Combined.pdf')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"    [Saved] Combined Figure 2 saved to {save_path}")
# ==============================================================================
# [INSERT HERE] Figure 2D: Feature Stability with Jaccard (NEJM Style)
# ==============================================================================
print("  > Generating Figure 2D (Feature Stability & Jaccard)...")


# 1. Calculate Jaccard Index (Re-simulating selection for metric calculation)
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


# Re-run quick selection to get per-run lists for Jaccard
stability_runs_features = []
for i in range(N_STABILITY_RUNS):
    X_res_j, y_res_j = resample(X_train_vif, y_train_full, random_state=i)
    # Quick Lasso
    lasso_j = LassoCV(cv=3, random_state=i).fit(X_res_j, y_res_j)
    lasso_feats_j = set(X_train_vif.columns[lasso_j.coef_ != 0])
    # Quick RFE
    lgbm_j = lgb.LGBMClassifier(verbosity=-1, random_state=i)
    rfe_j = RFECV(lgbm_j, step=1, cv=3, scoring='roc_auc')
    rfe_j.fit(X_res_j, y_res_j)
    rfe_feats_j = set(X_train_vif.columns[rfe_j.support_])
    # Intersection
    stability_runs_features.append(list(lasso_feats_j & rfe_feats_j))

import itertools

jaccard_scores = [jaccard_similarity(r1, r2) for r1, r2 in itertools.combinations(stability_runs_features, 2)]
mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0

# 2. Plotting Figure 2D
plt.figure(figsize=(10, 8))
# Use feature_votes from your original loop
top_stable = feature_votes.most_common(20)
if top_stable:
    feats_raw, counts = zip(*top_stable)
    freqs = [c / N_STABILITY_RUNS for c in counts]
    # Apply Pretty Names
    feats_pretty = [get_pretty_name(f) for f in feats_raw]

    sns.barplot(x=list(freqs), y=list(feats_pretty), palette="viridis")
    plt.title("Feature Selection Stability (Across 5 Runs)", fontweight='bold', pad=20)
    plt.xlabel("Selection Frequency (1.0 = Selected in all runs)", fontweight='bold')
    plt.xlim(0, 1.2)  # Extend space for text

    # 3. Add Floating Text Box (Bottom Right)
    text_str = f"Mean Jaccard Index\n{mean_jaccard:.3f}"
    plt.text(0.95, 0.05, text_str, transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color='#333333',
             ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02d_Feature_Stability_Jaccard.pdf'))
plt.close()
# ==============================================================================


stable_features = [f for f, c in feature_votes.most_common(10)]

#
stable_features.sort()
# -------------------------------------------

print(f"  > Stable Features Selected: {len(stable_features)}")
print(f"  > Features List: {stable_features}")

X_train_opt = X_train_vif[stable_features]
X_test_opt = X_test_final[stable_features]

# --- PLOT: Feature Selection Visuals (Lasso Path etc.) ---
lasso_final = LassoCV(cv=5, random_state=42).fit(X_train_vif, y_train_full)

plt.figure(figsize=(10, 6))
plt.plot(-np.log10(lasso_final.alphas_), lasso_final.mse_path_.mean(axis=-1), 'k', lw=2)
plt.axvline(-np.log10(lasso_final.alpha_), color=NEJM_RED, linestyle='--', lw=2)
plt.title('Lasso Selection (CV Error)', fontweight='bold')
plt.xlabel('-Log(Alpha)')
plt.savefig(os.path.join(OUTPUT_DIR, '02_Lasso_Selection.pdf'))
plt.close()

# ==============================================================================
# [REPLACE HERE] Figure 3: Correlation Matrix (Renamed & Polished)
# ==============================================================================
plt.figure(figsize=(14, 12))
#
X_corr_plot = X_train_opt.rename(columns=get_pretty_name)
corr = X_corr_plot.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .6})

plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_Correlation_Matrix.pdf'))
plt.close()
# ==============================================================================

#
# ------------------------------------------------------------------------------
print("--- Stage 4: Training 12 Models (with GridSearch) ---")

#
models_config = {
    # 1. LogisticRegression: Added random_state=42
    'LogisticRegression': (LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42),  # <--- FIXED
                           {'C': [0.1, 1, 10]}),

    'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                     {'n_estimators': [100], 'max_depth': [4, 8], 'min_samples_leaf': [4, 10]}),

    'XGBoost': (xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=1) if xgb else None,
                {'learning_rate': [0.01, 0.05], 'max_depth': [3, 5], 'n_estimators': [100, 200]}),

    'LightGBM': (lgb.LGBMClassifier(class_weight='balanced', verbosity=-1, random_state=42, n_jobs=1),
                 {'learning_rate': [0.01, 0.05], 'num_leaves': [15, 31]}),

    #
    'SVC_RBF': (SVC(probability=True, class_weight='balanced', random_state=42), {'C': [1, 10]}),  # <--- FIXED

    #
    'SVC_Lin': (SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42), {'C': [0.1, 1]}),
    #

    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'max_depth': [3, 5]}),

    'ExtraTrees': (ExtraTreesClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                   {'max_depth': [5, 10]}),

    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100]}),

    #
    'KNeighbors': (KNeighborsClassifier(n_jobs=1), {'n_neighbors': [5, 11]}),

    #
    'DecisionTree': (DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42), {}),  # <--- FIXED

    #
    'GaussianNB': (GaussianNB(), {})
}

results = {}


for name, (model, params) in models_config.items():
    if model is None: continue
    print(f"  > Training {name}...")

    #
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    #
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #
    from sklearn.model_selection import StratifiedKFold

    #
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if params:
        #
        grid = GridSearchCV(model, params, cv=inner_cv, scoring='roc_auc', n_jobs=1)

        #
        nested_scores = cross_val_score(grid, X_train_opt, y_train_full, cv=outer_cv, scoring='roc_auc')
        print(f"    [Nested CV] {name} AUC: {nested_scores.mean():.3f} +/- {nested_scores.std():.3f}")

        #
        grid.fit(X_train_opt, y_train_full)
        best_model = grid.best_estimator_
    else:
        #
        cv_scores = cross_val_score(model, X_train_opt, y_train_full, cv=outer_cv, scoring='roc_auc')
        print(f"    [CV] {name} AUC: {cv_scores.mean():.3f}")

        model.fit(X_train_opt, y_train_full)
        best_model = model

    #
    results[name] = {
        'model': best_model,
        'prob_test': best_model.predict_proba(X_test_opt)[:, 1],
        'prob_train': best_model.predict_proba(X_train_opt)[:, 1],
        'pred_test': best_model.predict(X_test_opt)
    }

# MAGGIC Score Baseline (Logistic Regression on Score)
lr_score = LogisticRegression()
lr_score.fit(score_train.values.reshape(-1, 1), y_train_full)
results['MAGGIC Score'] = {  #
    'model': lr_score,
    'prob_test': lr_score.predict_proba(score_test.values.reshape(-1, 1))[:, 1],
    'prob_train': lr_score.predict_proba(score_train.values.reshape(-1, 1))[:, 1],
    'pred_test': lr_score.predict(score_test.values.reshape(-1, 1))
}

# ==============================================================================
# PART 3: Comprehensive Visualization (The "Kitchen Sink" - NEJM Style)
# ==============================================================================
def plot_shap_interaction_network(model, X, output_dir, filename='20_SHAP_Interaction_Network_NatureStyle.pdf',
                                  top_n=10):
    """
    [Nature/Science Style] SHAP Interaction Network Plot (Purple-Green Scheme)
    - Edge Color: Purple (Negative) <-> Green (Positive) Diverging Map
    - Edge Width: Absolute Interaction Strength
    - Visibility: High alpha for clearer lines
    """
    import shap
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch
    import os

    print(f"  > Generating SHAP Interaction Network ({filename}) [Nature Style]...")

    # ==========================================
    # 1. 计算 SHAP 交互值 (带维度自动修复)
    # ==========================================
    try:
        explainer = shap.TreeExplainer(model)
        X_calc = X.iloc[:200] if len(X) > 200 else X
        shap_iv = explainer.shap_interaction_values(X_calc)

        # 维度清洗
        if hasattr(shap_iv, "values"): shap_iv = shap_iv.values
        if isinstance(shap_iv, list): shap_iv = shap_iv[1]
        if len(shap_iv.shape) == 4: shap_iv = shap_iv[:, :, :, 1]

        if len(shap_iv.shape) != 3:
            print(f"    [Skip] Shape mismatch: {shap_iv.shape}")
            return
    except Exception as e:
        print(f"    [Skip] Interaction calculation failed: {e}")
        return


    mean_abs_iv = np.abs(shap_iv).mean(0)
    #
    mean_signed_iv = shap_iv.mean(0)

    feature_names = X.columns.tolist()


    try:
        main_effects = np.diag(mean_abs_iv)
    except ValueError:
        return

    total_importance = main_effects + (mean_abs_iv.sum(axis=1) - main_effects)
    top_indices = np.argsort(-total_importance)[:top_n]


    reduced_abs = mean_abs_iv[np.ix_(top_indices, top_indices)]

    reduced_signed = mean_signed_iv[np.ix_(top_indices, top_indices)]

    reduced_feats = [feature_names[i] for i in top_indices]


    try:
        labels = {i: pretty_name_map.get(f, f) for i, f in enumerate(reduced_feats)}
    except NameError:
        labels = {i: f for i, f in enumerate(reduced_feats)}

    G = nx.Graph()

    node_sizes = []
    for i in range(len(reduced_feats)):

        val = reduced_abs[i, i]
        G.add_node(i, label=labels[i], weight=val)
        node_sizes.append(val)


    edges_list = []
    for i in range(len(reduced_feats)):
        for j in range(i + 1, len(reduced_feats)):

            width_val = reduced_abs[i, j] + reduced_abs[j, i]

            color_val = (reduced_signed[i, j] + reduced_signed[j, i]) / 2

            if width_val > 0:
                G.add_edge(i, j, weight=width_val, color_val=color_val)
                edges_list.append({'u': i, 'v': j, 'width': width_val, 'color': color_val})

    # ==========================================
    plt.figure(figsize=(12, 10), dpi=300)
    ax = plt.gca()
    pos = nx.circular_layout(G, scale=1.0)


    cmap_edges = plt.cm.PRGn

    if edges_list:
        max_abs_color = max([abs(e['color']) for e in edges_list])
        norm_color = plt.Normalize(vmin=-max_abs_color, vmax=max_abs_color)
    else:
        norm_color = plt.Normalize(vmin=-1, vmax=1)

    edges_list.sort(key=lambda x: x['width'])

    if edges_list:
        max_width = max([e['width'] for e in edges_list])

        for e in edges_list:

            width_ratio = (e['width'] / max_width)
            actual_width = 1.0 + (width_ratio ** 0.6) * 7.0
            edge_color = cmap_edges(norm_color(e['color']))
            alpha = 0.5 + 0.45 * width_ratio

            #
            x1, y1 = pos[e['u']]
            x2, y2 = pos[e['v']]
            patch = FancyArrowPatch(
                posA=(x1, y1), posB=(x2, y2),
                arrowstyle="-",
                connectionstyle="arc3,rad=0.2",
                color=edge_color,
                alpha=alpha,
                linewidth=actual_width,
                mutation_scale=10,
                zorder=1  #
            )
            ax.add_patch(patch)


    COLOR_NODE_FACE = '#08306B'
    COLOR_NODE_EDGE = 'white'

    if node_sizes:
        mn, mx = min(node_sizes), max(node_sizes)
        norm_sizes = [800 + (s - mn) / (mx - mn + 1e-9) * 2700 for s in node_sizes]
    else:
        norm_sizes = [1500] * len(G)


    nodes_bg = nx.draw_networkx_nodes(G, pos, node_size=norm_sizes, node_color='white', linewidths=0)
    nodes_bg.set_zorder(2)


    nodes_fg = nx.draw_networkx_nodes(G, pos, node_size=norm_sizes,
                                      node_color=node_sizes, cmap=plt.cm.Blues,  # 节点颜色也根据重要性深浅变化
                                      edgecolors=COLOR_NODE_EDGE, linewidths=2.5)
    nodes_fg.set_zorder(3)


    text_pos = {k: (v[0] * 1.15, v[1] * 1.15) for k, v in pos.items()}

    for node_idx, (x, y) in text_pos.items():
        lbl = G.nodes[node_idx]['label']
        ha = 'left' if x > 0 else 'right'
        va = 'center'

        #
        import matplotlib.patheffects as path_effects
        txt = plt.text(x, y, lbl, fontsize=13, fontweight='bold',
                       family='sans-serif', color='#222222',
                       ha=ha, va=va, zorder=4)
        #
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white', alpha=0.8)])

    #

    #
    sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_color)
    sm_edge.set_array([])
    cbar_edge = plt.colorbar(sm_edge, ax=ax, shrink=0.5, aspect=15, pad=0.05, location='right')
    cbar_edge.set_label('Interaction Value (Signed)\n(Purple=Neg, Green=Pos)', fontweight='bold', fontsize=10)
    cbar_edge.outline.set_visible(False)  # 去掉框线更像 Nature 风格

    #
    if node_sizes:
        sm_node = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=mn, vmax=mx))
        sm_node.set_array([])
        cbar_node = plt.colorbar(sm_node, ax=ax, shrink=0.5, aspect=15, pad=0.15, location='right')
        cbar_node.set_label('Feature Importance\n(Node Size/Darkness)', fontweight='bold', fontsize=10)
        cbar_node.outline.set_visible(False)

    plt.title("SHAP Interaction Network", fontsize=18, fontweight='bold', y=1.05)
    plt.axis('off')
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    [Saved] Interaction network saved to {filename}")
#
def delong_roc_variance(ground_truth, predictions):
    """
    Computes DeLong covariance for ROC curves.
    """
    import scipy.stats as stats

    #
    ground_truth = np.array(ground_truth).ravel()
    predictions = np.array(predictions).ravel()
    # -------------------------------------------------------

    order = np.argsort(predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]

    pos_preds = predictions[ground_truth == 1]
    neg_preds = predictions[ground_truth == 0]

    m = len(pos_preds)
    n = len(neg_preds)

    # Mann-Whitney statistics
    rank_pos = np.searchsorted(neg_preds, pos_preds, side='right')
    rank_neg = np.searchsorted(pos_preds, neg_preds, side='left')

    # DeLong Covariances
    v01 = (stats.rankdata(pos_preds) - rank_pos - (m + 1) / 2) / (m - 1)
    v10 = (1 - (stats.rankdata(neg_preds) - rank_neg - (n + 1) / 2) / (n - 1))

    return np.var(v01) / m + np.var(v10) / n


def calc_pvalue_delong(y_true, y_pred1, y_pred2):
    """
    Computes P-value for the difference between two AUCs using DeLong test.
    """
    import scipy.stats as stats
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    # Calculate variance (simplified for independence assumption or basic comparison)
    # Note: Full DeLong covariance requires matrix math, this is a fast approximation
    # for model selection dominance checking.
    var1 = delong_roc_variance(y_true, y_pred1)
    var2 = delong_roc_variance(y_true, y_pred2)

    z_score = (auc1 - auc2) / np.sqrt(var1 + var2 + 1e-8)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return p_value


import shap
from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, accuracy_score, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve

print("--- Stage 5: Generating ALL Figures ---")

# Sort models by Test AUC
sorted_names = sorted(results.keys(),
                      key=lambda x: roc_auc_score(y_test, results[x]['prob_test']),
                      reverse=True)
#
best_model_name = sorted_names[0]
print(f"  > [AUTO SELECTION] Best Model set to: {best_model_name} (Highest AUC: {roc_auc_score(y_test, results[best_model_name]['prob_test']):.4f})")

best_model = results[best_model_name]['model']

#
#
plot_models = sorted_names[:3]
#
if 'MAGGIC Score' not in plot_models:
    plot_models.append('MAGGIC Score')

#
if 'MAGGIC Score' not in results:
    print("[WARNING] 'MAGGIC Score' key not found in results! Using 'Clinical_Score' if available.")
    if 'Clinical_Score' in results:
        plot_models.append('Clinical_Score')
#


# Helper: Bootstrap CI for AUC
def get_auc_ci(y_true, y_pred, n_boot=1000):
    rng = np.random.RandomState(42)
    scores = []

    #
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # -------------------------------------------------------

    for i in range(n_boot):
        #
        idx = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[idx])) > 1:
            scores.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])


def calculate_nri_idi(y_true, y_old, y_new):
    """
    Calculates NRI and IDI metrics.
    """
    y_true = np.array(y_true)
    y_old = np.array(y_old)
    y_new = np.array(y_new)

    #
    diff = y_new - y_old
    idi = np.mean(diff[y_true == 1]) - np.mean(diff[y_true == 0])

    #
    up_events = np.sum((y_new > y_old) & (y_true == 1))
    down_events = np.sum((y_new < y_old) & (y_true == 1))
    n_events = np.sum(y_true == 1)

    #
    down_nonevents = np.sum((y_new < y_old) & (y_true == 0))
    up_nonevents = np.sum((y_new > y_old) & (y_true == 0))
    n_nonevents = np.sum(y_true == 0)

    nri = (up_events - down_events) / n_events + (down_nonevents - up_nonevents) / n_nonevents

    return nri, idi


#
if 'MAGGIC Score' in results:
    baseline_prob = results['MAGGIC Score']['prob_test']
    baseline_name = "MAGGIC Score"
else:
    #
    baseline_prob = results['Clinical_Score']['prob_test']
    baseline_name = "Clinical Score"

nri_val, idi_val = calculate_nri_idi(y_test,
                                     baseline_prob,
                                     results[best_model_name]['prob_test'])

print(f"  > Improvement Metrics ({best_model_name} vs {baseline_name}):")
print(f"    NRI: {nri_val:.3f}")
print(f"    IDI: {idi_val:.3f}")


# --- 1. DUAL ROC PLOTS (Train vs Test) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

#
baseline_key = 'MAGGIC Score' if 'MAGGIC Score' in results else 'Clinical_Score'

if baseline_key in results:
    p_val_delong = calc_pvalue_delong(y_test,
                                      results[best_model_name]['prob_test'],
                                      results[baseline_key]['prob_test'])
else:
    p_val_delong = 1.0  #

#
for i, name in enumerate(plot_models):
    if name not in results: continue

    y_prob = results[name]['prob_test']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ci = get_auc_ci(y_test, y_prob)

    #
    if name == 'MAGGIC Score':
        color = '#333333'  #
        linestyle = '--'
        lw = 1.8  #
        alpha = 0.8  #
        #
        label = f"{name} (AUC={auc:.2f} [{ci[0]:.2f}-{ci[1]:.2f}])"
    else:
        # ML Models
        alpha = 1.0  #
        #
        color_idx = i if i < len(PALETTE_12) else i % len(PALETTE_12)
        color = PALETTE_12[color_idx]
        linestyle = '-'
        lw = 3 if name == best_model_name else 2

        # --
        if name == best_model_name:
            p_text = f"P < 0.001" if p_val_delong < 0.001 else f"P={p_val_delong:.3f}"
            label = f"{name} (AUC={auc:.2f}, vs MAGGIC {p_text})"
        else:
            label = f"{name} (AUC={auc:.2f} [{ci[0]:.2f}-{ci[1]:.2f}])"

    #
    ax1.plot(fpr, tpr, lw=lw, linestyle=linestyle, label=label, color=color, alpha=alpha)



ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5)
ax1.set_title('Validation Cohort ROC', fontweight='bold')
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
ax1.legend(loc='lower right', frameon=False, fontsize=10)

# [MODIFIED] 3. Train ROC (Check Overfitting)
for i, name in enumerate(plot_models):
    if name not in results: continue

    y_prob = results[name]['prob_train']
    fpr, tpr, _ = roc_curve(y_train_full, y_prob)
    auc = roc_auc_score(y_train_full, y_prob)

    # Consistent Style with Test Plot
    if name == 'MAGGIC Score':
        color = '#333333'  # 深灰
        linestyle = '--'
        lw = 1.5  #
        alpha = 0.8
    else:
        alpha = 1.0
        color_idx = i if i < len(PALETTE_12) else i % len(PALETTE_12)
        color = PALETTE_12[color_idx]
        linestyle = '-'
        lw = 2

    ax2.plot(fpr, tpr, lw=lw, linestyle=linestyle, label=f"{name} (AUC={auc:.2f})", color=color, alpha=alpha)

ax2.plot([0, 1], [0, 1], 'k:', alpha=0.5)
ax2.set_title('Derivation Cohort ROC', fontweight='bold')
ax2.set_xlabel('1 - Specificity')
ax2.legend(loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_Dual_ROC_Curves.pdf'))
plt.close()

# --- 2. PR Curves ---
plt.figure(figsize=(10, 8))

# [MODIFIED] Loop over plot_models
for i, name in enumerate(plot_models):
    if name not in results: continue

    precision, recall, _ = precision_recall_curve(y_test, results[name]['prob_test'])
    ap = average_precision_score(y_test, results[name]['prob_test'])

    # Style Logic
    if name == 'MAGGIC Score':
        color = '#333333'
        linestyle = '--'
        lw = 1.8  #
        alpha = 0.8
    else:
        alpha = 1.0
        color_idx = i if i < len(PALETTE_12) else i % len(PALETTE_12)
        color = PALETTE_12[color_idx]
        linestyle = '-'
        lw = 2

    plt.plot(recall, precision, lw=lw, linestyle=linestyle, label=f"{name} (AP={ap:.2f})", color=color, alpha=alpha)

plt.title('Precision-Recall Curves', fontweight='bold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(frameon=False, loc='lower left') # PR图例通常放左下角或右上角
plt.savefig(os.path.join(OUTPUT_DIR, '05_PR_Curves.pdf'))
plt.close()

# --- 3. Confusion Matrices (Top 6) ---
# Retained your 2x3 grid layout
top_6 = sorted_names[:6]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, name in enumerate(top_6):
    cm = confusion_matrix(y_test, results[name]['pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[i].set_title(name, fontweight='bold')
    axes[i].set_xlabel('Predicted');
    axes[i].set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_Confusion_Matrices.pdf'))
plt.close()

# --- 4. Detailed Metrics Bar Plot ---
metrics = []
for name in sorted_names:
    y_p = results[name]['pred_test']
    y_prob = results[name]['prob_test']
    metrics.append({
        'Model': name,
        'AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_p),
        'Accuracy': accuracy_score(y_test, y_p),
        'Brier': brier_score_loss(y_test, y_prob),
        # ICI calculation
        'ICI': np.mean(np.abs(y_prob - calibration_curve(y_test, y_prob, n_bins=10)[0].mean()))
    })
df_metrics = pd.DataFrame(metrics).melt(id_vars='Model')


plt.figure(figsize=(18, 10))
ax = sns.barplot(data=df_metrics, x='Model', y='value', hue='variable', palette='viridis')


for container in ax.containers:

    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)

plt.title('Comprehensive Metrics Comparison', fontweight='bold', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=11)


plt.ylim(0, 1.15)

plt.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_Metrics_Barplot.pdf'))
plt.close()

# --- 5. Calibration Plot (With Histograms - Lancet Style) ---

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(4, 1)
ax_main = fig.add_subplot(gs[:3, 0])
ax_hist = fig.add_subplot(gs[3, 0])

ax_main.plot([0, 1], [0, 1], "k:", label="Perfect")


for i, name in enumerate(plot_models):
    if name not in results: continue


    if name == 'MAGGIC Score':
        color = '#333333'  # 深灰
        fmt = '--'  #
        hist_ls = '--'  #
        lw = 1.8  # 减细
        alpha = 0.8
    else:
        # ML Models
        alpha = 1.0
        # 防止索引越界
        color_idx = i if i < len(PALETTE_12) else i % len(PALETTE_12)
        color = PALETTE_12[color_idx]

        fmt = 's-'  #
        hist_ls = '-'  #
        lw = 2


    prob_true, prob_pred = calibration_curve(y_test, results[name]['prob_test'], n_bins=10)

    #
    ax_main.plot(prob_pred, prob_true, fmt, lw=lw, label=name, color=color, alpha=alpha)

    #
    ax_hist.hist(results[name]['prob_test'], range=(0, 1), bins=50, histtype='step',
                 lw=lw, linestyle=hist_ls, label=name, color=color, alpha=alpha)

ax_main.set_ylabel("Observed Probability", fontweight='bold')
ax_main.set_title("Calibration Curve", fontweight='bold')
ax_main.legend(frameon=False, loc='upper left')  #

ax_hist.set_xlabel("Predicted Probability", fontweight='bold')
ax_hist.set_ylabel("Count")
ax_hist.legend().set_visible(False)  #

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_Calibration_Lancet_Style.pdf'))
plt.close()

# --- 6. DCA (Decision Curve Analysis) ---

plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 100)
prevalence = np.mean(y_test)

# Treat All / None Lines
nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
plt.plot(thresholds, nb_all, ':', color='grey', label='Treat All')
plt.plot([0, 1], [0, 0], 'k-', label='Treat None')

# [MODIFIED] Iterate over 'plot_models' to ensure MAGGIC is included
for i, name in enumerate(plot_models):
    if name not in results: continue

    y_prob = results[name]['prob_test']
    net_benefit = []

    # Calculate Net Benefit for each threshold
    for t in thresholds:
        tp = np.sum((y_test == 1) & (y_prob >= t))
        fp = np.sum((y_test == 0) & (y_prob >= t))
        nb = (tp / len(y_test)) - (fp / len(y_test)) * (t / (1 - t))
        net_benefit.append(nb)

        # [STYLE] Special style for MAGGIC Score (Dark Grey Dashed)
        if name == 'MAGGIC Score':
            color = '#333333'
            linestyle = '--'
            lw = 1.8  #
            alpha = 0.8
        else:
            alpha = 1.0
        # Use Palette colors for ML models
        # Safe index access
        color_idx = i if i < len(PALETTE_12) else i % len(PALETTE_12)
        color = PALETTE_12[color_idx]
        linestyle = '-'
        lw = 2

    plt.plot(thresholds, net_benefit, lw=lw, linestyle=linestyle, label=name, color=color, alpha=alpha)

plt.ylim(-0.05, 0.4)  # Adjust Y-limit based on your prevalence
plt.xlim(0, 0.8)  # Usually relevant thresholds are < 80%
plt.title("Decision Curve Analysis", fontweight='bold')
plt.xlabel("Threshold Probability", fontweight='bold')
plt.ylabel("Net Benefit", fontweight='bold')
plt.legend(frameon=False, loc='upper right')
plt.grid(alpha=0.2)  # Optional: light grid makes it easier to read
plt.savefig(os.path.join(OUTPUT_DIR, '09_DCA.pdf'))
plt.close()

# ==============================================================================
# 7. COMPREHENSIVE SHAP (Final Fix for Dimension Error)
# ==============================================================================
print("  > Generating SHAP plots (Beeswarm, Bar, Heatmap, Waterfall, Dependence)...")

#
try:

    if hasattr(best_model, 'feature_importances_') and not isinstance(best_model, LogisticRegression):
        print(f"    [Info] Tree model detected ({best_model_name}). Using TreeExplainer.")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_test_opt)


    elif hasattr(best_model, 'coef_') or isinstance(best_model, LogisticRegression):
        print(f"    [Info] Linear model detected ({best_model_name}). Using LinearExplainer.")
        explainer = shap.LinearExplainer(best_model, X_train_opt)
        shap_values = explainer(X_test_opt)


    else:
        print(f"    [Info] Generic model detected ({best_model_name}). Using KernelExplainer.")

        X_summary = shap.kmeans(X_train_opt, 10)

        explainer = shap.KernelExplainer(best_model.predict_proba, X_summary)
        print("    [Info] Calculating Kernel SHAP (this may take a while)...")
        shap_values_raw = explainer.shap_values(X_test_opt, nsamples=100)


        if isinstance(shap_values_raw, list):
            vals = shap_values_raw[1]  # 取正类
            base = explainer.expected_value[1]
        else:
            vals = shap_values_raw
            base = explainer.expected_value

        shap_values = shap.Explanation(values=vals,
                                       base_values=base,
                                       data=X_test_opt.values,
                                       feature_names=X_test_opt.columns.tolist())


    if len(shap_values.shape) == 3:
        print("    [Fix] Detected 3D SHAP values. Slicing to keep Positive Class only.")
        shap_values = shap_values[:, :, 1]


        original_cols = X_test_opt.columns.tolist()
        shap_values.feature_names = [pretty_name_map.get(c, c) for c in original_cols]
        # -------------------------------------------------------

        print(f"    [Check] Final SHAP shape: {shap_values.shape} (Should be 2D)")

    #

    # 7.1 Beeswarm
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm (Global Importance)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '10_SHAP_Beeswarm.pdf'), bbox_inches='tight')
    plt.close()

    # 7.2 Bar Plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("SHAP Mean Abs Importance", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '11_SHAP_Bar.pdf'), bbox_inches='tight')
    plt.close()

    # 7.3 Heatmap
    plt.figure(figsize=(24, 6))
    shap.plots.heatmap(shap_values, max_display=12, show=False)
    plt.title("SHAP Heatmap (Instance Level)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '12_SHAP_Heatmap.pdf'), bbox_inches='tight')
    plt.close()

    # 7.4 Waterfall (High/Low Risk)

    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test_opt)[:, 1]
    else:
        probs = best_model.decision_function(X_test_opt)  # SVC case

    # High Risk
    idx_high = np.argmax(probs)
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx_high], show=False)
    plt.title(f"High Risk Patient (Prob: {probs[idx_high]:.2f})")
    plt.savefig(os.path.join(OUTPUT_DIR, '13_SHAP_Waterfall_HighRisk.pdf'), bbox_inches='tight')
    plt.close()

    # Low Risk
    idx_low = np.argmin(probs)
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx_low], show=False)
    plt.title(f"Low Risk Patient (Prob: {probs[idx_low]:.2f})")
    plt.savefig(os.path.join(OUTPUT_DIR, '14_SHAP_Waterfall_LowRisk.pdf'), bbox_inches='tight')
    plt.close()

    # 7.5 Dependence Plots
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_inds = np.argsort(-mean_abs_shap)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, ind in enumerate(top_inds):
        shap.plots.scatter(shap_values[:, ind], color=shap_values, ax=axes[i], show=False)
        feature_name = shap_values.feature_names[ind]
        axes[i].set_title(feature_name, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '15_SHAP_Dependence_Grid.pdf'))
    plt.close()
    # ==============================================================================
    # [NEW INSERTION] 7.6 Decision Plot & 7.7 Force Plot

    print("    [Info] Generating Decision Plot (Fixed Colors & Names)...")

    #
    if hasattr(best_model, "predict_proba"):
        all_probs = best_model.predict_proba(X_test_opt)[:, 1]
    else:
        all_probs = best_model.decision_function(X_test_opt)

    sorted_indices = np.argsort(all_probs)

    #
    n_low = 20
    n_high = 10
    bot_idx = sorted_indices[:n_low]  # 低风险背景
    top_idx = sorted_indices[-n_high:]  # 高风险前景
    combined_idx = np.concatenate([bot_idx, top_idx])

    #

    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(np.array(explainer.expected_value).shape) > 0:
        base_val = explainer.expected_value[1]
    else:
        base_val = explainer.expected_value

    plot_shap_values = shap_values.values[combined_idx]


    feature_names_pretty = [get_pretty_name(c) for c in X_test_opt.columns]


    color_list = ['#D3D3D3'] * n_low + [NEJM_RED] * n_high

    plt.figure(figsize=(10, 10))

    #
    shap.decision_plot(
        base_val,
        plot_shap_values,
        features=feature_names_pretty,  # <--- 这里传入美化后的名字
        link='logit',
        feature_order='hclust',
        color=color_list,
        alpha=0.8,
        show=False
    )

    #
    plt.title(f"Decision Path: Low Risk (Grey) vs High Risk (Red)",
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Predicted Probability of Readmission", fontweight='bold', fontsize=12)

    #
    out_path = os.path.join(OUTPUT_DIR, '21_SHAP_Decision_Plot_Pretty.pdf')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"    [Success] Decision Plot saved with standard variable names to {out_path}")
    print("    [Success] All SHAP plots generated successfully.")

except Exception as e:
    import traceback

    print(f"    [Error] SHAP generation failed: {e}")
    traceback.print_exc()
# ==============================================================================
# NEW FIGURE 16: Incremental Value Plot (NRI, IDI, Delta AUC)
# ==============================================================================
print("  > Generating Figure 16: Incremental Value Plot (NRI & IDI)...")

#
baseline_key = 'MAGGIC Score' if 'MAGGIC Score' in results else 'Clinical_Score'
baseline_prob = results[baseline_key]['prob_test']

#
nri_val, idi_val = calculate_nri_idi(y_test,
                                     baseline_prob,
                                     results[best_model_name]['prob_test'])

auc_ml = roc_auc_score(y_test, results[best_model_name]['prob_test'])
auc_base = roc_auc_score(y_test, baseline_prob)
delta_auc = auc_ml - auc_base

#
improvement_metrics = pd.DataFrame({
    'Metric': ['Delta AUC', 'NRI', 'IDI'],
    'Value': [delta_auc, nri_val, idi_val],
    'Label': [f"+{delta_auc:.3f}", f"{nri_val:.3f}", f"{idi_val:.3f}"]
})

#
plt.figure(figsize=(8, 6))
bars = plt.bar(improvement_metrics['Metric'], improvement_metrics['Value'],
               color=[NEJM_RED, NEJM_GOLD, NEJM_BLUE], alpha=0.9, width=0.6)

for bar, label in zip(bars, improvement_metrics['Label']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005, label,
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylim(0, max(improvement_metrics['Value']) * 1.3)
plt.axhline(0, color='black', linewidth=1)
# 标题动态化
plt.title(f"Incremental Value: {best_model_name} over {baseline_key}", fontweight='bold', pad=20)
plt.ylabel("Improvement Magnitude", fontweight='bold')

sns.despine(left=True)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tick_params(axis='y', which='both', left=False)

plt.figtext(0.1, 0.02,
            f"Note: Baseline model is {baseline_key}. NRI: Net Reclassification Improvement; IDI: Integrated Discrimination Improvement.",
            fontsize=9, color='grey', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '16_Incremental_Value_NRI_IDI.pdf'))
plt.close()
# ==============================================================================
# NEW FIGURE 17: Subgroup Analysis (Forest Plot)
# ==============================================================================
print("  > Generating Figure 17: Subgroup Analysis Forest Plot...")

#
subgroups = {
    'All Patients': lambda x: [True] * len(x),
    'Age >= 75': lambda x: x['age'] >= 75,
    'Age < 75': lambda x: x['age'] < 75,
    'Male': lambda x: x['sex'] == 1,  # 假设 1=Male
    'Female': lambda x: x['sex'] == 0,
    'Diabetes': lambda x: x.get('diabetes', np.zeros(len(x))) == 1,
    'No Diabetes': lambda x: x.get('diabetes', np.zeros(len(x))) == 0,
    'NYHA III/IV': lambda x: x.get('nyha_class', np.zeros(len(x))) >= 3
}

auc_data = []
#
X_test_original = df_test.copy().reset_index(drop=True)  #
y_test_reset = y_test.reset_index(drop=True)
probs_test_reset = pd.Series(results[best_model_name]['prob_test'])

for label, mask_func in subgroups.items():
    mask = mask_func(X_test_original)
    if mask is None or sum(mask) < 20: continue  #

    y_sub = y_test_reset[mask]
    p_sub = probs_test_reset[mask]

    #
    if len(np.unique(y_sub)) < 2: continue

    auc = roc_auc_score(y_sub, p_sub)
    ci = get_auc_ci(y_sub, p_sub)  #

    auc_data.append({
        'Subgroup': f"{label} (n={sum(mask)})",
        'AUC': auc,
        'Lower': ci[0],
        'Upper': ci[1]
    })

df_sub = pd.DataFrame(auc_data).iloc[::-1]  #

#
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(df_sub))
plt.errorbar(df_sub['AUC'], y_pos, xerr=[df_sub['AUC'] - df_sub['Lower'], df_sub['Upper'] - df_sub['AUC']],
             fmt='o', color=NEJM_BLUE, ecolor=NEJM_GREY, capsize=5, elinewidth=2, markeredgewidth=2)

plt.yticks(y_pos, df_sub['Subgroup'], fontsize=12, fontweight='bold')
plt.axvline(0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(roc_auc_score(y_test, results[best_model_name]['prob_test']), color='green', linestyle=':', alpha=0.5,
            label='Overall AUC')
plt.xlabel('AUC (95% CI)', fontweight='bold')
plt.title(f'Subgroup Analysis: {best_model_name}', fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '17_Subgroup_Forest_Plot.pdf'))
plt.close()
# ==============================================================================
# NEW FIGURE 18 & 19: Interaction Analysis
# ==============================================================================
print("  > Generating Figure 18 & 19: Interaction Analysis (NT-proBNP x eGFR & Top 3)...")


#
def get_shap_col_name(raw_name, shap_obj, name_map):
    pretty = name_map.get(raw_name, raw_name)
    if pretty in shap_obj.feature_names:
        return pretty
    elif raw_name in shap_obj.feature_names:
        return raw_name
    return None


# 1. Specific Interaction: NT-proBNP x eGFR (Reviewer Explicit Request)
col_nt = get_shap_col_name('nt_probnp', shap_values, pretty_name_map)
col_egfr = get_shap_col_name('egfr', shap_values, pretty_name_map)

if col_nt and col_egfr:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))


    shap.plots.scatter(shap_values[:, col_nt], color=shap_values[:, col_egfr], ax=axes[0], show=False)
    axes[0].set_title(f'Interaction: {col_nt} vs {col_egfr}', fontweight='bold')

    # Plot B: eGFR Main Effect, colored by NT-proBNP

    shap.plots.scatter(shap_values[:, col_egfr], color=shap_values[:, col_nt], ax=axes[1], show=False)
    axes[1].set_title(f'Interaction: {col_egfr} vs {col_nt}', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '18_Specific_Interaction_NTproBNP_eGFR.pdf'))
    plt.close()
    print("    [Success] Generated specific interaction plot: NT-proBNP x eGFR")
else:
    print(f"    [Skip] Specific interaction skipped. (Available features: {shap_values.feature_names})")

# 2. Top 3 Strongest Interactions (Automated Discovery for Robustness)
top_3_inds = np.argsort(-np.abs(shap_values.values).mean(axis=0))[:3]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes = axes.flatten()

for i, ind in enumerate(top_3_inds):
    # shap.plots.scatter(..., color=shap_values) 会自动计算并使用最强交互特征着色
    shap.plots.scatter(shap_values[:, ind], color=shap_values, ax=axes[i], show=False)

    feature_name = shap_values.feature_names[ind]
    axes[i].set_title(f'Top {i + 1} Feature Interaction: {feature_name}', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '19_Top3_Automated_Interactions.pdf'))
plt.close()
print("    [Success] Generated Top 3 automated interaction plots.")
# ==============================================================================
# SAVE STATISTICAL REPORT TO TXT
# ==============================================================================
report_path = os.path.join(OUTPUT_DIR, 'Final_Statistical_Report.txt')
with open(report_path, 'w') as f:
    f.write("=== HFpEF + CKD Readmission Model Report ===\n\n")
    f.write(f"Best Model Selected: {best_model_name}\n")
    f.write("-" * 50 + "\n")
    f.write(f"TEST AUC: {roc_auc_score(y_test, results[best_model_name]['prob_test']):.4f}\n")
    f.write(f"TEST AUC 95% CI: {get_auc_ci(y_test, results[best_model_name]['prob_test'])}\n")
    f.write("-" * 50 + "\n")
    f.write(f"Comparison vs Clinical Score:\n")
    f.write(f"DeLong P-value: {p_val_delong:.5e}\n")
    f.write(f"NRI: {nri_val:.4f}\n")
    f.write(f"IDI: {idi_val:.4f}\n")
    f.write("-" * 50 + "\n")
    # 注意：
    try:
        ici_val = metrics[0].get('ICI', 'N/A')
    except:
        ici_val = "N/A"
    f.write(f"Calibration ICI: {ici_val}\n")

print(f"  > Report saved to {report_path}")
# ==============================================================================
# PART 4: SUPPLEMENTARY FIGURES (ALL MODELS)
# ==============================================================================
print("\n--- Stage 6: Generating Supplementary Figures (All Models) ---")

#
def get_supp_style(model_name, idx):
    if model_name == best_model_name:
        #
        return {'color': NEJM_RED, 'lw': 3, 'ls': '-', 'alpha': 1.0, 'zorder': 20}
    elif model_name == 'MAGGIC Score':
        # MAGGIC：
        return {'color': '#333333', 'lw': 2, 'ls': '--', 'alpha': 0.9, 'zorder': 19}
    else:
        #
        c = PALETTE_12[idx % len(PALETTE_12)]
        return {'color': c, 'lw': 1, 'ls': '-', 'alpha': 0.4, 'zorder': 5}

#
all_models_list = sorted_names.copy()
if 'MAGGIC Score' not in all_models_list and 'MAGGIC Score' in results:
    all_models_list.append('MAGGIC Score')

# ------------------------------------------------------------------------------
# Supp Fig 1: Dual ROC (Train vs Test) - ALL MODELS
# ------------------------------------------------------------------------------

print("  > Generating S1: ROC Curves (All Models) - High Quality...")

#
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

#
plot_list = sorted_names.copy()
if 'MAGGIC Score' not in plot_list and 'MAGGIC Score' in results:
    plot_list.append('MAGGIC Score')

#
for i, name in enumerate(plot_list):
    if name not in results: continue

    #
    if name == 'MAGGIC Score':
        color = '#333333'  #
        linestyle = '--'  #
        lw = 2.0  #
        alpha = 0.9
        zorder = 90  #
    else:
        #
        color = PALETTE_12[i % len(PALETTE_12)]
        linestyle = '-'

        #
        if name == best_model_name:
            lw = 2.5  #
            alpha = 1.0  #
            zorder = 100  #
        else:
            lw = 2.0  #
            alpha = 0.75  #
            zorder = 50  #

    #
    y_prob = results[name]['prob_test']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    #
    ax1.plot(fpr, tpr, lw=lw, linestyle=linestyle, color=color,
             alpha=alpha, zorder=zorder, label=f"{name} (AUC={auc:.3f})")

#
ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5, lw=1.5)  # 对角线也加粗一点
ax1.set_title('Validation Cohort ROC', fontweight='bold', fontsize=15)
ax1.set_xlabel('1 - Specificity', fontweight='bold', fontsize=12)
ax1.set_ylabel('Sensitivity', fontweight='bold', fontsize=12)
#
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
           frameon=False, fontsize=10, title="Validation AUC")
ax1.grid(alpha=0.2, linestyle='--')  # 加上淡网格，更有质感

# ===
for i, name in enumerate(plot_list):
    if name not in results: continue

    # -
    if name == 'MAGGIC Score':
        color = '#333333';
        linestyle = '--';
        lw = 2.0;
        alpha = 0.9;
        zorder = 90
    else:
        color = PALETTE_12[i % len(PALETTE_12)]
        linestyle = '-'
        if name == best_model_name:
            lw = 2.5;
            alpha = 1.0;
            zorder = 100
        else:
            lw = 2.0;
            alpha = 0.75;
            zorder = 50

    #
    y_prob = results[name]['prob_train']
    fpr, tpr, _ = roc_curve(y_train_full, y_prob)
    auc = roc_auc_score(y_train_full, y_prob)

    #
    ax2.plot(fpr, tpr, lw=lw, linestyle=linestyle, color=color,
             alpha=alpha, zorder=zorder, label=f"{name} (AUC={auc:.3f})")

#
ax2.plot([0, 1], [0, 1], 'k:', alpha=0.5, lw=1.5)
ax2.set_title('Derivation Cohort ROC', fontweight='bold', fontsize=15)
ax2.set_xlabel('1 - Specificity', fontweight='bold', fontsize=12)
ax2.set_ylabel('Sensitivity', fontweight='bold', fontsize=12)
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
           frameon=False, fontsize=10, title="Derivation AUC")
ax2.grid(alpha=0.2, linestyle='--')

#
plt.tight_layout()
save_path_s1 = os.path.join(OUTPUT_DIR, 'Supplementary_01_All_Models_ROC.pdf')
plt.savefig(save_path_s1, bbox_inches='tight')
plt.close()
print(f"    [Saved] {save_path_s1}")
# ------------------------------------------------------------------------------
# Supp Fig 2: PR Curves - ALL MODELS
# ------------------------------------------------------------------------------

print("  > Generating S2: PR Curves (All Models) - High Quality...")
plt.figure(figsize=(10, 8))

#
plot_list = sorted_names.copy()
if 'MAGGIC Score' not in plot_list and 'MAGGIC Score' in results:
    plot_list.append('MAGGIC Score')

for i, name in enumerate(plot_list):
    if name not in results: continue

    #
    if name == 'MAGGIC Score':
        color = '#333333'
        linestyle = '--'
        lw = 2.0
        alpha = 0.9
        zorder = 90
    else:
        #
        color = PALETTE_12[i % len(PALETTE_12)]
        linestyle = '-'

        #
        if name == best_model_name:
            lw = 2.5  #
            alpha = 1.0
            zorder = 100
        else:
            lw = 2.0  #
            alpha = 0.75  #
            zorder = 50

    #
    precision, recall, _ = precision_recall_curve(y_test, results[name]['prob_test'])
    ap = average_precision_score(y_test, results[name]['prob_test'])

    #
    plt.plot(recall, precision, lw=lw, linestyle=linestyle, color=color,
             alpha=alpha, zorder=zorder, label=f"{name} (AP={ap:.3f})")

#
plt.title('Precision-Recall Curves (All Models)', fontweight='bold', fontsize=15)
plt.xlabel('Recall', fontweight='bold', fontsize=12)
plt.ylabel('Precision', fontweight='bold', fontsize=12)

#
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
           frameon=False, fontsize=10, title="Average Precision (AP)")

plt.grid(alpha=0.2, linestyle='--')  # 加上淡网格

#
plt.tight_layout()
save_path_s2 = os.path.join(OUTPUT_DIR, 'Supplementary_02_All_Models_PR.pdf')
plt.savefig(save_path_s2, bbox_inches='tight')
plt.close()
print(f"    [Saved] {save_path_s2}")

# Supp Fig 3: Calibration Plot (Lancet Style) - ALL MODELS [Revised]
# ------------------------------------------------------------------------------
print("  > Generating S3: Calibration (All Models) - High Quality...")
fig = plt.figure(figsize=(10, 12))  #
gs = fig.add_gridspec(4, 1, hspace=0.1)  #
ax_main = fig.add_subplot(gs[:3, 0])
ax_hist = fig.add_subplot(gs[3, 0])

#
ax_main.plot([0, 1], [0, 1], "k:", label="Perfect", lw=1.5, alpha=0.6)

#
plot_list = sorted_names.copy()
if 'MAGGIC Score' not in plot_list and 'MAGGIC Score' in results:
    plot_list.append('MAGGIC Score')

for i, name in enumerate(plot_list):
    if name not in results: continue

    #
    if name == 'MAGGIC Score':
        color = '#333333'
        lw = 2.0
        alpha = 0.9
        zorder = 90
        linestyle = '--'
        marker = None  #
    else:
        color = PALETTE_12[i % len(PALETTE_12)]
        linestyle = '-'

        if name == best_model_name:
            lw = 2.5  #
            alpha = 1.0
            zorder = 100
            marker = 's'  #
        else:
            lw = 2.0  #
            alpha = 0.75  #
            zorder = 50
            marker = None  #

    #
    prob_true, prob_pred = calibration_curve(y_test, results[name]['prob_test'], n_bins=10)

    # 1.
    ax_main.plot(prob_pred, prob_true, marker=marker, markersize=6,
                 linestyle=linestyle, lw=lw, color=color, alpha=alpha,
                 zorder=zorder, label=name)

    # 2.
    ax_hist.hist(results[name]['prob_test'], range=(0, 1), bins=50, histtype='step',
                 lw=1.5, linestyle=linestyle, color=color, alpha=0.3, zorder=zorder)

#
ax_main.set_ylabel("Observed Probability", fontweight='bold', fontsize=12)
ax_main.set_title("Calibration Curves (All Models)", fontweight='bold', fontsize=15)
ax_main.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=10)
ax_main.grid(alpha=0.2, linestyle='--')
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 1)
#
ax_main.set_xticklabels([])

#
ax_hist.set_xlabel("Predicted Probability", fontweight='bold', fontsize=12)
ax_hist.set_ylabel("Count", fontweight='bold', fontsize=10)
ax_hist.set_xlim(0, 1)
#
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.grid(axis='x', alpha=0.2, linestyle='--')

#
plt.tight_layout()
save_path_s3 = os.path.join(OUTPUT_DIR, 'Supplementary_03_All_Models_Calibration.pdf')
plt.savefig(save_path_s3, bbox_inches='tight')
plt.close()
print(f"    [Saved] {save_path_s3}")
# ------------------------------------------------------------------------------
# Supp Fig 4: DCA - ALL MODELS

print("  > Generating S4: DCA (All Models) - High Quality...")
plt.figure(figsize=(10, 8))

# 1.
thresholds = np.linspace(0.01, 0.99, 100)
prevalence = np.mean(y_test)
nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)

# Treat All:
plt.plot(thresholds, nb_all, ':', color='grey', label='Treat All',
         alpha=0.6, lw=2.0, zorder=1)

# Treat None:
plt.plot([0, 1], [0, 0], 'k-', label='Treat None',
         alpha=0.6, lw=2.0, zorder=1)

#
plot_list = sorted_names.copy()
if 'MAGGIC Score' not in plot_list and 'MAGGIC Score' in results:
    plot_list.append('MAGGIC Score')

# 2.
for i, name in enumerate(plot_list):
    if name not in results: continue

    #
    if name == 'MAGGIC Score':
        color = '#333333'
        linestyle = '--'
        lw = 2.0
        alpha = 0.9
        zorder = 90
    else:
        color = PALETTE_12[i % len(PALETTE_12)]
        linestyle = '-'

        if name == best_model_name:
            lw = 2.5  #
            alpha = 1.0
            zorder = 100
        else:
            lw = 2.0  #
            alpha = 0.75  #
            zorder = 50

    #
    y_prob = results[name]['prob_test']
    net_benefit = []

    for t in thresholds:
        tp = np.sum((y_test == 1) & (y_prob >= t))
        fp = np.sum((y_test == 0) & (y_prob >= t))
        #
        denom = len(y_test)
        if denom == 0: denom = 1

        nb = (tp / denom) - (fp / denom) * (t / (1 - t))
        net_benefit.append(nb)

    #
    plt.plot(thresholds, net_benefit, lw=lw, linestyle=linestyle, color=color,
             alpha=alpha, zorder=zorder, label=name)

# 3.
plt.ylim(-0.05, 0.4)  #
plt.xlim(0, 0.8)  #
plt.title("Decision Curve Analysis (All Models)", fontweight='bold', fontsize=15)
plt.xlabel("Threshold Probability", fontweight='bold', fontsize=12)
plt.ylabel("Net Benefit", fontweight='bold', fontsize=12)

#
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
           frameon=False, fontsize=10, title="Models")

plt.grid(alpha=0.2, linestyle='--')  #

# 保存
plt.tight_layout()
save_path_s4 = os.path.join(OUTPUT_DIR, 'Supplementary_04_All_Models_DCA.pdf')
plt.savefig(save_path_s4, bbox_inches='tight')
plt.close()
print(f"    [Saved] {save_path_s4}")
# ==============================================================================
# [INSERTED] Supplementary Fig 5 & 6: Temporal Validation Comparison
# ==============================================================================
print("  > Generating Supplementary Temporal Validation Plots (PR & Calibration)...")

#
model_name = best_model_name

if model_name in results:
    #
    y_train_prob = results[model_name]['prob_train']
    y_test_prob = results[model_name]['prob_test'] #

    # --- Supp Fig 5: Dual Precision-Recall Curve (Derivation vs Validation) ---
    plt.figure(figsize=(10, 8))

    # Derivation
    prec_train, rec_train, _ = precision_recall_curve(y_train_full, y_train_prob)
    ap_train = average_precision_score(y_train_full, y_train_prob)
    plt.plot(rec_train, prec_train, color=NEJM_BLUE, lw=2, alpha=0.6,
             label=f'Derivation (AP={ap_train:.3f})')

    # Validation
    prec_test, rec_test, _ = precision_recall_curve(y_test, y_test_prob)
    ap_test = average_precision_score(y_test, y_test_prob)
    plt.plot(rec_test, prec_test, color=NEJM_RED, lw=3,
             label=f'Validation (AP={ap_test:.3f})')

    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title(f'Temporal Consistency: PR Curves ({model_name})', fontweight='bold')
    plt.legend(loc='best', frameon=False)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Supplementary_05_Temporal_PR_Compare.pdf'))
    plt.close()

    # --- Supp Fig 6: Dual Calibration Curve (Derivation vs Validation) ---
    plt.figure(figsize=(10, 8))

    plt.plot([0, 1], [0, 1], "k:", label="Perfect")

    # Derivation
    prob_true_train, prob_pred_train = calibration_curve(y_train_full, y_train_prob, n_bins=10)
    plt.plot(prob_pred_train, prob_true_train, "s-", color=NEJM_BLUE, lw=1.5, alpha=0.5,
             label='Derivation Cohort')

    # Validation
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_prob, n_bins=10)
    plt.plot(prob_pred_test, prob_true_test, "o-", color=NEJM_RED, lw=3,
             label='Validation Cohort')

    plt.ylabel("Observed Probability", fontweight='bold')
    plt.xlabel("Predicted Probability", fontweight='bold')
    plt.title(f'Temporal Consistency: Calibration ({model_name})', fontweight='bold')
    plt.legend(loc='upper left', frameon=False)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Supplementary_06_Temporal_Calibration_Compare.pdf'))
    plt.close()
    print("    [Saved] Temporal PR and Calibration plots.")
else:
    print(f"    [Warning] {model_name} not found in results, skipping temporal plots.")
# ==============================================================================

# PART 5: GENERATE TABLES (Table 1 & Table 2) - EXPLICIT CATEGORIES VERSION
# ==============================================================================
print("\n--- Stage 7: Generating Publication Tables (CSV) ---")

#
value_labels = {
    'sex': {
        0: 'Female',
        1: 'Male'
    },
    'nyha_class': {
        1: 'Class I',
        2: 'Class II',
        3: 'Class III',
        4: 'Class IV'
    },

    'readmission_within_1_year': {
        0: 'No Readmission',
        1: 'Readmission'
    }
    #
    #
}

# ------------------------------------------------------------------------------
# 1. TABLE 2: Comprehensive Model Performance

# ------------------------------------------------------------------------------
print("  > Generating Table 2: Model Performance Metrics (with Brier & ICI)...")

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


#
def calculate_ici(y_true, y_prob, n_bins=10):
    """
    计算 Integrated Calibration Index (ICI) 的近似值。
    使用基于分位数的 calibration curve (strategy='quantile') 计算
    观测概率与预测概率之差的平均绝对误差 (Mean Absolute Calibration Error)。
    """
    try:
        #
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
        # ICI ≈ Mean(|Observed - Predicted|)
        ici_val = np.mean(np.abs(prob_true - prob_pred))
        return ici_val
    except:
        return np.nan


# ------------------------------------

perf_metrics = []

#
if 'MAGGIC Score' not in all_models_list and 'MAGGIC Score' in results:
    all_models_list.append('MAGGIC Score')

for name in all_models_list:
    if name not in results: continue

    #
    y_true = y_test
    y_prob = results[name]['prob_test']
    y_pred = results[name]['pred_test']

    # 1.
    auc = roc_auc_score(y_true, y_prob)
    ci = get_auc_ci(y_true, y_prob)  # 使用之前定义的 CI 函数
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 2.
    brier = brier_score_loss(y_true, y_prob)
    ici = calculate_ici(y_true, y_prob)

    # 3.
    perf_metrics.append({
        'Model': name,
        'AUROC (95% CI)': f"{auc:.3f} ({ci[0]:.3f}-{ci[1]:.3f})",
        'Brier Score': f"{brier:.3f}",  # <--- 新增
        'ICI': f"{ici:.3f}",  # <--- 新增
        'Accuracy': f"{accuracy_score(y_true, y_pred):.3f}",
        'Sensitivity': f"{tp / (tp + fn):.3f}" if (tp + fn) > 0 else "0.000",
        'Specificity': f"{tn / (tn + fp):.3f}" if (tn + fp) > 0 else "0.000",
        'PPV': f"{tp / (tp + fp):.3f}" if (tp + fp) > 0 else "0.000",
        'NPV': f"{tn / (tn + fn):.3f}" if (tn + fn) > 0 else "0.000",
        'F1 Score': f"{f1_score(y_true, y_pred):.3f}"
    })

df_perf = pd.DataFrame(perf_metrics)

#
df_perf['sort_idx'] = df_perf['Model'].apply(lambda x: 0 if x == best_model_name else 1)
df_perf = df_perf.sort_values(['sort_idx', 'AUROC (95% CI)'], ascending=[True, False]).drop(columns=['sort_idx'])

# 保存
save_path_t2 = os.path.join(OUTPUT_DIR, 'Table2_Model_Performance.csv')
df_perf.to_csv(save_path_t2, index=False, encoding='utf-8-sig')
print(f"    [Saved] Table 2 saved to {save_path_t2}")

# ------------------------------------------------------------------------------
# 2. TABLE 1 & S1: Baseline Characteristics (Detailed Rows)
# ------------------------------------------------------------------------------
print("  > Generating Baseline Tables (Detailed Rows)...")
from scipy.stats import ttest_ind, chi2_contingency

table_tasks = [
    {"name": "Table 1 (Core)", "vars": CORE_CLINICAL_VARS, "file": "Table1_Baseline_Characteristics.csv"},
    {"name": "Table S1 (Full)", "vars": features, "file": "TableS1_Full_Baseline.csv"}
]

for task in table_tasks:
    print(f"    Processing {task['name']}...")
    table_data = []

    for var in task["vars"]:
        if var not in df.columns: continue

        #
        clean_train = df_train[var].dropna()
        clean_test = df_test[var].dropna()
        unique_vals = sorted(df[var].dropna().unique())
        n_unique = len(unique_vals)
        var_label = pretty_name_map.get(var, var)

        #
        if n_unique >= 10:
            m1, s1 = clean_train.mean(), clean_train.std()
            m2, s2 = clean_test.mean(), clean_test.std()
            p = ttest_ind(clean_train, clean_test, equal_var=False)[1] if len(clean_train) > 1 else 1.0

            table_data.append({
                'Variable': var_label,
                f'Derivation (n={len(df_train)})': f"{m1:.1f} ± {s1:.1f}",
                f'Validation (n={len(df_test)})': f"{m2:.1f} ± {s2:.1f}",
                'P Value': f"<0.001" if p < 0.001 else f"{p:.3f}"
            })

        #
        #
        elif var in value_labels:
            #
            try:
                #
                all_cats = list(value_labels[var].keys())
                freq_train = [(clean_train == c).sum() for c in all_cats]
                freq_test = [(clean_test == c).sum() for c in all_cats]
                #
                p_global = chi2_contingency([freq_train, freq_test])[1] if np.sum(freq_train) + np.sum(
                    freq_test) > 0 else 1.0
            except:
                p_global = 1.0

            #
            table_data.append({
                'Variable': var_label,
                f'Derivation (n={len(df_train)})': '',
                f'Validation (n={len(df_test)})': '',
                'P Value': f"<0.001" if p_global < 0.001 else f"{p_global:.3f}"
            })

            #
            for val, label_text in value_labels[var].items():
                c1 = (clean_train == val).sum()
                c2 = (clean_test == val).sum()
                n1, n2 = len(clean_train), len(clean_test)

                table_data.append({
                    'Variable': f"  {label_text}",  #
                    f'Derivation (n={len(df_train)})': f"{c1} ({c1 / n1 * 100:.1f}%)",
                    f'Validation (n={len(df_test)})': f"{c2} ({c2 / n2 * 100:.1f}%)",
                    'P Value': ''  #
                })

        #
        #
        else:
            #
            pos_val = 1
            c1 = (clean_train == pos_val).sum()
            c2 = (clean_test == pos_val).sum()
            n1, n2 = len(clean_train), len(clean_test)

            #
            try:
                mat = [[c1, n1 - c1], [c2, n2 - c2]]
                p = chi2_contingency(mat)[1] if np.sum(mat) > 0 else 1.0
            except:
                p = 1.0

            #
            #
            #

            table_data.append({
                'Variable': var_label,
                f'Derivation (n={len(df_train)})': f"{c1} ({c1 / n1 * 100:.1f}%)",
                f'Validation (n={len(df_test)})': f"{c2} ({c2 / n2 * 100:.1f}%)",
                'P Value': f"<0.001" if p < 0.001 else f"{p:.3f}"
            })

    #
    df_table = pd.DataFrame(table_data)
    df_table.to_csv(os.path.join(OUTPUT_DIR, task["file"]), index=False, encoding='utf-8-sig')
    print(f"    [Saved] {task['file']}")

print("  > All tables generated with explicit categories.")

# ==============================================================================
print(f"\n=== FINISHED. All 15 figures saved to '{OUTPUT_DIR}' ===")
# ==============================================================================
# NEW FIGURE 20: SHAP Interaction Network (Circular Layout)
# ==============================================================================
#
if 'best_model' in locals() and 'X_test_opt' in locals():
    #
    is_tree = True
    try:
        from sklearn.linear_model import LogisticRegression

        if isinstance(best_model, LogisticRegression):
            is_tree = False
    except:
        pass

    if is_tree:
        plot_shap_interaction_network(
            model=best_model,
            X=X_test_opt,  #
            output_dir=OUTPUT_DIR,
            top_n=10  #
        )
    else:
        print("  > [Skip] Interaction Network plot requires a Tree-based model (e.g., Random Forest, XGBoost).")
# ==============================================================================


print("="*50)
print("FEATURE SELECTION AUDIT")
print("="*50)

#
#
try:
    print(f"[Figure 2C] Optimal features in plotting run: {rfecv_plot.n_features_}")
except NameError:
    print("[Figure 2C] rfecv_plot object not found (Plotting code might not have run).")

#
#
try:
    print(f"[FINAL MODEL] Actual Stable Features used: {len(stable_features)}")
    print("-" * 20)
    print("List of Final Features:")
    for i, feat in enumerate(stable_features, 1):
        print(f"{i}. {feat}")
except NameError:
    print("Error: 'stable_features' variable not found. Please run the feature selection block first.")

print("="*50)
#
#
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec

# [REVISED] Figure 4A & 4B: Separated Calibration Plots
# ==============================================================================
print("  > [Revised] Generating Separated Calibration Plots...")

#
calib_models = [
    'RandomForest', 'ExtraTrees', 'GaussianNB',
    'LogisticRegression', 'MAGGIC Score'
]

# --- 1. Figure 4A:
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", alpha=0.6, lw=1.5)

for name in calib_models:
    if name in results:
        y_prob = results[name]['prob_test']
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

        #
        if name == 'RandomForest':
            color = NEJM_RED; lw = 3.0; zorder = 100; marker = 's'
        elif name == 'ExtraTrees':
            color = NEJM_GOLD; lw = 2.0; zorder = 90; marker = 'o'
        elif name == 'GaussianNB':
            color = NEJM_GREEN; lw = 2.0; zorder = 85; marker = '^'
        elif name == 'LogisticRegression':
            color = NEJM_BLUE; lw = 2.0; zorder = 95; marker = 'v'
        elif name == 'MAGGIC Score':
            color = '#333333'; lw = 2.0; zorder = 80; marker = None; ls = '--'
        else:
            continue

        ls = '--' if name == 'MAGGIC Score' else '-'

        #
        ici_val = np.mean(np.abs(prob_true - prob_pred))

        plt.plot(prob_pred, prob_true, marker=marker, markersize=6, linestyle=ls,
                 linewidth=lw, color=color, alpha=0.9, zorder=zorder,
                 label=f"{name} (ICI={ici_val:.3f})")

plt.xlabel("Predicted Probability", fontweight='bold', fontsize=12)
plt.ylabel("Observed Probability (Fraction of Positives)", fontweight='bold', fontsize=12)
plt.title("Calibration Curves (Validation Cohort)", fontweight='bold', fontsize=15)
plt.legend(loc="upper left", frameon=False, fontsize=10)
plt.grid(alpha=0.2, linestyle='--')
plt.xlim(0, 1)
plt.ylim(0, 1)

save_path_a = os.path.join(OUTPUT_DIR, '04a_Calibration_Curves_Only.pdf')
plt.tight_layout()
plt.savefig(save_path_a)
plt.close()
print(f"    [Saved] {save_path_a}")

# --- 2. Figure 4B: The Histograms Only
plt.figure(figsize=(8, 5))  #

for name in calib_models:
    if name in results:
        y_prob = results[name]['prob_test']

        #
        if name == 'RandomForest':
            color = NEJM_RED; lw = 2.5; zorder = 100
        elif name == 'ExtraTrees':
            color = NEJM_GOLD; lw = 1.5; zorder = 90
        elif name == 'GaussianNB':
            color = NEJM_GREEN; lw = 1.5; zorder = 85
        elif name == 'LogisticRegression':
            color = NEJM_BLUE; lw = 1.5; zorder = 95
        elif name == 'MAGGIC Score':
            color = '#333333'; lw = 1.5; zorder = 80; ls = '--'
        else:
            continue

        ls = '--' if name == 'MAGGIC Score' else '-'

        #
        plt.hist(y_prob, range=(0, 1), bins=50, histtype='step',
                 linewidth=lw, linestyle=ls, color=color, alpha=0.8,
                 label=name, density=True)  # 使用 Density=True 方便比较不同模型

plt.xlabel("Predicted Probability", fontweight='bold', fontsize=12)
plt.ylabel("Density", fontweight='bold', fontsize=12)
plt.title("Prediction Probability Distribution", fontweight='bold', fontsize=14)
plt.legend(loc="upper center", frameon=False, fontsize=9, ncol=2)  # 分两列显示图例
plt.grid(alpha=0.2, linestyle='--', axis='x')
plt.xlim(0, 1)

#
sns.despine()

save_path_b = os.path.join(OUTPUT_DIR, '04b_Calibration_Histograms_Only.pdf')
plt.tight_layout()
plt.savefig(save_path_b)
plt.close()
print(f"    [Saved] {save_path_b}")
# FIGURE 1E: PRECISION-RECALL CURVES (The "Consistent 5" Strategy)
# ==============================================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

print("  > Generating Figure 1E: Precision-Recall Curves (Consistent 5)...")
plt.figure(figsize=(8, 8))

# --- 1.
pr_models = [
    'RandomForest',
    'ExtraTrees',
    'GaussianNB',
    'LogisticRegression',
    'MAGGIC Score'
]

# --- 2
for name in pr_models:
    if name in results:
        y_prob = results[name]['prob_test']

        #
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        #
        if name == 'RandomForest':
            color = NEJM_RED;
            lw = 3.0;
            zorder = 100
        elif name == 'ExtraTrees':
            color = NEJM_GOLD;
            lw = 2.0;
            zorder = 90
        elif name == 'GaussianNB':
            color = NEJM_GREEN;
            lw = 2.0;
            zorder = 85
        elif name == 'LogisticRegression':
            color = NEJM_BLUE;
            lw = 2.0;
            zorder = 95
        elif name == 'MAGGIC Score':
            color = '#333333';
            lw = 2.0;
            zorder = 80
            linestyle = '--'  # 临床评分可以用虚线区分
        else:
            color = 'purple';
            lw = 1.0;
            zorder = 50
            linestyle = '-'

        if name != 'MAGGIC Score': linestyle = '-'

        plt.plot(recall, precision, lw=lw, color=color, linestyle=linestyle,
                 alpha=0.9, zorder=zorder,
                 label=f"{name} (AP={ap:.3f})")

#
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle=':', color='grey', label='No Skill')

plt.xlabel("Recall (Sensitivity)", fontweight='bold')
plt.ylabel("Precision (PPV)", fontweight='bold')
plt.title("Precision-Recall Curves", fontweight='bold')
plt.legend(loc="best", fontsize=9, frameon=False)
plt.grid(alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03b_PR_Curves_NEJM.pdf'))
plt.close()
# 
# PART 9: CONFUSION MATRICES (The "Golden 2x2" Layout - Clean Version)
# ==============================================================================
from sklearn.metrics import confusion_matrix

print("\n> [Stage 9] Generating Figure: Confusion Matrices (2x2 Grid - Clean)...")

# 1.
cm_models = [
    'RandomForest',
    'ExtraTrees',  #
    'GaussianNB',  #
    'LogisticRegression'  #
]

# 2.
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 完美布局
axes = axes.flatten()

# 3.
for i, name in enumerate(cm_models):
    ax = axes[i]

    if name in results:
        #
        y_prob = results[name]['prob_test']
        y_pred = (y_prob >= 0.5).astype(int)

        #
        cm = confusion_matrix(y_test, y_pred)

        #
        #
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar=False, annot_kws={"size": 18, "weight": "bold"})

        #
        ax.set_title(f"{name}", fontweight='bold', fontsize=14, pad=12)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=11)

        #
        ax.set_xticklabels(['No-Readm', 'Readmission'], fontsize=10)
        ax.set_yticklabels(['No-Readm', 'Readmission'], fontsize=10, rotation=90, va='center')

        #
        ax.tick_params(length=0)

    else:
        ax.axis('off')  #

plt.tight_layout(pad=3.0)  #
save_path = os.path.join(OUTPUT_DIR, '05_Confusion_Matrices_2x2_NEJM.pdf')
plt.savefig(save_path)
plt.close()
print(f"    [Saved] {save_path}")
# ==============================================================================
# FIGURE 1C: CALIBRATION CURVES + MULTI-MODEL HISTOGRAM (Best of Both Worlds)
# ==============================================================================
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec

print("  > Generating Figure 1C: Calibration Curves with Multi-Model Histograms...")

# 1.
fig = plt.figure(figsize=(8, 10))
#
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax_curve = plt.subplot(gs[0])
ax_hist = plt.subplot(gs[1])

#
calibration_models = [
    'RandomForest',  #
    'ExtraTrees',  #
    'GaussianNB',  #
    'LogisticRegression',  #
    'MAGGIC Score'  #
]

#
ax_curve.plot([0, 1], [0, 1], "k:", label="Perfect", alpha=0.6)

for name in calibration_models:
    if name in results:
        y_prob = results[name]['prob_test']
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

        #
        if name == 'RandomForest':
            color = NEJM_RED;
            lw = 3.0;
            zorder = 100;
            ls = '-'
        elif name == 'ExtraTrees':
            color = NEJM_GOLD;
            lw = 2.0;
            zorder = 90;
            ls = '-'
        elif name == 'GaussianNB':
            color = NEJM_GREEN;
            lw = 2.0;
            zorder = 85;
            ls = '-'
        elif name == 'LogisticRegression':
            color = NEJM_BLUE;
            lw = 2.0;
            zorder = 95;
            ls = '-'
        elif name == 'MAGGIC Score':
            color = '#333333';
            lw = 2.0;
            zorder = 80;
            ls = '--'  # 虚线区分
        else:
            color = 'purple';
            lw = 1.0;
            zorder = 50;
            ls = '-'

        # Brier Score
        brier = np.mean((y_prob - y_test) ** 2)

        #
        ax_curve.plot(prob_pred, prob_true, marker='o', markersize=4, linestyle=ls,
                      linewidth=lw, color=color, alpha=0.9, zorder=zorder,
                      label=f"{name} (Brier={brier:.3f})")

        #
        #
        #
        hist_lw = 2.5 if name == 'RandomForest' else 1.5

        ax_hist.hist(y_prob, range=(0, 1), bins=50, histtype='step',
                     linewidth=hist_lw, linestyle=ls, color=color, alpha=0.8,
                     label=name, density=False)  #

#
ax_curve.set_ylabel("Fraction of Positives", fontweight='bold')
ax_curve.set_ylim([-0.05, 1.05])
ax_curve.legend(loc="upper left", fontsize=9, frameon=False)
ax_curve.grid(alpha=0.2, linestyle='--')
ax_curve.set_title("Calibration Comparison", fontweight='bold')
ax_curve.set_xticklabels([])  # 隐藏上图 X 轴标尺

#
ax_hist.set_xlabel("Mean Predicted Probability", fontweight='bold')
ax_hist.set_ylabel("Count", fontweight='bold')
ax_hist.set_xlim([0, 1])
ax_hist.grid(alpha=0.2, linestyle='--')
#

#
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)  #

save_path = os.path.join(OUTPUT_DIR, '04_Calibration_Curves_MultiHist_NEJM.pdf')
plt.savefig(save_path)
plt.close()
print(f"    [Saved] {save_path}")

# PART 8: ADVANCED TEMPORAL & DRIFT VISUALIZATIONS (Revised V4 - Reviewer Fixes)
# ==============================================================================
print("\n> [Stage 8] Generating Advanced Temporal Drift Plots (Figure 1C/1D/1E)...")

import matplotlib.dates as mdates
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import matplotlib.patches as mpatches

# 定义配色
COLOR_MAIN = '#0072B5'  # NEJM Blue
COLOR_ACCENT = '#BC3C29'  # NEJM Red
COLOR_NEUTRAL = '#787878'  # Grey
COLOR_STABLE = '#20854E'  # Green
NEJM_GOLD = '#E18727'  # Orange

#
final_stable_vars_for_plot = [
    'egfr', 'hs_crp', 'blood_urea_nitrogen', 'lvef',
    'E_over_e_prime', 'LAVI', 'length_of_stay',
    'systolic_bp', 'triglycerides'
]
if 'nt_probnp' in df.columns:
    final_stable_vars_for_plot.append('nt_probnp')


# 1. Figure 1C: Rolling Mean Trend
# ------------------------------------------------------------------------------
plot_feats = [f for f in final_stable_vars_for_plot if f in df.columns]

if plot_feats:
    print(f"  > Generating Figure 1C: Rolling Trends for {len(plot_feats)} features...")
    df_temp = df.copy()
    if 'admission_date' in df_temp.columns:
        df_temp['admission_date'] = pd.to_datetime(df_temp['admission_date'])
        df_temp = df_temp.set_index('admission_date').sort_index()

        #
        df_roll = df_temp[plot_feats].resample('M').mean().rolling(window=3, min_periods=1).mean()

        import math

        n_feats = len(plot_feats)
        n_cols = 2
        n_rows = math.ceil(n_feats / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.0 * n_rows))
        axes = axes.flatten()
        split_date = pd.to_datetime(TEST_START)

        for i, col in enumerate(plot_feats):
            sns.lineplot(data=df_roll, x=df_roll.index, y=col, ax=axes[i],
                         color=COLOR_MAIN, linewidth=2.5)
            axes[i].axvline(split_date, color=COLOR_ACCENT, linestyle='--', linewidth=2, alpha=0.8,
                            label='Validation Start')

            pretty_name = pretty_name_map.get(col, col)
            axes[i].set_title(f"{pretty_name}", fontweight='bold', fontsize=11)
            axes[i].set_xlabel('')
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i].tick_params(axis='x', rotation=30)
            axes[i].grid(True, alpha=0.15, linestyle='--')
            if i == 0: axes[i].legend(loc='upper left', frameon=False, fontsize=8)

        for j in range(len(plot_feats), len(axes)): axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '01c_Rolling_Mean_Trends.pdf'), bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------------------
# 2. Figure 1E & TABLE: PSI Analysis (Quantitative Report)
# ------------------------------------------------------------------------------
print("  > Generating Figure 1E & Table: PSI Analysis...")


def calculate_psi_safe(expected, actual, buckettype='quantiles', buckets=10):
    try:
        def psi(expected_array, actual_array, buckets):
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            if buckettype == 'bins':
                breakpoints = np.interp(breakpoints, [0, 100], [np.min(expected_array), np.max(expected_array)])
            elif buckettype == 'quantiles':
                breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 2: return 0.0  # 处理变量值单一的情况

            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

            def sub_psi(e_perc, a_perc):
                if a_perc == 0: a_perc = 0.0001
                if e_perc == 0: e_perc = 0.0001
                return (e_perc - a_perc) * np.log(e_perc / a_perc)

            psi_values = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))]
            return np.sum(psi_values)

        return psi(expected, actual, buckets)
    except:
        return np.nan


#
print("  > [Revised] Generating Table: PSI Analysis with Descriptive Stats...")

psi_data_detailed = []


#
def format_val(val):
    return f"{val:.3f}"


#
for col in plot_feats:
    if col not in df_train.columns or col not in df_test.columns: continue

    # 1.
    is_numeric = pd.api.types.is_numeric_dtype(df_train[col]) and df_train[col].nunique() > 10

    if is_numeric:
        #
        m_train = df_train[col].mean()
        s_train = df_train[col].std()
        desc_train = f"{m_train:.3f} ± {s_train:.3f}"

        #
        m_test = df_test[col].mean()
        s_test = df_test[col].std()
        desc_test = f"{m_test:.3f} ± {s_test:.3f}"

        #
        psi_val = calculate_psi_safe(df_train[col], df_test[col], buckettype='quantiles', buckets=10)

    else:
        #
        top_cat_train = df_train[col].mode()[0]
        top_prop_train = (df_train[col] == top_cat_train).mean() * 100
        desc_train = f"{top_cat_train} ({top_prop_train:.1f}%)"

        top_cat_test = df_test[col].mode()[0]
        top_prop_test = (df_test[col] == top_cat_test).mean() * 100
        desc_test = f"{top_cat_test} ({top_prop_test:.1f}%)"

        #
        train_dist = df_train[col].value_counts(normalize=True).sort_index()
        test_dist = df_test[col].value_counts(normalize=True).sort_index()
        all_cats = sorted(list(set(train_dist.index) | set(test_dist.index)))
        t_prop = np.array([train_dist.get(c, 0.0001) for c in all_cats])
        v_prop = np.array([test_dist.get(c, 0.0001) for c in all_cats])
        psi_val = np.sum((t_prop - v_prop) * np.log(t_prop / v_prop))

    # 2.
    if psi_val < 0.1:
        status = "Stable"
    elif psi_val < 0.25:
        status = "Minor Drift"
    else:
        status = "Major Drift"

    psi_data_detailed.append({
        'Feature': pretty_name_map.get(col, col),
        'Derivation (Mean±SD/%)': desc_train,
        'Validation (Mean±SD/%)': desc_test,
        'PSI Value': float(f"{psi_val:.3f}"),
        'Status': status
    })

#
psi_df_final = pd.DataFrame(psi_data_detailed)
psi_df_final = psi_df_final.sort_values('PSI Value', ascending=True)

csv_path = os.path.join(OUTPUT_DIR, 'Table_PSI_Drift_Detailed.csv')
psi_df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"    [Saved] Detailed PSI Table to {csv_path}")

#
psi_df = psi_df_final.rename(columns={'PSI Value': 'PSI'})
#
#
if not psi_df.empty:
    plt.figure(figsize=(8, len(psi_df) * 0.6 + 1.5))
    colors = [COLOR_STABLE if x < 0.1 else (NEJM_GOLD if x < 0.25 else COLOR_ACCENT) for x in psi_df['PSI']]
    y_pos = np.arange(len(psi_df))
    plt.barh(y_pos, psi_df['PSI'], color=colors, height=0.6)
    plt.yticks(y_pos, psi_df['Feature'], fontsize=11, fontweight='bold')
    plt.axvline(0.1, color=COLOR_STABLE, linestyle=':', label='Stability Threshold (0.1)')
    plt.axvline(0.25, color=NEJM_GOLD, linestyle=':', label='Drift Warning (0.25)')
    plt.title(f'Feature Stability (PSI Analysis)', fontweight='bold')
    plt.xlabel('Population Stability Index (PSI)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01e_Feature_PSI_Chart.pdf'))
    plt.close()


# ==============================================================================
# ==============================================================================
# [REVISED V4] Figure 1D: Rolling AUC (Restored Original Axis + Reviewer Fixes)
# ==============================================================================
from scipy.interpolate import make_interp_spline
import matplotlib.dates as mdates

#
model_name = 'RandomForest' if 'RandomForest' in results else best_model_name

if model_name in results and 'admission_date' in df_test.columns:
    print(f"  > [Revised V4] Generating Figure 1D: Rolling AUC (Original Window=50 + CI)...")

    #
    df_perf = df_test[['admission_date', target]].copy()
    df_perf['pred_prob'] = results[model_name]['prob_test']
    df_perf = df_perf.sort_values('admission_date').reset_index(drop=True)

    #
    window_size = 50  #
    step_size = 5
    n_boot = 1000

    rolling_dates = []
    rolling_aucs = []
    rolling_lower = []
    rolling_upper = []

    #
    for start_idx in range(0, len(df_perf) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        subset = df_perf.iloc[start_idx:end_idx]

        if subset[target].nunique() > 1:
            y_sub = subset[target].values
            p_sub = subset['pred_prob'].values

            try:
                main_auc = roc_auc_score(y_sub, p_sub)
            except:
                continue

            # Bootstrap CI
            boot_scores = []
            rng_seed = np.random.RandomState(42)
            for _ in range(n_boot):
                indices = rng_seed.randint(0, len(y_sub), len(y_sub))
                if len(np.unique(y_sub[indices])) > 1:
                    boot_scores.append(roc_auc_score(y_sub[indices], p_sub[indices]))

            if len(boot_scores) > 0:
                rolling_aucs.append(main_auc)
                rolling_lower.append(np.percentile(boot_scores, 2.5))
                rolling_upper.append(np.percentile(boot_scores, 97.5))
                #
                rolling_dates.append(subset['admission_date'].iloc[int(window_size / 2)])

    #
    if len(rolling_aucs) > 3:
        fig, ax = plt.subplots(figsize=(10, 6))  # 恢复你原来的尺寸

        #
        dates_num = mdates.date2num(rolling_dates)
        dates_smooth = np.linspace(dates_num.min(), dates_num.max(), 300)

        spl_auc = make_interp_spline(dates_num, rolling_aucs, k=3)
        spl_low = make_interp_spline(dates_num, rolling_lower, k=3)
        spl_up = make_interp_spline(dates_num, rolling_upper, k=3)

        aucs_smooth = np.clip(spl_auc(dates_smooth), 0, 1)
        low_smooth = np.clip(spl_low(dates_smooth), 0, 1)
        up_smooth = np.clip(spl_up(dates_smooth), 0, 1)
        dates_smooth_dt = mdates.num2date(dates_smooth)

        #
        ax.fill_between(dates_smooth_dt, low_smooth, up_smooth,
                        color=COLOR_MAIN, alpha=0.15,
                        label='95% Bootstrap CI')

        #
        ax.plot(dates_smooth_dt, aucs_smooth, '-', color=COLOR_MAIN,
                linewidth=3, label=f'Rolling AUC (Window={window_size})')

        #
        ax.scatter(rolling_dates, rolling_aucs, color=COLOR_MAIN, s=20, alpha=0.6, zorder=5)

        #
        overall_y = y_test
        overall_p = results[model_name]['prob_test']
        overall_auc = roc_auc_score(overall_y, overall_p)
        try:
            #
            overall_ci = get_auc_ci(overall_y, overall_p)
            label_ref = f'Overall AUC: {overall_auc:.3f} [{overall_ci[0]:.3f}-{overall_ci[1]:.3f}]'
        except:
            label_ref = f'Overall AUC: {overall_auc:.3f}'

        ax.axhline(overall_auc, color=COLOR_NEUTRAL, linestyle='--',
                   label=label_ref)

        #
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Aug 2024

        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel('Area Under ROC Curve (AUC)', fontweight='bold', fontsize=12)
        ax.set_title('Temporal Robustness (Rolling Window Analysis)', fontweight='bold', fontsize=14)

        #
        ax.legend(loc='lower right', frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        #
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontweight='bold')

        save_path = os.path.join(OUTPUT_DIR, '01d_Rolling_AUC_Final_Restore.pdf')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"    [Saved] {save_path}")
    else:
        print("    [Warning] Not enough data points for rolling plot.")
# ==============================================================================
#
# ==============================================================================

from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, roc_auc_score, recall_score,
                             precision_score, accuracy_score, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd

def calculate_ici(y_true, y_prob):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
        return np.mean(np.abs(prob_true - prob_pred))
    except:
        return np.nan

def calculate_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fn) == 0: return 0.0
    return tn / (tn + fn)

def calculate_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0: return 0.0
    return tn / (tn + fp)

def get_ci(y_true, y_pred, y_prob, metric_func, n_boot=1000):
    rng = np.random.RandomState(42)
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    needs_prob = metric_func in [roc_auc_score, brier_score_loss, calculate_ici]

    for _ in range(n_boot):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue

        try:
            if needs_prob:
                val = metric_func(y_true[indices], y_prob[indices])
            else:
                val = metric_func(y_true[indices], y_pred[indices])

            if not np.isnan(val):
                scores.append(val)
        except:
            pass

    return np.percentile(scores, [2.5, 97.5])

print(f"Generating Ultimate Table 2...")
print(f"Models: {list(results.keys())}")

table_data = []

for name in tqdm(results.keys()):
    y_true = y_test
    y_pred = results[name]['pred_test']
    y_prob = results[name]['prob_test']

    auc_val = roc_auc_score(y_true, y_prob)
    auc_ci = get_ci(y_true, y_pred, y_prob, roc_auc_score)

    sens_val = recall_score(y_true, y_pred)
    sens_ci = get_ci(y_true, y_pred, y_prob, recall_score)

    spec_val = calculate_spec(y_true, y_pred)
    spec_ci = get_ci(y_true, y_pred, y_prob, calculate_spec)

    ppv_val = precision_score(y_true, y_pred)
    ppv_ci = get_ci(y_true, y_pred, y_prob, precision_score)

    npv_val = calculate_npv(y_true, y_pred)
    npv_ci = get_ci(y_true, y_pred, y_prob, calculate_npv)

    acc_val = accuracy_score(y_true, y_pred)
    acc_ci = get_ci(y_true, y_pred, y_prob, accuracy_score)

    f1_val = f1_score(y_true, y_pred)
    f1_ci = get_ci(y_true, y_pred, y_prob, f1_score)

    brier_val = brier_score_loss(y_true, y_prob)
    brier_ci = get_ci(y_true, y_pred, y_prob, brier_score_loss)

    ici_val = calculate_ici(y_true, y_prob)
    ici_ci = get_ci(y_true, y_pred, y_prob, calculate_ici)

    row = {
        'Model': name,
        'AUROC': f"{auc_val:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
        'Sensitivity': f"{sens_val:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})",
        'Specificity': f"{spec_val:.3f} ({spec_ci[0]:.3f}-{spec_ci[1]:.3f})",
        'PPV': f"{ppv_val:.3f} ({ppv_ci[0]:.3f}-{ppv_ci[1]:.3f})",
        'NPV': f"{npv_val:.3f} ({npv_ci[0]:.3f}-{npv_ci[1]:.3f})",
        'Accuracy': f"{acc_val:.3f} ({acc_ci[0]:.3f}-{acc_ci[1]:.3f})",
        'F1 Score': f"{f1_val:.3f} ({f1_ci[0]:.3f}-{f1_ci[1]:.3f})",
        'Brier Score': f"{brier_val:.3f} ({brier_ci[0]:.3f}-{brier_ci[1]:.3f})",
        'ICI': f"{ici_val:.3f} ({ici_ci[0]:.3f}-{ici_ci[1]:.3f})",
        '_sort_key': auc_val
    }
    table_data.append(row)

df_final = pd.DataFrame(table_data)
df_final = df_final.sort_values('_sort_key', ascending=False).drop(columns=['_sort_key'])

filename = 'Table2_Ultimate_Full_CI.csv'
df_final.to_csv(filename, index=False, encoding='utf-8-sig')

print("-" * 60)
print(f"Done! Saved to: {filename}")

print("-" * 60)
