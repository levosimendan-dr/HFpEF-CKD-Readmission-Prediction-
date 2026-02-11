import pandas as pd
import numpy as np
import os

#
np.random.seed(42)


def generate_dummy_data(filename='dummy_data.csv', n_samples=1000):
    print(f"Generating comprehensive synthetic data: {filename} ...")

    #
    dates = pd.date_range(start='2022-01-01', end='2024-10-15', freq='D')
    sim_dates = np.random.choice(dates, n_samples, replace=True)

    #
    df = pd.DataFrame({
        #
        'admission_date': sim_dates,
        'readmission_within_1_year': np.random.randint(0, 2, n_samples),

        # --- 人口学特征 (Demographics) ---
        'age': np.random.normal(70, 12, n_samples).astype(int),
        'sex': np.random.randint(0, 2, n_samples),
        'bmi': np.random.normal(26, 5, n_samples),
        'smoker': np.random.randint(0, 2, n_samples),
        'obesity': np.random.randint(0, 2, n_samples),  # 虽然可能和BMI相关，但为了防报错直接生成
        'length_of_stay': np.random.randint(2, 20, n_samples),

        # --- 临床体征 (Vitals & Status) ---
        'systolic_bp': np.random.normal(130, 20, n_samples),
        'diastolic_bp': np.random.normal(80, 10, n_samples),
        'heart_rate': np.random.normal(75, 15, n_samples),
        'nyha_class': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.4, 0.1]),

        # --- 并发症 (Comorbidities - Binary) ---
        'diabetes': np.random.randint(0, 2, n_samples),
        'hypertension': np.random.randint(0, 2, n_samples),
        'atrial_fibrillation': np.random.randint(0, 2, n_samples),
        'copd': np.random.randint(0, 2, n_samples),
        'coronary_artery_disease': np.random.randint(0, 2, n_samples),
        'anemia': np.random.randint(0, 2, n_samples),
        'history_of_aki': np.random.randint(0, 2, n_samples),

        # --- 用药情况 (Medications - Binary) ---
        'raas_use': np.random.randint(0, 2, n_samples),
        'beta_blocker_use': np.random.randint(0, 2, n_samples),
        'sglt2i_use': np.random.randint(0, 2, n_samples),
        'mra_use': np.random.randint(0, 2, n_samples),

        # --- 超声心动图 (Echocardiography) ---
        'lvef': np.random.normal(55, 10, n_samples),  # HFpEF通常>50
        'E_over_e_prime': np.random.normal(12, 4, n_samples),
        'TR_velocity': np.random.normal(2.8, 0.6, n_samples),
        'LAVI': np.random.normal(40, 12, n_samples),
        'LVMI': np.random.normal(110, 25, n_samples),

        # --- 实验室指标 (Labs - Renal & Metabolic) ---
        'egfr': np.random.normal(50, 20, n_samples),
        'serum_creatinine': np.random.lognormal(0.2, 0.5, n_samples),
        'blood_urea_nitrogen': np.random.normal(20, 8, n_samples),
        'cystatin_c': np.random.normal(1.2, 0.5, n_samples),
        'serum_uric_acid': np.random.normal(350, 80, n_samples),
        'proteinuria_quantification': np.random.lognormal(5, 1, n_samples),  # 随机生成

        # --- 实验室指标 (Labs - Inflammatory & Others) ---
        'nt_probnp': np.random.lognormal(7, 1.5, n_samples),  # 对数正态分布更像真实BNP
        'hs_crp': np.random.lognormal(1.5, 1, n_samples),
        'il_6': np.random.lognormal(2, 1, n_samples),
        'd_dimer': np.random.lognormal(6, 1, n_samples),
        'homocysteine': np.random.normal(15, 5, n_samples),
        'serum_sodium': np.random.normal(140, 3, n_samples),
        'albumin': np.random.normal(40, 5, n_samples),
        'total_protein': np.random.normal(70, 5, n_samples),
    })

    #
    df['age'] = df['age'].clip(lower=40, upper=100)
    df['lvef'] = df['lvef'].clip(lower=20, upper=80)
    df['admission_date'] = pd.to_datetime(df['admission_date'])

    # 
    df = df.sort_values(by=['admission_date']).reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(f"Done! Created {filename} with {df.shape[1]} variables.")
    print(f"Includes: {', '.join(df.columns[:5])} ... and {df.shape[1] - 5} more.")


if __name__ == "__main__":
    generate_dummy_data()