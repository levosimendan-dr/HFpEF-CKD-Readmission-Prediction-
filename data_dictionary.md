# Data Dictionary

This document provides a comprehensive description of all variables used in the analysis for the paper "Development and Temporal Validation of a Machine Learning Model for Readmission in HFpEF and CKD".

## 1. Outcomes (Target) & Administrative
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `readmission_within_1_year` | Readmission (1-Year) | Binary | 0=No, 1=Yes (Primary Outcome) |
| `admission_date` | Admission Date | Date | Format: YYYY-MM-DD |
| `length_of_stay` | Length of Stay | Numeric | Days |

## 2. Demographics & Lifestyle
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `age` | Age | Numeric | Years |
| `sex` | Sex | Binary | 0=Female, 1=Male |
| `bmi` | BMI | Numeric | kg/m² |
| `obesity` | Obesity | Binary | 0=No, 1=Yes (BMI ≥ 30) |
| `smoker` | Smoking History | Binary | 0=No, 1=Yes |

## 3. Clinical Vitals & Status
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `systolic_bp` | Systolic BP | Numeric | mmHg |
| `diastolic_bp` | Diastolic BP | Numeric | mmHg |
| `heart_rate` | Heart Rate | Numeric | bpm |
| `nyha_class` | NYHA Class | Ordinal | I, II, III, IV (Encoded as 1-4) |

## 4. Echocardiography Parameters
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `lvef` | LVEF | Numeric | % (Left Ventricular Ejection Fraction) |
| `E_over_e_prime` | E/e' | Numeric | Ratio |
| `LAVI` | LAVI | Numeric | mL/m² (Left Atrial Volume Index) |
| `TR_velocity` | TR Velocity | Numeric | m/s (Tricuspid Regurgitation Velocity) |
| `LVMI` | LVMI | Numeric | g/m² (Left Ventricular Mass Index) |

## 5. Biomarkers & Laboratory Results
### 5.1 Renal & Metabolic
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `nt_probnp` | NT-proBNP | Numeric | pg/mL |
| `egfr` | eGFR | Numeric | mL/min/1.73m² |
| `serum_creatinine` | Serum Creatinine | Numeric | mg/dL or µmol/L |
| `blood_urea_nitrogen` | BUN | Numeric | mg/dL |
| `cystatin_c` | Cystatin C | Numeric | mg/L |
| `serum_uric_acid` | Uric Acid | Numeric | mg/dL |
| `serum_sodium` | Serum Sodium | Numeric | mmol/L |
| `proteinuria_quantification`| Proteinuria | Numeric | g/24h |
| `history_of_aki` | History of AKI | Binary | 0=No, 1=Yes |

### 5.2 Inflammatory & Others
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `hs_crp` | hs-CRP | Numeric | mg/L |
| `il_6` | IL-6 | Numeric | pg/mL |
| `d_dimer` | D-Dimer | Numeric | ng/mL |
| `homocysteine` | Homocysteine | Numeric | µmol/L |
| `albumin` | Albumin | Numeric | g/dL |
| `total_protein` | Total Protein | Numeric | g/dL |

## 6. Comorbidities
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `diabetes` | Diabetes | Binary | 0=No, 1=Yes |
| `hypertension` | Hypertension | Binary | 0=No, 1=Yes |
| `atrial_fibrillation` | Atrial Fibrillation | Binary | 0=No, 1=Yes |
| `copd` | COPD | Binary | 0=No, 1=Yes |
| `coronary_artery_disease` | CAD | Binary | 0=No, 1=Yes |
| `anemia` | Anemia | Binary | 0=No, 1=Yes |

## 7. Medications
| Variable Name | Pretty Name (Paper) | Type | Unit/Notes |
| :--- | :--- | :--- | :--- |
| `raas_use` | RAAS Inhibitor Use | Binary | 0=No, 1=Yes (ACEi/ARB/ARNI) |
| `beta_blocker_use` | Beta-Blocker Use | Binary | 0=No, 1=Yes |
| `mra_use` | MRA Use | Binary | 0=No, 1=Yes |
| `sglt2i_use` | SGLT2i Use | Binary | 0=No, 1=Yes |

## 8. Engineered Interaction Terms
| Variable Name | Description | Formula |
| :--- | :--- | :--- |
| `Inter_eGFR_NTproBNP` | Cardio-Renal Interaction | `eGFR * NT-proBNP` |
| `Inter_BMI_NTproBNP` | Obesity-Peptide Interaction | `BMI * NT-proBNP` |
| `Inter_hsCRP_NTproBNP` | Inflammatory-Stress Interaction | `hs-CRP * NT-proBNP` |
