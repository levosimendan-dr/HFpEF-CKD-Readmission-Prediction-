# Development and Temporal Validation of a Machine Learning Model for Readmission in HFpEF and CKD

This repository contains the source code and reproduction pipeline for the paper:  
**"Development and Temporal Validation of a Machine Learning Model for Readmission in Heart Failure With Preserved Ejection Fraction and Chronic Kidney Disease"**

**Authors:** Ping Xie*, Yaoting Deng, Weijie Lu, Yang Zhong, JiaJia Liu, Pengcheng Sheng, Mengyang Liu, Kang Yang, Yujie Hu, Nan Ma (*Corresponding Author)

## 🚀 Key Features

* **Adherence to FAIR Principles:** Findable, Accessible, Interoperable, and Reusable.
* **Privacy-Preserving:** Includes a built-in synthetic data generator.
* **Robust Analysis:** Implements temporal validation, MAGGIC score approximation, and stability selection.

## 📂 Repository Structure

* `src/`: Contains the main analysis script (`main_analysis.py`).
* `data/`: Contains the Data Dictionary (`data_dictionary.md`).
* `output/`: Generated figures and tables will be saved here.

## 🛠️ Getting Started

### Prerequisites

* Python 3.8+
* Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data Availability & Reproduction

Due to patient privacy regulations, the original clinical dataset cannot be shared publicly. However, to ensure **reproducibility** without compromising privacy:

1.  **Automatic Dummy Data Generation:** The script `src/main_analysis.py` contains a built-in generator. If the real dataset (`HFpEF_CKD_Simulated_Data_V24_SplitByDate.csv`) is not found, the script will **automatically generate synthetic data** with the same statistical structure and variable names as the original study.
2.  **Data Dictionary:** See `data/data_dictionary.md` for strict variable definitions.

### 🏃 How to Run

1.  Clone the repository.
2.  Run the analysis:
    ```bash
    python src/main_analysis.py
    ```
3.  The script will verify data availability. If the real data is missing, it will generate a synthetic cohort to demonstrate the pipeline.
4.  Check the `output/` directory for generated figures (e.g., Temporal Drift Plots, ROC Curves, Feature Importance).

## ⚙️ Methodology Highlights

* **Temporal Split:** Strict date-based splitting (Derivation vs. Validation cohorts).
* **Feature Selection:** Stability selection using LassoCV and RFECV (50 iterations).
* **Reproducibility:** Global random seed locked (`SEED=42`) for Python, NumPy, and SKLearn.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.