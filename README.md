# Temporal Validation of a Machine Learning Readmission Model in Heart Failure With Preserved Ejection Fraction and Chronic Kidney Disease

This repository contains the source code and reproduction pipeline for the paper:  
**"Temporal Validation of a Machine Learning Readmission Model in Heart Failure With Preserved Ejection Fraction and Chronic Kidney Disease"**

**Authors:** Ping Xie*, Yaoting Deng, Weijie Lu, Yang Zhong, JiaJia Liu, Pengcheng Sheng, Mengyang Liu, Kang Yang, Yujie Hu, Nan Ma (*Corresponding Author)

## 🚀 Key Features

* **Adherence to FAIR Principles:** Findable, Accessible, Interoperable, and Reusable.
* **Privacy-Preserving:** Includes synthetic dummy data and a built-in data generator to protect patient privacy.
* **Robust Analysis:** Implements temporal validation, MAGGIC score approximation, and stability selection.

## 📂 Repository Structure

* `main_analysis.py`: The main script to execute the machine learning pipeline and temporal validation.
* `generate_data.py`: Script used to generate the synthetic cohort.
* `dummy_data.csv`: A pre-generated synthetic dataset mirroring the statistical properties of the original cohort.
* `data_dictionary.md`: Comprehensive definitions of all clinical variables.
* `requirements.txt`: Required Python dependencies.
* `LICENSE.txt`: Project license.

## 🛠️ Getting Started

### Prerequisites

* Python 3.8+
* Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data Availability & Reproduction

Due to stringent patient privacy regulations, the original clinical EHR dataset cannot be shared publicly. However, to ensure **reproducibility** without compromising privacy:

1.  **Synthetic Dummy Data:** We provide `dummy_data.csv`, a synthetic dataset generated to have the exact same statistical structure, distributions, and variable names as the original real-world study cohort.
2.  **Data Generator:** The script `generate_data.py` is included to demonstrate how the synthetic data was constructed.
3.  **Data Dictionary:** Please refer to `data_dictionary.md` for strict variable definitions and encoding rules.

### 🏃 How to Run

1.  Clone this repository to your local machine.
2.  Ensure `dummy_data.csv` is in the same directory as the scripts.
3.  Run the main analysis:
    ```bash
    python main_analysis.py
    ```
4.  The script will execute the full machine learning pipeline using the synthetic cohort. Generated figures (e.g., Temporal Drift Plots, ROC Curves, Feature Importance) and tables will be saved in an automatically created `output/` directory.

## ⚙️ Methodology Highlights

* **Temporal Split:** Strict date-based splitting (Derivation vs. Validation cohorts).
* **Feature Selection:** Stability selection using LassoCV and RFECV (50 iterations).
* **Reproducibility:** Global random seed locked (`SEED=42`) for Python, NumPy, and scikit-learn.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

