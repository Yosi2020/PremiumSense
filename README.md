# PremiumSense

**End-to-End Insurance Risk Analytics & Predictive Modeling for AlphaCare Insurance Solutions**

PremiumSense is a modular, reproducible pipeline that takes raw historical claims data through data versioning, cleaning, exploratory analysis, hypothesis testing, and predictive modeling—culminating in a risk-based pricing framework. It’s built to help ACIS identify low-risk segments, optimize premium rates, and drive data-backed marketing strategies.

---

## 🚀 Project Overview

1. **Data Versioning (DVC)**  
   Track both raw and cleaned datasets with DVC; no large files in Git.

2. **Data Cleaning & Preparation**  
   • Drop ultra-sparse columns  
   • Impute missing values (numeric, categorical, binary flags)  
   • Winsorize outliers in premiums & claims

3. **Exploratory Data Analysis (EDA)**  
   • Overall & segment loss ratios (Province, VehicleType, Gender)  
   • Distributions & outliers in `TotalPremium`, `TotalClaims`, `CustomValueEstimate`  
   • Temporal trends in claim frequency & severity  
   • Top/bottom vehicle makes/models by average claim

4. **Hypothesis Testing**  
   • χ² tests for claim frequency by Province & PostalCode  
   • ANOVA for margin differences by PostalCode  
   • Z-test for gender differences  
   • Effect sizes (Cramér’s V, η², Cohen’s h) & Bonferroni corrections

5. **Predictive Modeling**  
   • **Regression**: predict claim severity (Linear Regression, Random Forest, XGBoost)  
   • **Classification**: predict claim probability (Logistic Regression, Random Forest, XGBoost)  
   • Evaluation: RMSE & R² (regression), accuracy/precision/recall/F1 (classification)  
   • SHAP-based feature importance  
   • Risk-based premium = P(claim) × E[claim]

---

## 📂 Repository Structure

```
PremiumSense/
├── data/
│ ├── raw/
│ │ └── historical_claims.txt # Original data (DVC-tracked)
│ └── processed/
│ └── claims_clean.csv # Cleaned data (DVC-tracked)
├── scripts/
│ ├── data_loader.py # load_insurance_data()
│ ├── data_cleaning.py # missingness, imputation, outlier capping
│ ├── eda_plots.py # reusable plotting functions
│ ├── hypothesis_testing.py # χ², ANOVA, z-test wrappers
│ └── statistical_modeling.py # pipelines, model training, SHAP, premium calc
├── notebooks/
│ ├── 1_Exploratory_Data_Analysis.ipynb # schema, missingness, summary stats
│ ├── data_cleaning # cleaning steps & output
│ ├── exploratory_analysis # univariate, bivariate, temporal EDA
│ ├── 2_hypothesis_testing.ipynb # hypothesis tests & effect sizes
│ └── 3_predictive_modeling.ipynb # regression, classification, SHAP, pricing
├── .gitignore
├── dvc.yaml & .dvc/ # DVC pipeline & metadata
├── dvc.lock
└── requirements.txt # Python dependencies
```

---

## ⚙️ Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Yosi2020/PremiumSense.git
   cd PremiumSense

   ```

2. **Install Python dependencies**`
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize DVC**
   ```bash
   pip init dvc
   dvc pull
   ```
4. **Run the notebooks**
   ```bash
   jupyter notebook
   ```

---

# 📒 Notebook Workflow

Work through the notebooks in numeric order:

1. **1_Exploratory_Data_Analysis.ipynb**  
   Inspect raw data, types, missingness, and compute overall loss ratio.
   **data_cleaning**  
   Apply cleaning pipeline, handle missing values and outliers, export `claims_clean.csv`.
   **exploratory_analysis**  
   Generate plots and tables to reveal geographic, temporal, and vehicle-based insights.

2. **2_hypothesis_testing.ipynb**  
   Conduct A/B tests on key risk drivers, compute effect sizes, and produce business recommendations.

3. **3_predictive_modeling.ipynb**  
   Train & evaluate regression and classification models, interpret with SHAP, and calculate risk-based premiums.

---

## 📝 Key Findings

- **Loss Ratio Variation**: Significant by Province & PostalCode (_p_<0.001), none by Gender (*p*≈0.84)
- **Temporal Trends**: Seasonal patterns in claim frequency & severity
- **Top Risk Vehicles**: Identified makes & models with highest average claims
- **Best Models**:
  - **Severity**: XGBoost (lowest RMSE, highest R²)
  - **Frequency**: Random Forest (highest F1)
- **SHAP Insights**: Vehicle age, cubic capacity, registration year, and location features drive risk
- **Pricing Implications**:
  - Adopt province-level rate tiers
  - Group ZIPs into risk bands
  - Forego gender-based adjustments

---

## 🤝 Collaboration & Contact

- **Facilitators**: Mahlet, Kerod, Rediet, Rehmet
- **Questions / Feedback**: Open an issue or reach out via Slack

---

## 🔮 Future Work

- Incorporate expense & profit loadings
- Deploy pipeline to production with CI/CD
- Monitor live model performance and recalibrate quarterly

