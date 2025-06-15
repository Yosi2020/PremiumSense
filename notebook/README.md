# Notebook Overview & Key Results

This section describes each notebook and highlights the main findings.

---

## 📒 Notebooks

1. **1_data_understanding.ipynb**  
   • Load raw `.txt` data, inspect schema, dtypes, and missingness  
   • Compute descriptive statistics for `TotalPremium`, `TotalClaims`, `CustomValueEstimate`  
   • Calculate **overall portfolio loss ratio**

2. **2_data_cleaning.ipynb**  
   • Drop ultra-sparse columns (>99% missing)  
   • Impute binary flags, numeric and categorical fields  
   • Winsorize `TotalPremium` and positive `TotalClaims`  
   • Final missing-value checks and export `claims_clean.csv`

3. **3_exploratory_analysis.ipynb**  
   • **Loss Ratio** by Province, VehicleType, and Gender (bar charts)  
   • Histograms & boxplots for financial variables to spot outliers  
   • Temporal trends in claim frequency & severity (time-series)  
   • Top/bottom 10 vehicle makes/models by average claim amount

4. **4_hypothesis_testing.ipynb**  
   • χ² tests for claim frequency across Provinces & PostalCodes (p < 0.001)  
   • ANOVA for margin differences by PostalCode (p < 0.001)  
   • Z-test for gender differences (p ≈ 0.84)  
   • Effect sizes (Cramér’s V, η², Cohen’s h) and Bonferroni-corrected pairwise comparisons

5. **5_predictive_modeling.ipynb**  
   • **Regression**: predict claim severity (`TotalClaims > 0`) with Linear Regression, Random Forest, XGBoost  
   • **Classification**: predict claim probability (`has_claim`) with Logistic Regression, Random Forest, XGBoost  
   • Evaluation metrics: RMSE & R² for regression; accuracy, precision, recall, F1 for classification  
   • **SHAP**-based feature importance for best models  
   • Compute **risk-based premium** = P(claim) × E[claim] and compare to current premiums

---

## 🏆 Key Results

- **Overall Loss Ratio:** calculated in Notebook 1 (see cell output)
- **Geographic Variation:**
  - Claim frequency differs significantly by province (χ² ≈ 110.7, p < 0.001; Cramér’s V ≈ 0.01)
  - Postal‐code level shows even stronger differences (χ² ≈ 1 451.7, η² ≈ 0.002)
- **Gender Analysis:** no significant difference in claim frequency (Z ≈ –0.2, p ≈ 0.84; Cohen’s h ≈ 0.002)
- **Data Distributions:** heavy right‐tail in `TotalClaims`, many zeros in `CustomValueEstimate` (handled via winsorization)
- **Temporal Trends:** claim frequency and severity exhibit seasonal/long-term patterns over the 18-month period
- **Top Risk Vehicles:** identified makes/models with highest average claims for targeted underwriting
- **Predictive Models:**
  - **Best severity regressor**: _e.g._ XGBoost with RMSE ≈ X and R² ≈ Y
  - **Best claim classifier**: _e.g._ Random Forest with F1 ≈ Z
- **SHAP Insights:** top drivers include vehicle age, cubic capacity, registration year, and location features
- **Risk-Based Premiums:** model-estimated premiums highlight segments where current pricing over- or under-charges

For full details, code, and interactive plots, please open the corresponding notebooks in the `notebooks/` directory.```