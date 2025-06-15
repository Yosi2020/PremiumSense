# 📓 Notebook Index

Run these in order to reproduce the full PremiumSense workflow:

- **1_data_understanding.ipynb**  
  • Load raw Feb 2014–Aug 2015 data  
  • Inspect schema, data types, missingness  
  • Compute overall & by-group loss ratios
  **data_cleaning.ipynb**  
  • Impute or drop missing values  
  • Convert dtypes (dates, categoricals)  
  • Cap outliers & save cleaned dataset
  **exploratory_analysis.ipynb**  
  • Univariate distributions (histograms, bar charts)  
  • Bivariate/multivariate trends (scatter plots, correlation heatmap)  
  • Produce 3 key “insight” plots

- **2_hypothesis_testing.ipynb**  
  • Define risk (frequency, severity) & margin metrics  
  • Segment by province, postal code, gender  
  • Run χ², t-tests, ANOVA  
  • Interpret results in business terms

- **3_predictive_modeling.ipynb**  
  • Prepare train/test splits  
  • Build regression models for claim severity (Linear, RF, XGBoost)  
  • Build classification model for claim probability  
  • Evaluate (RMSE, R², precision/recall/F1)  
  • Analyze top features with SHAP

---

| **Notebook**                          | **Purpose**                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **1_Exploratory_Data_Analysis.ipynb** | Load raw Feb 2014–Aug 2015 data; inspect schema, dtypes, missingness; compute overall & by-group loss ratios. |
| **data_cleaning**                     | Impute/drop missing values; convert dtypes; cap outliers; save cleaned dataset.                               |
| **exploratory_analysis**              | Plot univariate distributions & bivariate trends; create 3 key insight plots.                                 |
| **2_hypothesis_testing.ipynb**        | Define metrics; segment data; run χ², t-tests, ANOVA; interpret business impact.                              |
| **3_predictive_modeling.ipynb**       | Train/test split; regression & classification models; evaluate metrics; SHAP feature-importance analysis.     |
