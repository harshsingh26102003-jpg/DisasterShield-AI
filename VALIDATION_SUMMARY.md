# Validation summary

- Final script: `run_pipeline.py` (136 physical lines).
- Successfully executed in the available Python environment; exit code 0.
- Dataset dimensions: climate 7,310 × 13; district rainfall 51,648 × 22; state rainfall 2,841 × 21.
- EDA includes data types, missingness, duplicates, descriptive statistics, histograms, and correlation heatmaps.
- Statistical tests include Pearson/Spearman correlations, city-level ANOVA/Kruskal where groups exist, and chi-square where categorical columns are available.
- Classification models: Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, and SVM.
- KNN, Decision Tree, and SVM use GridSearchCV with stratified cross-validation.
- Metrics include accuracy, macro precision/recall/F1, log loss, and macro one-vs-rest ROC-AUC.
- Results and charts are saved in `outputs/eda` and `outputs/models`.

Limitations: the climate CSV's provenance has not been independently verified. Observed classification metrics were modest. Rainfall bins are exploratory indicators, not verified flood/no-flood labels. The original CPCB AQI files were not supplied.
