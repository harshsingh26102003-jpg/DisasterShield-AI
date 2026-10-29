# DisasterShield AI — code review against faculty requirements and reference rent-project workflow

## Faculty requirement traceability
- Societal challenge and measurable objectives: documented in the report.
- Datasets larger than 2,000 rows: climate (7,310 × 13), district rainfall (51,648 × 22), state rainfall (2,841 × 21).
- Data preparation: schema inspection, missingness/duplicate audit, date parsing, numeric conversion, cleaned rainfall analysis copies.
- Visual EDA: missingness charts, numeric histograms, descriptive statistics, correlation heatmaps, model metric and confusion matrix charts.
- Statistical analysis: Pearson and Spearman correlations, city-group ANOVA and Kruskal-Wallis tests, state/category chi-square where available.
- Predictive models: DummyClassifier baseline, Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, SVM.
- Preprocessing: median/mode imputation, scaling, one-hot encoding inside sklearn Pipeline/ColumnTransformer.
- Model tuning: GridSearchCV and stratified cross-validation for KNN, Decision Tree, SVM.
- Metrics: accuracy, macro precision, macro recall, macro-F1, log-loss and multiclass ROC-AUC when probability outputs are available.
- Notebook split-ratio request: 70:30, 60:40, 75:25, 80:20; KNN, Decision Tree, Naive Bayes and baseline; outputs saved to `outputs/models/split_ratio_comparison.csv`.
- Report, references, code excerpts, reproducibility guide, viva questions and final checklist are included in the DOCX/PDF.

## Comparison with the supplied friend rent-project code
Transferable elements are included: data shape/info, missing/duplicate checks, distribution and correlation plots, Pearson/Spearman, group tests, categorical association test, pipeline preprocessing, train/test split, CV/grid search, model-wise metrics and confusion matrix, split-ratio comparison. Rent-specific fields/targets (Rent, Floor parsing, City-median High_Rent, rent log transform) are not applicable and were not copied.

## Results integrity
The earlier completed five-model run produced the model comparison CSV. A later full run with the new DummyClassifier baseline completed successfully and updated the model table. The classifier target remains `AQI_Category`, not flood/no-flood. AQI itself is excluded from predictors to reduce target leakage. Rainfall bins are exploratory categories, not validated flood labels. Do not describe the current model as a validated flood prediction or early-warning system.

## Before submitting
1. Fill registration number, course code and faculty details in the report.
2. Verify the authoritative source URL and license for each CSV; the climate dataset provenance has not been independently verified.
3. Add your genuine GitHub repository URL and LinkedIn post URL; do not invent engagement or contribution evidence.
4. Capture genuine screenshots of your own execution if the instructor requires screenshots.
5. Re-run from a clean extraction if time permits: `python -m pip install -r requirements.txt` then `python run_pipeline.py`.
