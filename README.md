# DisasterShield AI — Enhanced EDA and ML pipeline

Run from this folder:
```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The script includes dataset inspection, missingness and duplicate checks, descriptive statistics, histograms, correlation heatmaps, Pearson/Spearman tests, ANOVA/Kruskal tests by city, chi-square tests where suitable, rainfall cleaning and summaries, preprocessing pipelines, five classification algorithms, cross-validation/grid search for KNN/Decision Tree/SVM, and accuracy/precision/recall/macro-F1/log-loss/ROC-AUC metrics where supported.

Limitations: the climate dataset's provenance is not independently verified. Rainfall bins are exploratory indicators, not verified flood labels. This does not claim validated flood prediction. Original CPCB `city_day.csv`/hourly datasets were not supplied.
