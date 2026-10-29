import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,"data","raw")
OUT=os.path.join(ROOT,"outputs")
for d in ["eda","models","evidence"]: os.makedirs(os.path.join(OUT,d),exist_ok=True)
sns.set_theme(style="whitegrid")
def load(name):
    p=os.path.join(DATA,name)
    if not os.path.exists(p): print("Missing dataset:",p); return None
    d=pd.read_csv(p,low_memory=False); d.columns=d.columns.astype(str).str.strip()
    print(f"\n{name}: {d.shape[0]} rows x {d.shape[1]} columns")
    print(d.head(3).to_string(index=False)); print(d.info())
    print("Missing cells:",d.isna().sum().sum(),"Duplicates:",d.duplicated().sum())
    return d
def eda(d,name):
    s=name.replace(".csv","").replace(" ","_")
    q=pd.DataFrame({"dtype":d.dtypes.astype(str),"missing":d.isna().sum(),
      "missing_pct":(d.isna().mean()*100).round(2),"unique":d.nunique()})
    q.to_csv(f"{OUT}/eda/{s}_quality.csv")
    q["missing_pct"].sort_values(ascending=False).head(20).plot(kind="bar",figsize=(9,4),title=f"Missing values: {s}")
    plt.ylabel("Percent"); plt.tight_layout(); plt.savefig(f"{OUT}/eda/{s}_missing.png",dpi=150); plt.close()
    n=d.select_dtypes(include=np.number)
    if len(n.columns):
        n.describe().T.to_csv(f"{OUT}/eda/{s}_descriptive_stats.csv")
    for c in n.columns[:6]:
            plt.figure(figsize=(6, 3))
            plt.hist(n[c].dropna(), bins=30)
            plt.title(f"{c} distribution")
            filename = f"{s}_{c.replace('/', '_').replace(chr(10), '').replace(chr(13), '')}_hist.png"
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, "eda", filename), dpi=130)
            plt.close()

            plt.figure(figsize=(9, 6))
            sns.heatmap(n.corr(), cmap="coolwarm", center=0)
            plt.tight_layout()
            plt.savefig(f"{OUT}/eda/{s}_correlation.png", dpi=150)
            plt.close()
def statistical_tests(d):
    d=d.copy()
    if "AQI" not in d: return
    d["AQI"]=pd.to_numeric(d["AQI"],errors="coerce")
    d=d.dropna(subset=["AQI"])
    rows=[]
    for c in ["Temperature_Avg (°C)","Humidity (%)","Rainfall (mm)","Wind_Speed (km/h)","Pressure (hPa)","Cloud_Cover (%)"]:
        if c in d:
            x=pd.to_numeric(d[c],errors="coerce"); z=pd.DataFrame({"x":x,"y":d.AQI}).dropna()
            if len(z)>2 and z.x.nunique()>1:
                pr=stats.pearsonr(z.x,z.y); sr=stats.spearmanr(z.x,z.y)
                rows.append([c,pr.statistic,pr.pvalue,sr.statistic,sr.pvalue])
    pd.DataFrame(rows,columns=["feature","pearson_r","pearson_p","spearman_rho","spearman_p"]).to_csv(f"{OUT}/eda/aqi_correlations_tests.csv",index=False)
    if "City" in d:
        groups=[g.AQI.values for _,g in d.groupby("City") if len(g)>1]
        if len(groups)>1:
            with open(f"{OUT}/eda/city_aqi_tests.txt","w") as f:
                f.write(f"ANOVA: {stats.f_oneway(*groups)}\nKruskal: {stats.kruskal(*groups)}\n")
    if "AQI_Category" in d and "State" in d:
        tab=pd.crosstab(d.State,d.AQI_Category)
        if tab.shape[0]>1 and tab.shape[1]>1:
            chi=stats.chi2_contingency(tab)
            with open(f"{OUT}/eda/state_category_chi_square.txt","w") as f: f.write(str(chi))
def rainfall(d,level):
    if d is None:return
    eda(d,f"rainfall_{level}.csv")
    if not {"Date","Daily Actual"}.issubset(d.columns):
        print(level,"has no Date/Daily Actual columns; EDA only."); return
    d["Date"]=pd.to_datetime(d.Date,errors="coerce")
    d["Daily Actual"]=pd.to_numeric(d["Daily Actual"],errors="coerce")
    c=d.drop_duplicates().dropna(subset=["Date","Daily Actual"]).copy()
    c["Rainfall_Risk"]=pd.cut(c["Daily Actual"],[-0.001,2.5,15.5,np.inf],labels=["Low","Moderate","High"])
    c.to_csv(f"{OUT}/eda/rainfall_{level}_cleaned.csv",index=False)
    if "State" in c:
        c.groupby("State",observed=False)["Daily Actual"].agg(["count","mean","median","max"]).sort_values("mean",ascending=False).to_csv(f"{OUT}/eda/{level}_state_summary.csv")
    print(level,"clean rows:",len(c),"; risk bins are exploratory, not flood labels.")
def model_run(d):
    target="AQI_Category"
    if target not in d or "AQI" not in d: print("No AQI classification target; skipping models."); return
    d=d.copy(); d["Date"]=pd.to_datetime(d.get("Date"),errors="coerce")
    d=d.drop_duplicates().dropna(subset=[target,"Date"])
    d["Month"]=d.Date.dt.month; d["DayOfYear"]=d.Date.dt.dayofyear
    d=d.drop(columns=["Date"])
    y=LabelEncoder().fit_transform(d[target].astype(str)); classes=np.unique(y)
    X=d.drop(columns=[target,"AQI"],errors="ignore")
    cats=X.select_dtypes(include=["object","category"]).columns.tolist()
    nums=X.select_dtypes(include=np.number).columns.tolist()
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    cv=StratifiedKFold(2,shuffle=True,random_state=42)
    base=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000))])
    models={"Dummy Baseline":Pipeline([("prep",prep),("model",DummyClassifier(strategy="most_frequent"))]),
      "Logistic Regression":base,
      "KNN":GridSearchCV(Pipeline([("prep",prep),("model",KNeighborsClassifier())]),{"model__n_neighbors":[3,5,7,11]},cv=cv,scoring="f1_macro",n_jobs=1),
      "Decision Tree":GridSearchCV(Pipeline([("prep",prep),("model",DecisionTreeClassifier(random_state=42))]),{"model__max_depth":[3,5,8,None],"model__min_samples_leaf":[1,5,10]},cv=cv,scoring="f1_macro",n_jobs=1),
      "Naive Bayes":Pipeline([("prep",prep),("model",GaussianNB())]),
      "SVM":GridSearchCV(Pipeline([("prep",prep),("model",SVC(probability=False,random_state=42))]),{"model__C":[1,5],"model__gamma":["scale"]},cv=cv,scoring="f1_macro",n_jobs=1)}
    rows=[]; best=(-1,None,None,None,None)
    for name,m in models.items():
        try:
            m.fit(Xtr,ytr); fitted=m.best_estimator_ if hasattr(m,"best_estimator_") else m
            pred=fitted.predict(Xte); proba=fitted.predict_proba(Xte) if hasattr(fitted,"predict_proba") else None
            row={"model":name,"accuracy":accuracy_score(yte,pred),"precision_macro":precision_score(yte,pred,average="macro",zero_division=0),
              "recall_macro":recall_score(yte,pred,average="macro",zero_division=0),"f1_macro":f1_score(yte,pred,average="macro",zero_division=0)}
            if proba is not None:
                row["log_loss"]=log_loss(yte,proba,labels=np.arange(len(np.unique(y))))
                row["roc_auc_ovr_macro"]=roc_auc_score(yte,proba,multi_class="ovr",average="macro",labels=np.arange(len(np.unique(y)))) if len(np.unique(y))>2 else np.nan
            row["best_parameters"]=json.dumps(getattr(m,"best_params_",{})); rows.append(row)
            if row["f1_macro"]>best[0]: best=(row["f1_macro"],row,fitted,pred,proba)
            print(name,row)
        except Exception as e: print("Skipped",name,str(e))
    res=pd.DataFrame(rows); res.to_csv(f"{OUT}/models/model_comparison.csv",index=False)
    if res.empty:return
    res.set_index("model")[["accuracy","precision_macro","recall_macro","f1_macro"]].plot(kind="bar",figsize=(10,5))
    plt.ylim(0,1); plt.tight_layout(); plt.savefig(f"{OUT}/models/model_metrics.png",dpi=160); plt.close()
    _,row,model,pred,proba=best
    ConfusionMatrixDisplay.from_predictions(yte,pred,display_labels=LabelEncoder().fit(d[target].astype(str)).classes_,xticks_rotation=45)
    plt.tight_layout(); plt.savefig(f"{OUT}/models/best_confusion_matrix.png",dpi=160); plt.close()
    with open(f"{OUT}/models/best_model_report.txt","w") as f:
        f.write(json.dumps(row,indent=2)+"\n\n"+classification_report(yte,pred,zero_division=0))
    print("Best model by macro-F1:",row)
def split_experiment(d):
    target="AQI_Category"
    if target not in d or "Date" not in d:return
    z=d.copy(); z["Date"]=pd.to_datetime(z.Date,errors="coerce"); z=z.drop_duplicates().dropna(subset=[target,"Date"])
    z["Month"]=z.Date.dt.month; z["DayOfYear"]=z.Date.dt.dayofyear; z=z.drop(columns=["Date"])
    y=LabelEncoder().fit_transform(z[target].astype(str)); X=z.drop(columns=[target,"AQI"],errors="ignore")
    cats=X.select_dtypes(include=["object","category"]).columns.tolist(); nums=X.select_dtypes(include=np.number).columns.tolist()
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)])
    rows=[]
    for label,test in [("70:30",.30),("60:40",.40),("75:25",.25),("80:20",.20)]:
      a,b,c,e=train_test_split(X,y,test_size=test,random_state=42,stratify=y)
      for name,est in [("Dummy Baseline",DummyClassifier(strategy="most_frequent")),("KNN",KNeighborsClassifier(n_neighbors=3)),("Decision Tree",DecisionTreeClassifier(random_state=42)),("Naive Bayes",GaussianNB())]:
        try:
          pipe=Pipeline([("prep",prep), ("model",est)]); pipe.fit(a,c); pred=pipe.predict(b)
          rows.append({"split_ratio_train_test":label,"train_rows":len(a),"test_rows":len(b),"model":name,"accuracy":accuracy_score(e,pred),"precision_macro":precision_score(e,pred,average="macro",zero_division=0),"recall_macro":recall_score(e,pred,average="macro",zero_division=0),"f1_macro":f1_score(e,pred,average="macro",zero_division=0)})
        except Exception as ex: print("Split experiment skipped",label,name,str(ex))
    r=pd.DataFrame(rows); r.to_csv(f"{OUT}/models/split_ratio_comparison.csv",index=False)
    if not r.empty:
      print("\nTRAIN/TEST SPLIT COMPARISON (higher macro-F1 is generally preferred):\n",r.sort_values(["split_ratio_train_test","f1_macro"],ascending=[True,False]).to_string(index=False))
      r.pivot(index="split_ratio_train_test",columns="model",values="accuracy").plot(kind="bar",figsize=(9,4)); plt.ylabel("Accuracy"); plt.ylim(0,1); plt.tight_layout(); plt.savefig(f"{OUT}/models/split_ratio_accuracy.png",dpi=160); plt.close()

def main():
    climate=load("Indian_Climate_Dataset_2024_2025(20260919-031130).csv")
    district=load("rainfall_districtwise_daily_imd.csv"); state=load("rainfall_statewise_daily_imd.csv")
    if climate is not None: eda(climate,"climate.csv"); statistical_tests(climate); model_run(climate); split_experiment(climate)
    rainfall(district,"districtwise"); rainfall(state,"statewise")
    print("Finished. See outputs/eda and outputs/models. Capture genuine screenshots of your own run.")
if __name__=="__main__": main()
