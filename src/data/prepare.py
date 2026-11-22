import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.config import load_config,project_path
from src.data.dataset import validate_dataset,CLASSES
def inspect_and_export(config_path=None):
    cfg=load_config(config_path); report=validate_dataset(project_path(cfg["dataset"]["root"],cfg))
    out=project_path(cfg["results_dir"],cfg); out.mkdir(parents=True,exist_ok=True)
    (out/"dataset_summary.json").write_text(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2),encoding="utf-8")
    pd.DataFrame(report["records"]).to_csv(out/"dataset_manifest.csv",index=False)
    pd.Series(report["class_counts"]).reindex(CLASSES).plot(kind="bar",figsize=(9,4),title="NEU-DET class distribution")
    plt.ylabel("Image count"); plt.tight_layout(); plt.savefig(out/"class_distribution.png"); plt.close()
    return report
def split_records(records,seed=42,train_fraction=.7,val_fraction=.15):
    labels=[r["label"] for r in records]
    tr,rest=train_test_split(records,test_size=1-train_fraction,random_state=seed,stratify=labels)
    val_share=val_fraction/(1-train_fraction)
    va,te=train_test_split(rest,test_size=1-val_share,random_state=seed,stratify=[r["label"] for r in rest])
    return tr,va,te
