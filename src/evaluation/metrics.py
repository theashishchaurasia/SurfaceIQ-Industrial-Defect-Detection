import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
def classification_metrics(y_true,y_pred,labels=None):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f),
      "weighted_precision":float(pw),"weighted_recall":float(rw),"weighted_f1":float(fw),
      "confusion_matrix":confusion_matrix(y_true,y_pred,labels=labels).tolist(),
      "report":classification_report(y_true,y_pred,labels=labels,zero_division=0,output_dict=True)}
def segmentation_scores(pred,target,eps=1e-7):
    pred=np.asarray(pred)>0; target=np.asarray(target)>0
    inter=np.logical_and(pred,target).sum(); union=np.logical_or(pred,target).sum()
    return {"iou":float((inter+eps)/(union+eps)),"dice":float((2*inter+eps)/(pred.sum()+target.sum()+eps))}
