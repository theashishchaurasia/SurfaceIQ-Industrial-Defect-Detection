from src.evaluation.metrics import classification_metrics,segmentation_scores
def test_metrics():
    r=classification_metrics([0,1,1],[0,1,0],labels=[0,1]); assert r["accuracy"]==2/3
    s=segmentation_scores([[1,0],[0,1]],[[1,0],[0,0]]); assert 0<=s["iou"]<=1 and 0<=s["dice"]<=1
