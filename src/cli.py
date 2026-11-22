import argparse
from src.data.prepare import inspect_and_export
from src.training.train_classifier import train_classifier
from src.inference import predict_image
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("inspect-data")
    t=sub.add_parser("train"); t.add_argument("--model",choices=["cnn","resnet18"],default="cnn"); t.add_argument("--epochs",type=int)
    q=sub.add_parser("predict"); q.add_argument("--image",required=True); q.add_argument("--model",choices=["cnn","resnet18"],default="cnn")
    sub.add_parser("fundamentals")
    a=p.parse_args()
    if a.cmd=="inspect-data":
        r=inspect_and_export(); print("Images",r["total_images"],"usable",r["valid_labeled_images"],r["class_counts"])
    elif a.cmd=="train": print("Epochs completed:",len(train_classifier(a.model,a.epochs)))
    elif a.cmd=="predict": print(predict_image(a.image,a.model))
    else:
        from src.fundamentals.optimizers import optimize
        for name in ["batch_gd","sgd","momentum","nesterov"]: print(name,float(optimize(name)[-1]))
if __name__=="__main__": main()
