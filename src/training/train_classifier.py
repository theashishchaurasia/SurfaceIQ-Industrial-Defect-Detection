import json,time
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.config import load_config,project_path
from src.data.dataset import discover_images,NEUDataset,CLASSES
from src.data.prepare import split_records
from src.models.classifiers import build_classifier
def epoch_run(model,loader,loss_fn,opt,device,train):
    model.train(train); loss_sum=correct=total=0
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            logits=model(x); loss=loss_fn(logits,y)
            if train: loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
    return loss_sum/max(total,1),correct/max(total,1)
def train_classifier(model_name="cnn",epochs=None,config_path=None):
    cfg=load_config(config_path); d=cfg["dataset"]; t=cfg["training"]
    records=[r for r in discover_images(project_path(d["root"],cfg)) if r["label"] in CLASSES]
    if len(records)<12: raise ValueError("Need at least 12 labeled images; check dataset folder names/XML.")
    tr,va,te=split_records(records,cfg["project"]["seed"],d["train_fraction"],d["val_fraction"])
    mk=lambda rs,aug=False: DataLoader(NEUDataset(rs,d["image_size"],aug),batch_size=d["batch_size"],shuffle=aug,num_workers=d["num_workers"])
    train_loader,val_loader=mk(tr,True),mk(va)
    device=torch.device("cuda" if t.get("device")=="cuda" and torch.cuda.is_available() else "cpu")
    model=build_classifier(model_name).to(device); loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=2)
    best=float("inf"); wait=0; history=[]
    ckpt=project_path(cfg["models"]["checkpoint_dir"],cfg); ckpt.mkdir(parents=True,exist_ok=True)
    for e in range(epochs or t["epochs"]):
        start=time.time(); tl,ta=epoch_run(model,train_loader,loss_fn,opt,device,True)
        vl,vacc=epoch_run(model,val_loader,loss_fn,opt,device,False); sched.step(vl)
        history.append({"epoch":e+1,"train_loss":tl,"train_accuracy":ta,"val_loss":vl,"val_accuracy":vacc,"seconds":time.time()-start})
        if vl<best:
            best=vl; wait=0
            torch.save({"model_name":model_name,"state_dict":model.state_dict(),"classes":CLASSES,"image_size":d["image_size"]},ckpt/f"{model_name}_best.pt")
        else:
            wait+=1
            if wait>=t["patience"]: break
    out=project_path(cfg["results_dir"],cfg); out.mkdir(parents=True,exist_ok=True)
    (out/f"{model_name}_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return history
