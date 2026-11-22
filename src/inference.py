import time,torch
from PIL import Image
from torchvision import transforms
from src.config import load_config,project_path
from src.data.dataset import CLASSES
from src.models.classifiers import build_classifier
def load_model(model_name="cnn",config_path=None):
    cfg=load_config(config_path); path=project_path(cfg["models"]["checkpoint_dir"],cfg)/f"{model_name}_best.pt"
    if not path.exists(): raise FileNotFoundError(f"Checkpoint missing: {path}; train first.")
    ck=torch.load(path,map_location="cpu",weights_only=False); classes=ck.get("classes",CLASSES)
    model=build_classifier(model_name,len(classes)); model.load_state_dict(ck["state_dict"]); model.eval()
    tf=transforms.Compose([transforms.Resize((ck.get("image_size",200),)*2),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    return model,tf,classes
def predict_image(image,model_name="cnn"):
    model,tf,classes=load_model(model_name)
    im=Image.open(image).convert("RGB") if isinstance(image,(str,bytes)) else image.convert("RGB")
    x=tf(im).unsqueeze(0); start=time.perf_counter()
    with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0]
    ms=(time.perf_counter()-start)*1000; idx=int(probs.argmax())
    return {"label":classes[idx],"confidence":float(probs[idx]),"probabilities":{c:float(probs[i]) for i,c in enumerate(classes)},"inference_ms":ms}
