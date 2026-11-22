from pathlib import Path
import xml.etree.ElementTree as ET
def inspect_voc_annotations(root):
    xmls=list(Path(root).rglob("*.xml")); valid=[]; errors=[]
    for p in xmls:
        try:
            r=ET.parse(p).getroot(); size=r.find("size")
            if size is None: raise ValueError("missing image size")
            w,h=int(size.findtext("width")),int(size.findtext("height")); n=0
            for obj in r.findall("object"):
                b=obj.find("bndbox")
                if b is None: raise ValueError("missing bndbox")
                x1,y1,x2,y2=[float(b.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
                if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError("invalid box bounds")
                n+=1
            valid.append({"xml":str(p),"objects":n})
        except Exception as e: errors.append({"xml":str(p),"error":str(e)})
    return {"xml_files":len(xmls),"valid":valid,"errors":errors}
def train_yolo(data_yaml,model="yolo11n.pt",epochs=30,imgsz=200):
    try: from ultralytics import YOLO
    except ImportError as e: raise RuntimeError("Install optional ultralytics") from e
    if not Path(data_yaml).exists(): raise FileNotFoundError(data_yaml)
    return YOLO(model).train(data=data_yaml,epochs=epochs,imgsz=imgsz,device="cpu")
def predict_yolo(image_path,weights):
    try: from ultralytics import YOLO
    except ImportError as e: raise RuntimeError("Install optional ultralytics") from e
    return YOLO(weights).predict(source=str(image_path),device="cpu",verbose=False)
