from pathlib import Path
import hashlib, xml.etree.ElementTree as ET
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES=["crazing","inclusion","patches","pitted_surface","rolled_in_scale","scratches"]
ALIASES={"pitted surface":"pitted_surface","rolled-in scale":"rolled_in_scale","rolled in scale":"rolled_in_scale"}
EXTS={".jpg",".jpeg",".png",".bmp"}
def canonical_label(s):
    key=" ".join(s.lower().replace("_"," ").replace("-"," ").split())
    return ALIASES.get(key,key.replace(" ","_"))
def xml_label(p):
    for x in (p.with_suffix(".xml"),p.parent/"annotations"/(p.stem+".xml")):
        if x.exists():
            try:
                names=[o.findtext("name") for o in ET.parse(x).getroot().findall(".//object") if o.findtext("name")]
                if names: return canonical_label(names[0])
            except (ET.ParseError,OSError): return None
    return None
def discover_images(root):
    root=Path(root).expanduser()
    if not root.exists(): raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    records=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            label=None
            for part in reversed(p.relative_to(root).parts[:-1]):
                candidate=canonical_label(part)
                if candidate in CLASSES: label=candidate; break
            records.append({"path":str(p),"label":label or xml_label(p)})
    return records
def validate_dataset(root):
    records=discover_images(root); hashes={}; valid=[]; corrupt=[]; duplicates=[]; unknown=[]
    for rec in records:
        p=Path(rec["path"])
        try:
            with Image.open(p) as im: im.verify()
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in hashes: duplicates.append({"path":str(p),"duplicate_of":hashes[digest]})
            else: hashes[digest]=str(p)
            if rec["label"] not in CLASSES: unknown.append(str(p))
            else: valid.append(rec)
        except (UnidentifiedImageError,OSError,ValueError): corrupt.append(str(p))
    counts={c:sum(r["label"]==c for r in valid) for c in CLASSES}
    return {"total_images":len(records),"valid_labeled_images":len(valid),"corrupt_images":corrupt,
            "duplicates":duplicates,"unknown_label_images":unknown,"class_counts":counts,"records":valid}
class NEUDataset(Dataset):
    def __init__(self,records,image_size=200,augment=False):
        self.records=records
        ops=[transforms.Resize((image_size,image_size))]
        if augment: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8)]
        ops += [transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(r["path"]) as im: image=im.convert("RGB")
        return self.tf(image),CLASSES.index(r["label"])
