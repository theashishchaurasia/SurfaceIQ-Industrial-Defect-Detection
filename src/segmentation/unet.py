from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True),
      nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(self,base=32):
        super().__init__(); self.e1=block(3,base); self.e2=block(base,base*2); self.e3=block(base*2,base*4)
        self.pool=nn.MaxPool2d(2); self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=block(base*2,base); self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b))
        y=self.d2(torch.cat([self.u2(c),b],1)); y=self.d1(torch.cat([self.u1(y),a],1))
        return self.out(y)
class PairedMaskDataset(Dataset):
    """Pair image and binary pixel mask by identical stem; masks must be real annotations."""
    def __init__(self,image_dir,mask_dir,size=200):
        self.items=[]; self.mask_dir=Path(mask_dir); self.size=size
        for p in Path(image_dir).rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"} and (self.mask_dir/(p.stem+".png")).exists(): self.items.append(p)
        if not self.items: raise ValueError("No image/mask pairs; provide pixel-level PNG masks.")
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        from torchvision.transforms import functional as TF
        p=self.items[i]; im=Image.open(p).convert("RGB").resize((self.size,self.size))
        mask=Image.open(self.mask_dir/(p.stem+".png")).convert("L").resize((self.size,self.size))
        return TF.to_tensor(im),(TF.to_tensor(mask)>0).float()
