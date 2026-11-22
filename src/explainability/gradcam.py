import torch,numpy as np,cv2
from PIL import Image
class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model; self.activations=None; self.gradients=None
        self.h1=target_layer.register_forward_hook(self._forward)
        self.h2=target_layer.register_full_backward_hook(self._backward)
    def _forward(self,m,i,o): self.activations=o.detach()
    def _backward(self,m,gi,go): self.gradients=go[0].detach()
    def generate(self,x,class_index=None):
        self.model.zero_grad(set_to_none=True); logits=self.model(x)
        idx=int(logits.argmax(1).item()) if class_index is None else int(class_index)
        logits[:,idx].sum().backward()
        weights=self.gradients.mean((2,3),keepdim=True)
        cam=torch.relu((weights*self.activations).sum(1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); return cam,idx
    def close(self): self.h1.remove(); self.h2.remove()
def overlay_heatmap(image,cam,alpha=.4):
    rgb=np.asarray(image.convert("RGB")); heat=cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    heat=cv2.cvtColor(cv2.resize(heat,(rgb.shape[1],rgb.shape[0])),cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(np.clip((1-alpha)*rgb+alpha*heat,0,255)))
