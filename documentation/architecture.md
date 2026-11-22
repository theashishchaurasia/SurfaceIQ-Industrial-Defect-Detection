# Architecture

NEU-DET images/XML → dataset discovery, validation, label inference and manifest → stratified split → custom CNN / ResNet18 → checkpoints and prediction → Grad-CAM / Streamlit. Separate optional paths: Pascal VOC boxes → YOLO; image plus real pixel mask → U-Net. The raw dataset is read-only. Config paths resolve relative to project root.
