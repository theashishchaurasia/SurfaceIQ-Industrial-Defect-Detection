
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from PIL import Image
from ultralytics import YOLO

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, project_path
from src.data.dataset import validate_dataset
from src.inference import predict_image, load_model


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SurfaceIQ | Industrial Inspection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background: #0b1220;
}
.block-container {
    padding-top: 2rem;
}
[data-testid=stMetric] {
    background: #142238;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CONFIGURATION AND MODEL LOADING
# --------------------------------------------------

cfg = load_config()


@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("SurfaceIQ")

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Dataset Explorer",
        "Classification",
        "Detection",
        "Segmentation",
        "Explainable AI",
        "Model Performance",
        "Inspection Reports"
    ]
)

st.title("Industrial Surface Defect Inspection")
st.caption("AI-assisted visual quality inspection | Python / PyTorch")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    a, b, c = st.columns(3)

    a.metric("Defect Classes", 6)
    b.metric("Classifier Models", 2)
    c.metric("Detection Model", "YOLOv8n")

    st.success("SurfaceIQ inspection platform is ready.")

    st.write("""
    SurfaceIQ is an AI-assisted industrial surface inspection
    platform designed to identify and locate manufacturing defects.
    """)

    st.subheader("Available Modules")

    st.markdown("""
    - **Dataset Explorer:** Dataset validation and class distribution.
    - **Classification:** CNN and ResNet18 defect classification.
    - **Detection:** YOLOv8 multiple-defect detection using bounding boxes.
    - **Segmentation:** Placeholder for future pixel-level segmentation.
    - **Explainable AI:** Grad-CAM visual explanations.
    - **Model Performance:** Training and validation performance graphs.
    - **Inspection Reports:** Downloadable inspection results.
    """)

    st.info(
        "Workflow: Dataset audit → Classification → Detection "
        "→ Grad-CAM → Human verification → Inspection report."
    )


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    root = project_path(cfg["dataset"]["root"], cfg)

    st.subheader("Dataset Explorer")
    st.write("Dataset location:", str(root))

    if st.button("Validate Dataset"):

        try:
            r = validate_dataset(root)

            st.metric(
                "Usable Labeled Images",
                r["valid_labeled_images"]
            )

            class_df = pd.DataFrame({
                "Class": list(r["class_counts"].keys()),
                "Images": list(r["class_counts"].values())
            })

            st.dataframe(class_df, use_container_width=True)

            st.write("Corrupt Images:", len(r["corrupt_images"]))
            st.write("Duplicates:", len(r["duplicates"]))
            st.write(
                "Unknown Labels:",
                len(r["unknown_label_images"])
            )

        except Exception as e:
            st.error(str(e))


# --------------------------------------------------
# CLASSIFICATION / EXPLAINABLE AI / REPORTS
# --------------------------------------------------

elif page in [
    "Classification",
    "Explainable AI",
    "Inspection Reports"
]:

    st.subheader(page)

    uploaded_file = st.file_uploader(
        "Upload Industrial Surface Image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="classification_upload"
    )

    model_name = st.selectbox(
        "Select Classification Model",
        ["cnn", "resnet18"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Surface Image",
            use_container_width=True
        )

        if st.button("Inspect Image"):

            try:

                result = predict_image(image, model_name)

                st.session_state["inspection"] = result

                st.success(
                    f"Predicted Defect: "
                    f"{result['label'].replace('_', ' ').title()}"
                )

                st.metric(
                    "Classification Confidence",
                    f"{result['confidence']:.2%}"
                )

                probability_df = pd.DataFrame({
                    "Defect Class": list(
                        result["probabilities"].keys()
                    ),
                    "Probability": list(
                        result["probabilities"].values()
                    )
                })

                fig = px.bar(
                    probability_df,
                    x="Defect Class",
                    y="Probability",
                    title="Defect Classification Probabilities"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    f"Inference Time: "
                    f"{result['inference_ms']:.1f} ms"
                )

                # Grad-CAM
                if page == "Explainable AI":

                    model, transform, classes = load_model(model_name)

                    from src.explainability.gradcam import (
                        GradCAM,
                        overlay_heatmap
                    )

                    target_layer = (
                        model.features[9]
                        if model_name == "cnn"
                        else model.layer4[-1]
                    )

                    cam = GradCAM(model, target_layer)

                    heatmap, _ = cam.generate(
                        transform(image).unsqueeze(0)
                    )

                    cam.close()

                    st.subheader("Grad-CAM Visualization")

                    st.image(
                        overlay_heatmap(image, heatmap),
                        caption="Model Attention Heatmap",
                        use_container_width=True
                    )

                # Inspection Report
                if page == "Inspection Reports":

                    report = pd.DataFrame([{
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "predicted_defect": result["label"],
                        "confidence": result["confidence"],
                        "model": model_name,
                        "inference_ms": result["inference_ms"],
                        "limitation": "Human verification required"
                    }])

                    st.download_button(
                        "Download Classification Report",
                        report.to_csv(index=False),
                        "classification_report.csv",
                        "text/csv"
                    )

            except Exception as e:

                st.error(f"Inspection failed: {e}")


# --------------------------------------------------
# YOLO OBJECT DETECTION
# --------------------------------------------------

elif page == "Detection":

    st.subheader("Industrial Defect Detection")

    st.write("""
    Detect and locate multiple industrial surface defects
    using the trained YOLOv8 model.
    """)

    model_path = PROJECT_ROOT / "models" / "yolo_best.pt"

    if not model_path.exists():

        st.error(
            f"YOLO model not found: {model_path}"
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload Surface Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="yolo_upload"
        )

        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05
        )

        if uploaded_file:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Original Surface Image",
                use_container_width=True
            )

            if st.button("Detect Defects"):

                try:

                    model = load_yolo_model(str(model_path))

                    with st.spinner("Detecting defects..."):

                        results = model.predict(
                            source=image,
                            conf=confidence_threshold,
                            imgsz=320,
                            verbose=False
                        )

                    result = results[0]
                    boxes = result.boxes

                    # Display annotated image
                    annotated_image = result.plot()

                    st.subheader("Detection Visualization")

                    st.image(
                        annotated_image,
                        channels="BGR",
                        caption="YOLOv8 Defect Detection",
                        use_container_width=True
                    )

                    # Extract detection information
                    detections = []

                    if boxes is not None:

                        for box in boxes:

                            class_id = int(box.cls[0].item())

                            confidence = float(
                                box.conf[0].item()
                            )

                            coordinates = box.xyxy[0].tolist()

                            detections.append({
                                "Defect": result.names[class_id],
                                "Confidence (%)": round(
                                    confidence * 100, 2
                                ),
                                "X1": round(coordinates[0], 1),
                                "Y1": round(coordinates[1], 1),
                                "X2": round(coordinates[2], 1),
                                "Y2": round(coordinates[3], 1)
                            })

                    # Display total detections
                    st.subheader("Inspection Summary")

                    st.metric(
                        "Total Defects Detected",
                        len(detections)
                    )

                    if detections:

                        detection_df = pd.DataFrame(detections)

                        st.dataframe(
                            detection_df,
                            use_container_width=True
                        )

                        # Class distribution
                        class_counts = (
                            detection_df["Defect"]
                            .value_counts()
                            .reset_index()
                        )

                        class_counts.columns = [
                            "Defect",
                            "Count"
                        ]

                        st.subheader("Detected Defect Distribution")

                        fig = px.bar(
                            class_counts,
                            x="Defect",
                            y="Count",
                            title="Number of Detected Defects"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # Download detection report
                        st.download_button(
                            "Download Detection Report",
                            detection_df.to_csv(index=False),
                            "yolo_detection_report.csv",
                            "text/csv"
                        )

                    else:

                        st.info(
                            "No defects detected above the selected "
                            "confidence threshold."
                        )

                except Exception as e:

                    st.error(
                        f"YOLO Detection Failed: {e}"
                    )


# --------------------------------------------------
# SEGMENTATION
# --------------------------------------------------

elif page == "Segmentation":

    st.subheader("Defect Segmentation")

    st.warning("""
    Pixel-level segmentation is currently pending.

    The current NEU-DET dataset provides bounding-box annotations,
    not pixel-level segmentation masks.
    """)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Model Performance Analysis")

    results_path = project_path(
        cfg["results_dir"],
        cfg
    )

    history_files = list(
        results_path.glob("*_history.json")
    )

    if history_files:

        selected_history = st.selectbox(
            "Select Training History",
            [file.name for file in history_files]
        )

        history_path = results_path / selected_history

        with open(history_path, "r") as f:
            history_data = json.load(f)

        df = pd.DataFrame(history_data)

        if "epoch" in df.columns:
            df = df.set_index("epoch")

        if {"train_loss", "val_loss"}.issubset(df.columns):

            st.subheader("Training and Validation Loss")

            st.line_chart(
                df[["train_loss", "val_loss"]]
            )

        if {
            "train_accuracy",
            "val_accuracy"
        }.issubset(df.columns):

            st.subheader("Training and Validation Accuracy")

            st.line_chart(
                df[["train_accuracy", "val_accuracy"]]
            )

    else:

        st.info(
            "No training history found. "
            "Train a model to populate performance graphs."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SurfaceIQ | Industrial Surface Defect Inspection Platform "
    "| AI-assisted inspection with human verification"
)
