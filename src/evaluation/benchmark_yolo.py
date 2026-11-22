
import json
import time
import torch

from pathlib import Path
from ultralytics import YOLO

from src.config import load_config, project_path
from src.data.dataset import discover_images, CLASSES


def main():
    cfg = load_config()

    root = project_path(cfg["dataset"]["root"], cfg)
    weights = project_path("models/yolo_best.pt", cfg)
    output_dir = project_path(cfg["results_dir"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    records = [
        r for r in discover_images(root)
        if r["label"] in CLASSES
    ]

    if not records:
        raise RuntimeError("No labelled validation images found.")

    # Use up to 30 images for a quick benchmark.
    samples = records[:30]

    print(f"Loading YOLO model: {weights}")
    model = YOLO(str(weights))

    print("Running warm-up...")
    for record in samples[:5]:
        model.predict(
            source=record["path"],
            imgsz=320,
            device="cpu",
            verbose=False
        )

    print(f"Benchmarking {len(samples)} images...")

    timings = []

    for record in samples:
        start = time.perf_counter()

        model.predict(
            source=record["path"],
            imgsz=320,
            device="cpu",
            verbose=False
        )

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    total_time = sum(timings)
    average_latency = total_time / len(timings)

    results = {
        "model": "YOLOv8n",
        "weights": str(weights),
        "device": "CPU",
        "image_size": 320,
        "benchmark_images": len(samples),
        "total_time_seconds": round(total_time, 4),
        "average_latency_ms": round(average_latency * 1000, 2),
        "average_fps": round(1 / average_latency, 2),
        "measurement": "Single-image end-to-end prediction latency"
    }

    output_file = output_dir / "yolo_benchmark.json"

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\nYOLO BENCHMARK RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
