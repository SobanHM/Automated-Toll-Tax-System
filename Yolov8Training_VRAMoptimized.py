"""
Train YOLOv8 on the Merged Vehicle Dataset (car, truck, van, bus)
Supports checkpoint saving, resume, and VRAM optimization for limited GPU memory.
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from config import YOLOV8_MODEL_DIR, ensure_directories

def train(
    data_path: Path,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    model_size: str = "s",
    patience: int = 10,
    device: str | None = None,
    resume: bool = True,
    fp16: bool = True,
):
    """
    Train YOLOv8 on a dataset with checkpoints and VRAM optimization.
    """
    ensure_directories()

    # Adjust batch size automatically based on GPU memory
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        try:
            free_mem = torch.cuda.get_device_properties(0).total_memory
            # Reduce batch if VRAM is small
            if free_mem <= 8 * 1024**3:  # <= 8GB
                batch = min(batch, 8)
        except Exception:
            pass

    print(f"Using device: {device}, batch size: {batch}, FP16: {fp16}")

    # Initialize model
    weights = f"yolov8{model_size}.pt"
    model = YOLO(weights)

    # Define project/run
    run_name = f"toll_yolov8{model_size}"

    # Resume if checkpoint exists
    resume_path = Path(YOLOV8_MODEL_DIR) / run_name / "weights" / "last.pt"
    if resume and resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        model = YOLO(str(resume_path))

    # Train model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        project=str(YOLOV8_MODEL_DIR),
        name=run_name,
        verbose=True,
        device=device,
        resume=resume,
        half=fp16  # enable mixed precision
    )

    # Copy best weights to stable path
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = Path(YOLOV8_MODEL_DIR) / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"Best model copied to: {best_dst}")
    else:
        print(f"Could not find best weights at {best_src}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Toll Vehicle Dataset with VRAM optimization")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model-size", type=str, default="s", help="YOLOv8 size (n,s,m,l,x)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., 'cpu'")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision to save VRAM")
    return parser.parse_args()


def cli():
    args = parse_args()
    data_yaml_path = Path(args.data_path)

    # Train YOLOv8
    train(
        data_path=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model_size=args.model_size,
        patience=args.patience,
        device=args.device,
        resume=args.resume,
        fp16=args.fp16
    )


if __name__ == "__main__":
    cli()
