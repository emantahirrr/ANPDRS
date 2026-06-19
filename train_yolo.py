"""
Step 2: Train YOLOv8 on license plate dataset
RTX 2050 4GB optimised settings.
"""

from ultralytics import YOLO
import torch

def main():
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} — {vram_gb:.1f} GB VRAM")
    else:
        print("[WARN] CUDA not found, training on CPU (will be slow!)")

    # Load YOLOv8 nano (smallest = fits in 4 GB VRAM easily)
    model = YOLO("yolov8n.pt")   # auto-downloads ~6 MB weights

    # Train
    results = model.train(
        data       = "dataset/yolo/plates.yaml",
        epochs     = 60,
        imgsz      = 640,
        batch      = 8,           # safe for 4 GB VRAM; increase to 16 if no OOM
        workers    = 2,
        device     = 0,           # GPU 0
        patience   = 15,          # early stopping
        save       = True,
        project    = "runs/detect",
        name       = "yolov8_plates",
        exist_ok   = True,
        # Augmentation (helps with small dataset)
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 1.0,
        degrees    = 5.0,
        translate  = 0.1,
        scale      = 0.5,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
    )

    print("\n[DONE] Training complete!")
    print(f"Best weights saved at: runs/detect/yolov8_plates/weights/best.pt")

    # Quick validation
    metrics = model.val()
    print(f"\n[RESULTS]")
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()