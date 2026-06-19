# ANPDRS — Automatic Number Plate Detection & Recognition System

## Folder Structure (Final)
```
number plate detection system/
├── dataset/
│   ├── images/          ← .jpg/.png files from Kaggle
│   ├── annotations/     ← .xml files from Kaggle
│   └── yolo/            ← AUTO-CREATED by convert_annotations.py
│       ├── train/images/
│       ├── train/labels/
│       ├── val/images/
│       ├── val/labels/
│       ├── test/images/
│       ├── test/labels/
│       └── plates.yaml
├── runs/                ← AUTO-CREATED during training
│   └── detect/yolov8_plates/weights/best.pt
├── venv/
├── convert_annotations.py
├── train_yolo.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup Steps (Do in Order)

### Step 1 — Install Tesseract (Windows)
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or it auto-detects

### Step 2 — Activate venv
```bash
venv\Scripts\activate
```

### Step 3 — Install PyTorch with CUDA
Check your CUDA version first:
```bash
nvidia-smi
```
Then install matching PyTorch:
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Install other packages
```bash
pip install -r requirements.txt
```

### Step 5 — Convert annotations
```bash
python convert_annotations.py
```
You should see: "YOLO dataset ready at: dataset/yolo"

### Step 6 — Train YOLOv8
```bash
python train_yolo.py
```
Training takes ~15-30 minutes on RTX 2050.
Best weights saved at: `runs/detect/yolov8_plates/weights/best.pt`

### Step 7 — Run Streamlit App
```bash
streamlit run app.py
```
Opens in browser at: http://localhost:8501

---

## Troubleshooting

**CUDA out of memory:**
- Open train_yolo.py → change `batch=8` to `batch=4`

**Tesseract not found error:**
- Add this line at top of app.py:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

**Image not found during conversion:**
- Make sure images and XML files have matching names
- e.g., `Cars0.png` and `Cars0.xml`