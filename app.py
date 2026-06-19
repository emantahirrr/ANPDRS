import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import re
import time
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANPDRS",  
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a2e; text-align: center;
    }
    .subtitle {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .plate-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 3px solid #f0c040;
        border-radius: 10px; padding: 18px;
        text-align: center; margin-top: 12px;
    }
    .plate-text {
        font-size: 2.2rem; font-weight: 900;
        letter-spacing: 8px; color: #f0c040;
        font-family: 'Courier New', monospace;
    }
    .conf-badge {
        display:inline-block; padding:3px 10px;
        border-radius:20px; font-size:0.8rem;
        font-weight:600; margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─── MODEL PATHS ──────────────────────────────────────────────────────────────
WEIGHTS_PATH = r"D:\University\projects\number plate detection system\runs\detect\runs\detect\yolov8_plates\weights\best.pt"
FALLBACK_PATH = "yolov8n.pt"

@st.cache_resource
def load_yolo():
    if Path(WEIGHTS_PATH).exists():
        m = YOLO(WEIGHTS_PATH)
        return m, "✅ Fine-tuned Model (best.pt)"
    m = YOLO(FALLBACK_PATH)
    return m, "Base YOLOv8n "

@st.cache_resource
def load_ocr():
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False
    return easyocr.Reader(['en'], gpu=use_gpu, verbose=False)


# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def keep_best_plate_box(boxes):
    """
    
    """
    if len(boxes) <= 1:
        return boxes

    scored = []
    for box in boxes:
        conf      = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            continue
        aspect = w / h   # license plate ~2.0–5.0

        # Score: high confidence + plate-like aspect ratio
        ar_score = 1.0 if 1.5 <= aspect <= 6.0 else 0.3
        score    = conf * ar_score
        scored.append((score, box))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Return only top 1 (best plate)
    return [scored[0][1]]


def preprocess_for_ocr(crop_bgr):
    """
    
    """
    h, w = crop_bgr.shape[:2]

    # Upscale if too small (EasyOCR struggles below 30px height)
    if h < 50:
        scale    = 100 / h
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

    # For display: threshold image (looks nice in UI)
    gray_disp = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    thresh    = cv2.adaptiveThreshold(gray_disp, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2)

    # For OCR: CLAHE enhanced color image (EasyOCR reads this better)
    lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return thresh, enhanced_bgr


def run_ocr(reader, crop_bgr, thresh):
    """
    """
    candidates = []

    attempts = [
        crop_bgr,                                    # color
        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY),  # grayscale
        thresh,                                       # threshold (last resort)
    ]

    for img in attempts:
        try:
            results = reader.readtext(img, detail=1, paragraph=False,
                                      allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
            for (_, text, conf) in results:
                cleaned = re.sub(r'[^A-Z0-9\-]', '', text.upper())
                if len(cleaned) >= 2:
                    candidates.append((cleaned, conf))
        except Exception:
            continue

    if not candidates:
        return "Unreadable", 0.0

    # Pick highest confidence result
    best_text, best_conf = max(candidates, key=lambda x: x[1])
    return best_text, round(best_conf, 3)


def detect_and_recognize(image_bgr, yolo_model, ocr_reader, conf_thresh):
    """
    Full pipeline with all fixes applied.
    """
    preds     = yolo_model.predict(source=image_bgr, conf=conf_thresh,
                                   iou=0.4, verbose=False)
    annotated = image_bgr.copy()
    detections = []

    for result in preds:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # FIX 1: keep only the best plate box
        best_boxes = keep_best_plate_box(list(result.boxes))

        for box in best_boxes:
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            bw = x2 - x1
            bh = y2 - y1

            # Skip boxes with terrible aspect ratio (not a plate)
            if bh == 0 or not (1.2 <= bw/bh <= 7.0):
                continue

            # Draw clean box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 100), 3)
            label = f"Plate  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), (50, 220, 100), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            # Crop with generous padding
            pad  = 8
            crop = image_bgr[max(0,y1-pad):min(image_bgr.shape[0],y2+pad),
                             max(0,x1-pad):min(image_bgr.shape[1],x2+pad)]
            if crop.size == 0:
                continue

            thresh_disp, crop_enhanced = preprocess_for_ocr(crop)
            plate_text, ocr_conf       = run_ocr(ocr_reader, crop_enhanced, thresh_disp)

            detections.append({
                "text"       : plate_text,
                "det_conf"   : conf,
                "ocr_conf"   : ocr_conf,
                "bbox"       : (x1, y1, x2, y2),
                "crop_rgb"   : cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                "thresh"     : thresh_disp,
            })

    return annotated, detections


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    conf_thresh = st.slider(
        "Detection Confidence", 0.10, 0.90, 0.30, 0.05,
        help="0.30 recommended. Lower = more detections but more false positives."
    )

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
**Models:**
- 🔵 YOLOv8n — Plate Detection
- 🟢 EasyOCR  — Character Recognition


    """)

    st.markdown("---")
    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"🖥️ {gpu}")
            st.caption(f"VRAM: {vram:.1f} GB")
        else:
            st.warning("CPU mode")
    except Exception:
        pass


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Automatic Number Plate Detection & Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ANN & Deep Learning — BS AI 6th Semester</div>', unsafe_allow_html=True)

# Load both models
c1, c2 = st.columns(2)
with c1:
    with st.spinner("Loading YOLO..."):
        yolo_model, model_label = load_yolo()
    st.info(f"YOLO: {model_label}")
with c2:
    with st.spinner("Loading EasyOCR (first run ~1 min)..."):
        ocr_reader = load_ocr()
    st.success("EasyOCR: ✅ Ready")

st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📂 Batch Processing", "📈 Model Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Image
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Upload Vehicle Image")
        uploaded = st.file_uploader("Choose an image",
                                    type=["jpg","jpeg","png","bmp"],
                                    key="single")
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_np  = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Input Image", use_container_width=True)
            detect_btn = st.button("🔍 Detect & Recognize Plate", type="primary")

    with right:
        st.markdown("### Results")
        if uploaded and 'detect_btn' in dir() and detect_btn:
            with st.spinner("Running detection & OCR..."):
                t0 = time.time()
                annotated_bgr, dets = detect_and_recognize(
                    img_bgr, yolo_model, ocr_reader, conf_thresh
                )
                elapsed = time.time() - t0

            ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption="Detection Output", use_container_width=True)
            st.caption(f"⏱️ {elapsed:.2f}s")

            if dets:
                for i, det in enumerate(dets):
                    st.markdown(f"---")
                    st.markdown(
                        f"**Plate {i+1}**  "
                        f"<span class='conf-badge' style='background:#1e3a5f;color:#5bc0f8'>"
                        f"Detection: {det['det_conf']:.2f}</span>  "
                        f"<span class='conf-badge' style='background:#1a3a1a;color:#5af07a'>"
                        f"OCR: {det['ocr_conf']:.2f}</span>",
                        unsafe_allow_html=True
                    )

                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.image(det['crop_rgb'],
                                 caption="Cropped Plate", use_container_width=True)
                    with cc2:
                        st.image(det['thresh'],
                                 caption="Preprocessed", use_container_width=True)

                    st.markdown(f"""
                    <div class="plate-box">
                        <div style="font-size:0.75rem;color:#aaa;letter-spacing:3px;
                                    margin-bottom:8px">NUMBER PLATE</div>
                        <div class="plate-text">{det['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No plate detected. Try lowering the confidence slider (e.g. 0.15).")

                # ── Fallback: run EasyOCR on whole image ──────────────────────
                st.markdown("#### 🔄 Fallback: OCR on full image")
                with st.spinner("Running full-image OCR..."):
                    fallback_results = ocr_reader.readtext(img_bgr, detail=1, paragraph=False)
                if fallback_results:
                    for (_, txt, c) in fallback_results:
                        cleaned = re.sub(r'[^A-Z0-9\-]', '', txt.upper())
                        if len(cleaned) >= 3:
                            st.code(f"{cleaned}  (conf: {c:.2f})")
                else:
                    st.info("No text found on full image either.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Processing
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Image Processing")
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key="batch"
    )

    if batch_files:
        if st.button("🚀 Process All", type="primary"):
            rows     = []
            prog     = st.progress(0)
            status   = st.empty()

            for idx, f in enumerate(batch_files):
                status.text(f"Processing {f.name}  ({idx+1}/{len(batch_files)})")
                pil  = Image.open(f).convert("RGB")
                bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                _, d = detect_and_recognize(bgr, yolo_model, ocr_reader, conf_thresh)

                if d:
                    for det in d:
                        rows.append({
                            "File"           : f.name,
                            "Plate"          : det["text"],
                            "Det Confidence" : f"{det['det_conf']:.2f}",
                            "OCR Confidence" : f"{det['ocr_conf']:.2f}",
                        })
                else:
                    rows.append({"File": f.name, "Plate": "Not detected",
                                 "Det Confidence": "-", "OCR Confidence": "-"})
                prog.progress((idx+1) / len(batch_files))

            status.success("✅ Done!")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                               "anpdrs_results.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Info
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Pipeline Architecture")

    steps = [
        ("📷","Input Image","Vehicle photo"),
        ("🎯","YOLOv8n","Plate detection"),
        ("✂️","Crop + CLAHE","Enhance contrast"),
        ("🔤","EasyOCR","Read characters"),
        ("✅","Output","Plate number"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, sub) in zip(cols, steps):
        col.markdown(f"""
        <div style='text-align:center;padding:12px;background:#f8f9fa;
                    border-radius:10px;margin:4px'>
            <div style='font-size:1.8rem'>{icon}</div>
            <div style='font-weight:700;font-size:0.95rem'>{title}</div>
            <div style='color:#888;font-size:0.78rem'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Evaluation Metrics")
    metrics = {
        "Metric"     : ["mAP@0.5","IoU","Precision","Recall","F1-Score"],
        "Task"       : ["Detection","Detection","OCR","OCR","OCR"],
        "Description": [
            "Mean Average Precision @ 50% overlap",
            "Intersection over Union of predicted vs actual box",
            "Correct detections / Total detections",
            "Correct detections / Total actual plates",
            "Harmonic mean of Precision & Recall",
        ],
        "Target"     : ["> 0.80","> 0.50","> 0.85","> 0.85","> 0.85"],
    }
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    # Training curves (show after training)
    results_csv = Path("runs/detect/yolov8_plates/results.csv")
    if results_csv.exists():
        st.markdown("### 📈 Training Results")
        df_r = pd.read_csv(results_csv)
        df_r.columns = [c.strip() for c in df_r.columns]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if "train/box_loss" in df_r.columns:
            axes[0].plot(df_r["train/box_loss"], label="Train", color="#3a86ff")
            if "val/box_loss" in df_r.columns:
                axes[0].plot(df_r["val/box_loss"], label="Val",   color="#ff6b6b")
            axes[0].set_title("Box Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        if "metrics/mAP50(B)" in df_r.columns:
            axes[1].plot(df_r["metrics/mAP50(B)"], color="#06d6a0", label="mAP@0.5")
            axes[1].set_title("mAP@0.5"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Training curves will appear here after `python train_yolo.py` completes.")