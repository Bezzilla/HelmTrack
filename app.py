import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
from collections import Counter

API_URL = "http://127.0.0.1:8000/predict"

COLOR_MAP = {
    "Helmet":    (0, 200, 0),
    "NoHelmet":  (220, 0, 0),
    "Motorbike": (255, 165, 0),
    "PNumber":   (180, 0, 220),
}

REASON_MAP = {
    "Helmet":    "Rider is wearing a helmet — no violation.",
    "NoHelmet":  "Rider has NO helmet — VIOLATION flagged.",
    "Motorbike": "Motorcycle detected — used to associate rider.",
    "PNumber":   "License plate detected — candidate for OCR.",
}

st.set_page_config(page_title="HelmTrack", layout="wide")

st.title("HelmTrack Helmet Violation Detection")
st.caption("Upload a traffic image to detect helmet violations using YOLOv8.")

uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    img_bytes = uploaded.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    with col1:
        st.subheader("Original Image")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Running model..."):
        try:
            resp = requests.post(API_URL, files={"file": (uploaded.name, img_bytes, "image/jpeg")})
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API at http://127.0.0.1:8000 — make sure uvicorn is running.")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    violation = data["violation_detected"]
    detections = data["detections"]

    # Draw boxes on image
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cls   = d["class"]
        conf  = d["confidence"]
        color = COLOR_MAP.get(cls, (128, 128, 128))
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(img_cv, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

    img_out = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Model Detections")
        st.image(img_out, use_container_width=True)

    # Violation banner
    st.markdown("---")
    if not detections:
        st.info("No objects detected in this image. Try a clearer traffic photo.")
    elif violation:
        st.error("## VIOLATION DETECTED — Rider without helmet identified")
    else:
        st.success("## CLEAR — No helmet violation detected")

    # Detection table
    if detections:
        st.subheader("Detection Details")
        for d in detections:
            cls  = d["class"]
            conf = d["confidence"]
            conf_label = "high" if conf >= 0.75 else ("medium" if conf >= 0.5 else "low")
            reason = REASON_MAP.get(cls, "Unknown class.")
            color_hex = {
                "Helmet": "#00c800", "NoHelmet": "#dc0000",
                "Motorbike": "#ffa500", "PNumber": "#b400dc"
            }.get(cls, "#888888")

            col_a, col_b, col_c = st.columns([1, 1, 4])
            col_a.markdown(f"**:{color_hex}[{cls}]**")
            col_b.markdown(f"`{conf:.2f}` ({conf_label})")
            col_c.markdown(reason)

        # Summary counts
        st.markdown("---")
        st.subheader("Summary")
        counts = Counter(d["class"] for d in detections)
        cols = st.columns(len(counts))
        for i, (cls, cnt) in enumerate(counts.items()):
            cols[i].metric(cls, cnt)
