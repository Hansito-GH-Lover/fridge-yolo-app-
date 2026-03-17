import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Seite konfigurieren ---
st.set_page_config(page_title="Kühlschrank-Objekterkennung", layout="wide")
st.markdown("""
<style>
body {
    background-color: #f0f8ff;  /* Kühlschrank-Blau */
    font-family: 'Arial', sans-serif;
}
h1, h2, h3 {
    color: #2f4f4f;
}
</style>
""", unsafe_allow_html=True)

st.title("🧊 Kühlschrank-Objekterkennung (YOLOv8)")

# --- Sidebar ---
st.sidebar.header("Einstellungen")
uploaded_file = st.sidebar.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- Modell laden ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # leichtes vortrainiertes Modell

model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # --- YOLO Prediction ---
    results = model(img_array)[0]

    # --- PIL Drawing ---
    draw = ImageDraw.Draw(image)
    colors = ["red", "green", "blue", "orange", "purple", "yellow"]
    detections = []

    for i, box in enumerate(results.boxes):
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]
        if conf >= confidence_threshold:
            detections.append((label, conf))
            x1, y1, x2, y2 = box.xyxy[0]
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 12), f"{label} {int(conf*100)}%", fill=color)

    # --- Sort detections ---
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # --- Mini-Karten für Objekte ---
    st.subheader("Erkannte Objekte")
    if detections:
        cols = st.columns(len(detections))
        for col, (label, conf) in zip(cols, detections):
            emoji = ""
            # Emoji-Zuweisung für Lebensmittel
            if label.lower() in ["apple", "banana", "orange"]: emoji = "🍎"
            elif label.lower() in ["bottle", "milk"]: emoji = "🥛"
            elif label.lower() in ["carrot", "broccoli"]: emoji = "🥕"
            col.metric(f"{emoji} {label}", f"{int(conf*100)}%")
    else:
        st.write(f"Keine Objekte erkannt (Confidence >= {int(confidence_threshold*100)}%)")

    # --- Visualisierung ---
    st.subheader("Visualisierung")
    st.image(image, use_column_width=True)

    # --- Download Button ---
    st.download_button(
        label="Annotiertes Bild herunterladen",
        data=image.tobytes(),
        file_name="annotated.png",
        mime="image/png"
    )
