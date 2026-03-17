import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Titel
st.title("Kühlschrank-Objekterkennung (YOLOv8)")

# Modell laden (einmalig cached auf Streamlit Cloud)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # vortrainiertes leichtes YOLOv8 Modell

model = load_model()

# Bild hochladen
uploaded_file = st.file_uploader("Bild deines Kühlschranks hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    img_array = np.array(image)

    # Objekterkennung
    results = model(img_array)[0]

    detections = []

    for box in results.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]
        if conf > 0.5:
            detections.append((label, conf))

    # Nach Wahrscheinlichkeit sortieren
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # Ergebnis anzeigen
    st.subheader("Erkannte Objekte")
    if detections:
        for label, conf in detections:
            st.write(f"{label} – {int(conf*100)}%")
    else:
        st.write("Keine Objekte erkannt (Confidence > 0.5)")

    # Bild mit Bounding Boxes anzeigen
    st.subheader("Visualisierung")
    plotted_image = results.plot()
    st.image(plotted_image, caption="Erkannte Objekte", use_column_width=True)
