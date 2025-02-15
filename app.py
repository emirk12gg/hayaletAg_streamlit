import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import torch

# Load YOLOv8 model with PyTorch deserialization fix
def load_yolo_model():
    try:
        # Allow loading of the full model (not just weights)
        with torch.serialization.safe_globals(["ultralytics.nn.tasks.DetectionModel"]):
            model = YOLO("best (1).pt")  # Ensure model file is correctly named
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_yolo_model()

st.title("Gerçek Zamanlı Nesne Tespiti (YOLOv8)")

uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

sample_images_path = "sample_images"
sample_images = os.listdir(sample_images_path) if os.path.exists(sample_images_path) else []

selected_sample = st.selectbox("Veya bir örnek görüntü seçin", [None] + sample_images)

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
elif selected_sample:
    image = Image.open(os.path.join(sample_images_path, selected_sample))
    st.image(image, caption="Örnek Görüntü", use_container_width=True)

if image and model:
    image_np = np.array(image)  # Convert image to NumPy array

    # Ensure image format is RGB
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Perform YOLO detection
    results = model(image_np)

    # Draw detection results using OpenCV
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index

            label = f"{model.names[cls]} ({conf:.2f})"
            color = (0, 255, 0)  # Green bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(image_np, caption="Tahmin Sonucu", use_container_width=True)
