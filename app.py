import streamlit as st
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import cv2

@st.cache_resource
def get_model_path():
    return "best.pt"

def load_yolo_model():
    return YOLO(get_model_path())

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
if image:
    image_np = np.array(image)
    results = model(image_np)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} ({conf:.2f})"
            color = (0, 255, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    st.image(image_np, caption="Tahmin Sonucu", use_container_width=True)
