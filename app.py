import streamlit as st
from tf_keras.models import load_model
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_modell():
    model = load_model(r"model_tf.h5")  # Model dosyanın adını değiştir
    return model

model = load_modell()

CLASS_NAMES = ["Hayalet Ağ", "Deniz Çöpü"] 

def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = np.expand_dims(image, axis=0) 
    return image

st.title("Gerçek Zamanlı Görüntü Sınıflandırma")

uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

sample_images_path = "sample_images"
sample_images = os.listdir(sample_images_path) if os.path.exists(sample_images_path) else []

selected_sample = st.selectbox("Veya bir örnek görüntü seçin", [None] + sample_images)

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
elif selected_sample:
    image = Image.open(os.path.join(sample_images_path, selected_sample))
    st.image(image, caption="Örnek Görüntü", use_column_width=True)

if image:
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    # 0-Hayalet Ağ
    #  1- Deniz çöp
    index = np.round(predictions)

    st.write(f"**Tahmin:**{'Hayalet Ağ Algılandı' if index==0 else 'Deniz Çöpü Algılandı'}"),
