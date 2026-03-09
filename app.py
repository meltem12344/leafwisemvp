import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Model Mimarisini Tanımlama (Senin dosyana göre uyarlandı)
def load_model(model_path):
    model = models.mobilenet_v2(weights=None)
    # Senin modelindeki özel classifier katmanı:
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 256), # Ara katman
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 10)    # 10 Sınıf (Domates hastalıkları)
    )
    
    # Modeli yükle
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# 2. Sınıf İsimleri 
# Modelin çıktılarıyla tam eşleşen liste
class_names = [
    "Tomato___Bacterial_spot",                        # 0
    "Tomato___Early_blight",                          # 1
    "Tomato___healthy",                               # 2  <- Sağlıklı olan artık burada!
    "Tomato___Late_blight",                           # 3
    "Tomato___Leaf_Mold",                             # 4
    "Tomato___Septoria_leaf_spot",                    # 5
    "Tomato___Spider_mites Two-spotted_spider_mite",  # 6
    "Tomato___Target_Spot",                           # 7
    "Tomato___Tomato_mosaic_virus",                   # 8
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"          # 9
]

# 3. Streamlit Arayüzü
st.set_page_config(page_title="Leafwise - Domates Hastalık Teşhisi", page_icon="🍅")
st.title("🍅 Leafwise Projesi Test Paneli")
st.write("MobileNet modelini test etmek için bir domates yaprağı fotoğrafı yükleyin.")

model = load_model("best_model (1).pth")

uploaded_file = st.file_uploader("Bir fotoğraf seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Fotoğraf', use_column_width=True)
    
    # Görsel Ön İşleme
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Tahmin
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, index = torch.max(probabilities, 0)

    st.subheader(f"Tahmin: **{class_names[index]}**")
    st.write(f"Güven Oranı: %{conf.item()*100:.2f}")

    # Alt kısma hastalıkla ilgili bilgi alanı (LLM entegrasyonu buraya gelecek)
    if class_names[index] != "Sağlıklı":
        st.info(f"💡 **Öneri:** {class_names[index]} için topladığın kaynaklardaki organik çözümleri buraya ekleyebiliriz.")