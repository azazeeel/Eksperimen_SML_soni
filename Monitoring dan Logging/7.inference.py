import requests
import time
import random
from sklearn.datasets import load_breast_cancer

# Load dataset untuk diambil baris per baris secara acak
data = load_breast_cancer()
features_list = data.data.tolist()

url = "http://localhost:8000/predict"
print("Robot Inference berjalan: Mengirimkan data nyata ke API Endpoint...")

while True:
    try:
        # Mengambil 1 baris data fitur sungguhan secara acak dari dataset
        sample_features = random.choice(features_list)
        
        # Mengirim data tersebut sebagai JSON
        payload = {"features": sample_features}
        response = requests.post(url, json=payload)
        
        print(f"Status: {response.status_code}, Respon: {response.json()}")
        
    except requests.exceptions.RequestException:
        print("Menunggu server API hidup...")
        
    time.sleep(1) # Kirim 1 request setiap detik