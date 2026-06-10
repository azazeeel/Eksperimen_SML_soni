import requests
import time

url = "http://localhost:8000/predict"
print("Menjalankan request simulasi inference pipeline (Menuju Level Skilled)...")

while True:
    try:
        response = requests.post(url)
        print(f"Status: {response.status_code}, Respon: {response.text}")
    except requests.exceptions.RequestException:
        print("Menunggu server API hidup...")
    time.sleep(1) # Kirim 1 request setiap detik