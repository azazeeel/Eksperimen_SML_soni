import requests
import time

url = "http://localhost:8000/predict"
print("Menjalankan request simulasi inference pipeline...")

while True:
    try:
        response = requests.post(url)
        print(f"Status: {response.status_code}, Respon: {response.json()}")
    except requests.exceptions.RequestException:
        print("Menunggu server API hidup...")
    time.sleep(2)