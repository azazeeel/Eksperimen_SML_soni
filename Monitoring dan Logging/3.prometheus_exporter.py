from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = Flask(__name__)

# --- 5 METRIK UNTUK LEVEL SKILLED ---
# 1. Total Requests (Counter)
REQUEST_COUNT = Counter('inference_requests_total', 'Total prediksi model')
# 2. Latency/Waktu Respon (Histogram)
LATENCY = Histogram('inference_latency_seconds', 'Waktu latensi prediksi')
# 3. Total Error (Counter) - Menghitung request yang gagal
ERROR_COUNT = Counter('inference_errors_total', 'Total prediksi error/gagal')
# 4. Akurasi Model (Gauge) - Simulasi fluktuasi akurasi model
ACCURACY_ESTIMATE = Gauge('model_accuracy_estimate', 'Estimasi akurasi model saat ini')
# 5. Request Aktif (Gauge) - Request yang sedang diproses detik ini
ACTIVE_REQUESTS = Gauge('active_requests', 'Jumlah request yang sedang berjalan')

@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc() # Tambah request aktif
    start = time.time()
    REQUEST_COUNT.inc()
    
    time.sleep(random.uniform(0.1, 0.4)) # Simulasi latensi AI
    
    # Simulasi kadang-kadang terjadi error (10% peluang gagal)
    if random.random() < 0.1:
        ERROR_COUNT.inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"status": "error", "message": "Simulasi gagal memproses data"}), 500
    
    # Simulasi fluktuasi akurasi (berubah-ubah antara 85% - 99%)
    ACCURACY_ESTIMATE.set(random.uniform(0.85, 0.99))
    
    latency = time.time() - start
    LATENCY.observe(latency)
    ACTIVE_REQUESTS.dec() # Kurangi request aktif karena sudah selesai
    
    return jsonify({"prediction": random.choice(["Malignant", "Benign"]), "status": "success"})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)