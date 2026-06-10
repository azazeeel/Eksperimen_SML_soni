from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = Flask(__name__)

# Definisi Metrik
REQUEST_COUNT = Counter('inference_requests_total', 'Total prediksi model')
LATENCY = Histogram('inference_latency_seconds', 'Waktu latensi prediksi')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    
    time.sleep(random.uniform(0.1, 0.4)) # Simulasi latensi AI
    
    latency = time.time() - start
    LATENCY.observe(latency)
    return jsonify({"prediction": random.choice(["Malignant", "Benign"]), "status": "success"})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)