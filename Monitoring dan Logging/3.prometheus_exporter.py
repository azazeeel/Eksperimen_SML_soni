from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# =====================================================================
# 1. PERSIAPAN MODEL NYATA (Bukan Simulasi)
# =====================================================================
print("Memuat dan menyiapkan model Machine Learning...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Melatih model ringan sebagai backend prediksi nyata
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)
print("Model siap melayani request API!")

# =====================================================================
# 2. DEFINISI 5 METRIK NYATA (Memenuhi Syarat Skilled)
# =====================================================================
REQUEST_COUNT = Counter('inference_requests_total', 'Total request ke endpoint')
ERROR_COUNT = Counter('inference_errors_total', 'Total request yang gagal/error')
LATENCY = Histogram('inference_latency_seconds', 'Waktu komputasi nyata prediksi')
PREDICT_MALIGNANT = Counter('inference_malignant_total', 'Total prediksi Ganas (Malignant)')
PREDICT_BENIGN = Counter('inference_benign_total', 'Total prediksi Jinak (Benign)')

# =====================================================================
# 3. ENDPOINT INFERENCE
# =====================================================================
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        # Menerima data asli dari request POST
        json_data = request.get_json()
        if not json_data or 'features' not in json_data:
            raise ValueError("Data fitur 'features' tidak ditemukan pada request")
        
        # Melakukan prediksi nyata dengan model
        features_df = pd.DataFrame([json_data['features']], columns=data.feature_names)
        prediction_result = model.predict(features_df)[0]
        
        # Mencatat hasil prediksi nyata ke Prometheus
        if prediction_result == 0:
            PREDICT_MALIGNANT.inc()
            label = "Malignant"
        else:
            PREDICT_BENIGN.inc()
            label = "Benign"

        # Mengukur latensi komputasi nyata (tanpa time.sleep)
        latency = time.time() - start_time
        LATENCY.observe(latency)

        return jsonify({
            "prediction": label,
            "latency_seconds": latency,
            "status": "success"
        })

    except Exception as e:
        # Jika terjadi error (misal format data salah), catat sebagai error
        ERROR_COUNT.inc()
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)