import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train():
    # 1. BARIS PENYELAMAT (Memaksa MLflow menulis langsung ke folder lokal, mengabaikan error server 500)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Breast_Cancer_Experiment")
    
    # 2. AUTOLOG SCIKIT-LEARN (Menjamin Artifacts pasti muncul)
    mlflow.sklearn.autolog(
        log_models=True, 
        log_input_examples=True, 
        log_model_signatures=True
    )
    
    # Load dataset
    train_df = pd.read_csv('namadataset_preprocessing/train.csv')
    test_df = pd.read_csv('namadataset_preprocessing/test.csv')
    
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_test, y_test = test_df.drop('target', axis=1), test_df['target']
    
    with mlflow.start_run(run_name="Basic_Training_Autolog"):
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        # Fit akan memicu pembuatan artifacts
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        print(f"Model berhasil dilatih menggunakan SKLEARN AUTOLOG LENGKAP. Akurasi: {score}")

if __name__ == "__main__":
    train()