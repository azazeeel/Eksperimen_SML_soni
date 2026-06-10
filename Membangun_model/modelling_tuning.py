import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def tune():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Breast_Cancer_Experiment_Tuning")
    
    # 1. MEMATIKAN AUTOLOG (Memastikan logging 100% manual sesuai kriteria)
    mlflow.autolog(disable=True)
    
    # Load dataset
    train_df = pd.read_csv('namadataset_preprocessing/train.csv')
    test_df = pd.read_csv('namadataset_preprocessing/test.csv')
    
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_test, y_test = test_df.drop('target', axis=1), test_df['target']
    
    # Grid Search Parameters
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    
    with mlflow.start_run(run_name="Tuning_Manual_Logging"):
        clf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(clf, param_grid, cv=3)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        
        # 2. MENGHITUNG METRICS MANUAL (Setara dengan Autolog)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')
        
        # 3. MANUAL LOGGING PARAMETERS & METRICS
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("training_cv_score", grid.best_score_)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        
        # 4. MEMBUAT SIGNATURE & INPUT EXAMPLE MANUAL
        contoh_input = X_test.head(1) # Mengambil 1 baris teratas sebagai contoh
        signature = infer_signature(X_test, preds) # Memetakan tipe data
        
        # 5. MENYIMPAN MODEL BESERTA ARTIFACTS-NYA
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model",
            signature=signature,
            input_example=contoh_input
        )
        
        print(f"Tuning berhasil (Manual Log LENGKAP). Parameter: {grid.best_params_}")

if __name__ == "__main__":
    tune()