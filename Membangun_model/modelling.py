import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    # Jika mengejar poin Advanced, hubungkan dengan DagsHub URI di bawah ini
    # mlflow.set_tracking_uri("https://dagshub.com/username/repo.mlflow")
    mlflow.set_experiment("Breast_Cancer_Experiment")
    
    train_df = pd.read_csv('namadataset_preprocessing/train.csv')
    test_df = pd.read_csv('namadataset_preprocessing/test.csv')
    
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_test, y_test = test_df.drop('target', axis=1), test_df['target']
    
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
        print(f"Model berhasil dilatih. Akurasi: {acc}")

if __name__ == "__main__":
    train()