import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune():
    mlflow.set_experiment("Breast_Cancer_Tuning")
    train_df = pd.read_csv('namadataset_preprocessing/train.csv')
    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    
    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(clf, param_grid, cv=3)
        grid.fit(X_train, y_train)
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_score", grid.best_score_)
        mlflow.sklearn.log_model(grid.best_estimator_, "best_model")
        print(f"Tuning berhasil. Parameter terbaik: {grid.best_params_}")

if __name__ == "__main__":
    tune()