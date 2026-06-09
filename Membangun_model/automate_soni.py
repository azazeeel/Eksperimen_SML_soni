import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    print("Memulai otomatisasi preprocessing...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Re-build DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    
    # Save files
    out_dir = "namadataset_preprocessing"
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    print(f"File preprocessing selesai dan disimpan di folder: {out_dir}/")

if __name__ == "__main__":
    main()