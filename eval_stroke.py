import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    # 1. Load real test data and your new synthetic data
    # Checking for standard TabDiff preprocessed data layouts
    real_test = pd.read_csv('data/stroke/test.csv')
    synth_train = pd.read_csv('synthetic/stroke/test.csv')
    
    target = 'stroke'
    
    # Clean up identifiers if they exist
    for df in [real_test, synth_train]:
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
            
    # 2. Separate features and target labels
    X_synth = synth_train.drop(columns=[target])
    y_synth = synth_train[target].astype(int)
    
    X_real_test = real_test.drop(columns=[target])
    y_real_test = real_test[target].astype(int)
    
    # 3. Align categorical string columns using One-Hot Encoding
    X_combined = pd.concat([X_synth, X_real_test], axis=0)
    X_combined_encoded = pd.get_dummies(X_combined, drop_first=True)
    
    X_synth_encoded = X_combined_encoded.iloc[:len(X_synth)]
    X_real_test_encoded = X_combined_encoded.iloc[len(X_synth):]
    
    # 4. Train downstream classifier on synthetic data, test on real data (TSTR)
    print("Training downstream classifier on synthetic data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_synth_encoded, y_synth)
    
    # 5. Compute performance evaluations
    preds = clf.predict(X_real_test_encoded)
    
    print("\n=========================================")
    print(f"Stroke Dataset Downstream MLE Results:")
    print(f"Accuracy:       {accuracy_score(y_real_test, preds):.4f}")
    print(f"Macro F1-Score: {f1_score(y_real_test, preds, average='macro'):.4f}")
    print("=========================================\n")

except Exception as e:
    print(f"Error running evaluation: {e}")
    print("Please double check that 'data/stroke/test.csv' and 'synthetic/stroke/test.csv' are present.")
