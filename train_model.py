import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

def train(input_path='data/clean_insurance_claims.csv', model_path='models/claim_model.joblib'):
    df = pd.read_csv(input_path)
    # Simple preprocessing
    df = df.copy()
    df['gender'] = df['gender'].map({'M':0,'F':1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    features = ['age','policy_tenure_months','claim_amount','num_previous_claims','gender'] + [c for c in df.columns if c.startswith('region_')]
    X = df[features]
    y = df['claim_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    train()
