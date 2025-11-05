from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = None
MODEL_PATH = 'models/claim_model.joblib'

def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df['gender'] = df['gender'].map({'M':0,'F':1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    # ensure missing region columns
    for col in ['region_East','region_North','region_South','region_West']:
        if col not in df.columns:
            df[col] = 0
    features = ['age','policy_tenure_months','claim_amount','num_previous_claims','gender','region_East','region_North','region_South','region_West']
    X = df[features]
    pred = model.predict(X)[0]
    return jsonify({'claim_approved': int(pred)})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
