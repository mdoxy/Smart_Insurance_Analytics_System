# Smart_Insurance_Analytics_System
An analytics platform for insurance claim prediction and data reporting. The system preprocesses and cleans insurance data using PySpark, stores it in SQL, and exposes a web interface where users can upload claim data, view dashboards, and interact with an AI assistant that provides insights.

## Project Structure
- `data/insurance_claims.csv` - synthetic dataset
- `scripts/etl.py` - ETL script (pandas)
- `scripts/train_model.py` - trains a RandomForest model and saves it to `models/`
- `app/app.py` - Flask app to serve predictions
- `app/templates/index.html` - simple UI to test predictions
- `requirements.txt` - Python dependencies

## How to run locally
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run ETL:
   ```bash
   python scripts/etl.py
   ```
3. Train model:
   ```bash
   python scripts/train_model.py
   ```
4. Start the Flask app:
   ```bash
   python app/app.py
   ```
5. Open `http://localhost:5000` to use the demo UI.

## Notes and next steps
- Replace pandas ETL with PySpark notebooks for Databricks integration.
- Add Databricks notebooks and schedule jobs for production pipelines.
- Integrate an AI assistant (LangChain/OpenAI) to query the analytics dataset in natural language.
- Deploy to Azure App Service or similar for production use.
