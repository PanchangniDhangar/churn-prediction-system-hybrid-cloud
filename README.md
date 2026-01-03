#  Churn Prediction System: Hybrid-Cloud MLOps Pipeline
Developed a production-grade end-to-end Machine Learning system to predict customer attrition in the telecom sector. Unlike static research projects, this system implements a decoupled, API-first architecture, separating high-performance model inference from the user interface to ensure scalability and industrial reliability.
The system is designed to handle real-world challenges such as highly skewed usage data, class imbalance, and the need for real-time inference using a "Hybrid Cloud" deployment strategy.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-23aaff?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## Live Deployment Links
* **Interactive Dashboard:** [Paste your Streamlit Cloud URL here]
* **API Documentation (Swagger UI):** [Paste your Render URL here]/docs

##  Technical Stack & Architecture
* **Backend:** FastAPI with Pydantic for data validation and high-concurrency serving.
* **Frontend:** Streamlit for real-time interactive dashboards and "What-If" analysis.
* **ML Engine:** XGBoost Classifier optimized for behavioral signals.
* **Data Pipeline:** Modular structure for Ingestion, Transformation (QuantileTransformer), and Training.
* **Deployment:** Hybrid Cloud Strategy (Render for API, Streamlit Cloud for UI).

---

##  Project Structure
```text
├── app/                     # FastAPI Backend logic & Schemas
├── frontend/                # Streamlit Dashboard UI
├── src/                     # Modular MLOps components
│   ├── components/          # Data Ingestion, Transformation, Trainer
│   ├── pipeline/            # Train & Predict logic
│   └── logger.py            # Custom logging system
├── artifacts/               # Serialized models (.pkl) and test data
├── notebooks/               # EDA and Feature Engineering research
├── Dockerfile               # Containerization instructions
├── requirements.txt         # Project dependencies
└── setup.py                 # Project packaging
```

Feature Engineering & Insights
The model utilizes 28 high-impact features derived from raw telecom usage data. Key behavioral signals include:

Usage Decay: change_mou (Percentage drop in minutes) as a leading indicator of churn.

Equipment Lifecycle: eqpdays (Age of device) correlating with upgrade-driven attrition.

Bill Shock: ovrmou_Mean (Overage minutes) causing sudden customer dissatisfaction.

Normalization: Implemented QuantileTransformer to handle extreme skewness in usage metrics.

Installation & Local Execution
1. Clone the Repository
```
git clone [https://github.com/YOUR_USERNAME/churn-prediction-system-hybrid-cloud.git](https://github.com/YOUR_USERNAME/churn-prediction-system-hybrid-cloud.git)
cd churn-prediction-system-hybrid-cloud
```
2. Setup Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Run the System
You need to open two terminals:
Terminal 1 (Backend API):
```
python -m uvicorn app.main:app --reload
```
Terminal 2 (Frontend UI):
```
python -m streamlit run frontend/streamlit_app.py
```
Docker Deployment
To run the entire system using Docker:
```
docker-compose up --build
```
Executive Summary (Insights)
Proactive Retention: The system identifies "silent churners" (usage decay) before they cancel, allowing for targeted loyalty offers.

Scalability: By decoupling the API, the model can be integrated into existing CRM tools or mobile apps.

Reliability: Automated Pydantic validation prevents malformed data from reaching the inference engine.

Author
Panchangni Dhangar
* [LinkedIn](https://www.linkedin.com/in/panchangni-dhangar/)
