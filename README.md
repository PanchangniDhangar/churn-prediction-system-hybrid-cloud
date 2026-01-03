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

## System Architecture & Data Flow

The system follows a modular request-response lifecycle:
1. **User Interface:** The Streamlit dashboard collects 10 user-defined behavioral inputs.
2. **Data Orchestration:** The frontend sends a JSON payload to the FastAPI backend.
3. **Reference Merging:** The backend merges user inputs with a stored "reference profile" (test.csv) to create a complete 28-feature vector.
4. **Transformation:** The full vector is processed through the QuantileTransformer and StandardScaler saved during training.
5. **Inference:** The XGBoost model calculates the churn probability.
6. **Response:** The API returns the churn status (Churn/Not Churn) and confidence scores back to the dashboard.

##  TeckStack 
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

****Feature Engineering & Insights****
The model utilizes 28 high-impact features derived from raw telecom usage data. Key behavioral signals include:
Usage Decay: change_mou (Percentage drop in minutes) as a leading indicator of churn.
Equipment Lifecycle: eqpdays (Age of device) correlating with upgrade-driven attrition.
Bill Shock: ovrmou_Mean (Overage minutes) causing sudden customer dissatisfaction.
Normalization: Implemented QuantileTransformer to handle extreme skewness in usage metrics.

****Installation & Local Execution****
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
## Local Development

### Clone and Environment Setup
```bash
git clone [https://github.com/YOUR_USERNAME/churn-prediction-system-hybrid-cloud.git](https://github.com/YOUR_USERNAME/churn-prediction-system-hybrid-cloud.git)
cd churn-prediction-system-hybrid-cloud
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

****Testing & Quality Assurance****
Unit Testing: Implemented via the tests/ directory to validate model loading and API endpoint connectivity.
Data Validation: Utilized Pydantic models in app/schemas.py to prevent malformed data from reaching the inference pipeline.
Logging: Integrated a centralized logging system (src/logger.py) to track API requests, model predictions, and system errors in real-time.
Reproducibility: Used setup.py and requirements.txt to ensure environment consistency across different deployment platforms.

****Model Performance & Impact****
**Technical Metrics**
Model: XGBoost Classifier.
Key Indicators: Focused on ROC-AUC and Recall to ensure high detection rates of at-risk customers.
Top Predictors: change_mou (Usage decay), eqpdays (Handset age), and ovrmou_Mean (Overage billing).

**Business Impact**
Revenue Protection: By identifying "silent churners" (those with declining usage), the company can intervene before the customer cancels the service.
Resource Optimization: Enables the marketing team to target only high-risk/high-value customers with retention offers, reducing campaign costs.
Handset Lifecycle Management: Predicts when customers are likely to leave for an upgrade, allowing for proactive device upgrade offers.

****Summary****
This project demonstrates a complete MLOps lifecycle—from exploratory data analysis and modular pipeline development to containerization and hybrid-cloud deployment. By decoupling the inference engine from the user interface, the system achieves the flexibility required for modern enterprise applications.

**Author**
Panchangni Dhangar
* [LinkedIn](https://www.linkedin.com/in/panchangni-dhangar/)

