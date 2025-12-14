# Credit Risk Probability Model – Week 4

I am building an end-to-end credit scoring workflow for Bati Bank’s buy-now-pay-later partnership. The project follows a modular structure so I can iterate fast during the interim submission and add automation for the final hand-in.

## Repository Layout
```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Credit Scoring Business Understanding
- **Basel II and interpretability.** Basel II pushes banks to quantify risk exposure, document assumptions, and prove model governance. Because of that, I need a modelling stack that is auditable: clear data lineage, explainable features, and probability scores I can defend to credit committee stakeholders. Every transformation and hyperparameter choice has to be reproducible and traceable.
- **Why I need a proxy default label.** The eCommerce feed has no explicit “default”. If I do not engineer a proxy, the model cannot learn which customers fail to repay. I will create an RFM-derived high-risk label, but I am aware that this proxy might misclassify good customers who simply paused activity or new customers without history. Those mistakes can lead to lost sales or unexpected defaults, so I will keep monitoring drift and periodically recalibrate the proxy.
- **Interpretable vs. complex models.** Logistic Regression with Weight of Evidence gives me clean scorecards, easy cut-off management, and better regulatory comfort. Gradient Boosting can squeeze out extra performance, but it is harder to explain, requires more governance, and complicates stress testing. My plan is to start with an interpretable baseline for approval, then experiment with more complex learners and only promote them if the uplift justifies the added validation cost.

## Interim Deliverables (14 Dec 2025)
1. Business understanding captured in this README.
2. `notebooks/eda.ipynb` with structured exploratory analysis and 3–5 headline insights.

## Local Development
```bash
python3.11 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/eda.ipynb
```

Or spin up the lightweight container:
```bash
docker-compose up --build
```

## Next Steps
- Engineer RFM features and cluster customers to derive the high-risk proxy.
- Operationalise feature pipelines with scikit-learn, WoE/IV transformations, and MLflow tracking.
- Containerise the FastAPI scoring service and wire CI/CD before the final deadline.
