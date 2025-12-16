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

## Feature Engineering Overview
- **Customer roll-ups.** `src/data_processing.engineer_customer_features` aggregates raw transactions into customer-level summaries (total and average spend, variability, channel diversity, temporal preferences) while extracting datetime signals (hour/day/month).
- **RFM enrichment.** `compute_rfm` calculates recency, frequency, and monetary metrics; `assign_high_risk_labels` applies KMeans (k=3) to define the proxy delinquency label and exposes the dominant high-risk cluster.
- **WoE + IV transformation.** `WoEEncoder` replaces key categorical anchors (provider, channel, product) with Weight-of-Evidence scores and records Information Value to gauge predictive power.
- **scikit-learn Pipeline.** `build_preprocessing_pipeline` chains WoE encoding, imputation, scaling, and one-hot encoding, ensuring reproducible transformations for every model run.

## Proxy Target Construction
- RFM metrics are computed against a configurable snapshot date (default = last transaction + 1 day).
- KMeans (random_state=42) segments customers; the most disengaged cluster (high recency, low spend/frequency) maps to `is_high_risk=1`.
- The resulting dataset (features + proxy label) is materialised to `data/processed/model_training_dataset.parquet` during training for auditability.

## Model Training & Tracking
- Run `python -m src.train` (optionally with `--snapshot-date YYYY-MM-DD`) to trigger the full workflow.
- Two estimators are tuned via GridSearchCV (F1 scoring): Logistic Regression and Random Forest.
- Every run logs to MLflow (`mlruns/`), capturing parameters, metrics, feature columns, WoE/IV diagnostics, and the fitted pipeline.
- The best pipeline is registered as `credit-risk-probability-model` and auto-promoted to **Production** stage. A copy of the production pipeline is stored at `models/latest_model.joblib`.

## Serving the Model
- `docker compose up --build` exposes the FastAPI service on `http://localhost:8000` and keeps Jupyter’s port (8888) available for ad-hoc notebooks.
- `POST /predict` accepts a list of transaction records mirroring the raw schema and returns per-customer default probabilities. Example payload:

```json
{
	"snapshot_date": "2025-12-16",
	"transactions": [
		{
			"TransactionId": "sample-1",
			"CustomerId": "CUST-001",
			"Amount": 20000,
			"Value": 20000,
			"TransactionStartTime": "2025-12-10T08:15:00Z",
			"ProductCategory": "financial_services",
			"ChannelId": "ChannelId_2",
			"ProviderId": "ProviderId_4",
			"PricingStrategy": 2
		}
	]
}
```

## Local Development
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/eda.ipynb
```

> ℹ️ `mlflow-skinny` and `fastparquet` are baked into `requirements.txt` so the training workflow runs on Python 3.13 without compiling Arrow from source.

Or spin up the lightweight container:
```bash
docker-compose up --build
```

To observe MLflow locally: `mlflow ui --backend-store-uri mlruns --port 5000`.

## Final Submission Checklist
1. Run `python -m src.train` to refresh the MLflow registry and update processed datasets.
2. Capture screenshots for:
	- MLflow experiment Comparison page (best model highlighted).
	- GitHub Actions workflow green check (after push).
	- Docker container logs showing the API booted (`Uvicorn running on http://0.0.0.0:8000`).
3. Execute `docker compose up --build` and hit `/predict` with a sample payload; keep the terminal output for the report.
4. Update `reports/final_report.md` with modelling outcomes, API demo, and lessons learned.

## Historical Interim Milestones (14 Dec 2025)
1. Documented business context in this README.
2. `notebooks/eda.ipynb` published with top 5 insights and visuals stored in `reports/images/`.
