# Exoplanet Classifier API & Dashboard

This repository contains the backend and frontend that our team built for the NASA Space Apps Challenge 2025 track **“A World Away: Hunting for Exoplanets with AI.”** The solution ingests the Kepler KOI catalog, trains ensemble classifiers (Random Forest and XGBoost), serves real-time predictions through a FastAPI REST API, and exposes a companion Angular dashboard for health monitoring, predictions, metrics and visual insights.

## Project Structure

```
.
├── app/                # FastAPI application and training utilities
├── models/             # Persisted joblib bundles (ignored by Git)
├── frontend/           # Angular 17 web application
├── koi-cumulative.csv  # NASA KOI dataset used for training
└── requirements.txt    # Python dependencies for the backend
```

## Backend (FastAPI)

### 1. Install Python dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train (optional) and start the API
If you need to regenerate the model artifacts:
```bash
python app/model_training.py
```

Run the FastAPI server in development mode:
```bash
uvicorn app.api:app --reload
```

### 3. Explore the API
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Key endpoints:
- `GET /` – Health check (reports if the model is loaded)
- `POST /predict` – Real-time classification for KOI feature payloads
- `POST /train` – Upload a CSV to retrain the models
- `GET /metrics` – Latest evaluation metrics
- `GET /feature-importance` – Ranked feature contribution
- `GET /plots` – Confusion matrix & importance plots encoded as base64 PNG

## Frontend (Angular Dashboard)

### 1. Install Node dependencies
```bash
cd frontend
npm install
```

### 2. Run the development server
```bash
npm start
```

The Angular CLI will serve the application at [http://localhost:4200](http://localhost:4200). The dashboard expects the FastAPI backend to be running at `http://localhost:8000`.

## Data Source
The project relies on the Kepler Objects of Interest cumulative catalog (`koi-cumulative.csv`) distributed by NASA. Additional datasets can be integrated by uploading a CSV through the `/train` endpoint or the Training page in the dashboard.

## Team
This project was created by **Chubby Rockets**, a team of ADS (Analysis and Systems Development) students from Fatec Sorocaba, AMS program:

- Davi Ryan Konuma Lima
- Matheus Henrique Schopp Peixoto
- Luiz Filipe de Camargo
- Lucas Feitosa Almeida Rocha
- Matheus de Araujo Emidio

## License
This hackathon project is provided as-is for educational and demonstration purposes. Feel free to fork and adapt it to your own space exploration ideas!
