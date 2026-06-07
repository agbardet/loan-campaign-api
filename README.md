# AllLife Bank — Loan Propensity API

FastAPI service wrapping the [loan-campaign](https://github.com/agbardet/loan-campaign) post-pruned Decision Tree (Test F1 0.915, Recall 0.903). Accepts raw customer data, applies a full sklearn preprocessing pipeline, and returns a loan acceptance prediction with probability.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- UI: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 120,
    "family": 3,
    "cc_avg": 3.0,
    "education": 2,
    "mortgage": 0,
    "zip_code": "94025",
    "securities_account": 0,
    "cd_account": 0,
    "online": 1,
    "credit_card": 0
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.872,
  "label": "Likely to Accept",
  "confidence": "High"
}
```

| Field | Description |
|-------|-------------|
| `prediction` | 0 = will not accept, 1 = will accept |
| `probability` | Model confidence (0–1) that customer accepts |
| `label` | Human-readable prediction |
| `confidence` | High / Medium based on probability distance from 0.5 |

## Rebuild the Pipeline

If you retrain the model, regenerate `models/pipeline.pkl`:
```bash
python build_pipeline.py
```

## Tech Stack

FastAPI · scikit-learn Pipeline · Pydantic · Uvicorn
