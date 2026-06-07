import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import CustomerInput, PredictionOutput
from app.predictor import predict

MODEL_PATH = Path("models/pipeline.pkl")
STATIC_DIR = Path("app/static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        app.state.pipeline = pickle.load(f)
    yield


app = FastAPI(
    title="AllLife Bank — Loan Propensity API",
    description="Predicts whether a liability customer will accept a personal loan offer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": "post-pruned-decision-tree", "test_f1": 0.915}


@app.post("/predict", response_model=PredictionOutput)
def predict_loan(body: CustomerInput, request: Request):
    return predict(request.app.state.pipeline, body)
