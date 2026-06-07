import pandas as pd
from app.schemas import CustomerInput, PredictionOutput

_EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced_Professional"}


def build_input_df(data: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "Age": data.age,
        "Income": data.income,
        "Family": data.family,
        "CCAvg": data.cc_avg,
        "Education": _EDU_MAP[data.education],
        "Mortgage": data.mortgage,
        "ZIPCode": str(data.zip_code)[:2],
        "Securities_Account": data.securities_account,
        "CD_Account": data.cd_account,
        "Online": data.online,
        "CreditCard": data.credit_card,
    }])


def predict(pipeline, data: CustomerInput) -> PredictionOutput:
    df = build_input_df(data)
    prob = float(pipeline.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)
    label = "Likely to Accept" if pred == 1 else "Unlikely to Accept"
    if prob >= 0.95 or prob <= 0.05:
        confidence = "Very High"
    elif prob >= 0.75 or prob <= 0.25:
        confidence = "High"
    else:
        confidence = "Medium"
    return PredictionOutput(
        prediction=pred,
        probability=round(prob, 3),
        label=label,
        confidence=confidence,
    )
