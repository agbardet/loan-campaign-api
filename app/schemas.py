from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    income: int = Field(..., ge=0, description="Annual income in thousands USD", example=120)
    family: int = Field(..., ge=1, le=4, example=3)
    cc_avg: float = Field(..., ge=0, description="Monthly credit card spend in thousands USD", example=3.0)
    education: int = Field(..., ge=1, le=3, description="1=Undergrad 2=Graduate 3=Advanced/Professional", example=2)
    mortgage: int = Field(..., ge=0, description="Mortgage value in thousands USD", example=0)
    zip_code: str = Field(..., description="5-digit ZIP code", example="94025")
    securities_account: int = Field(..., ge=0, le=1, example=0)
    cd_account: int = Field(..., ge=0, le=1, example=0)
    online: int = Field(..., ge=0, le=1, example=1)
    credit_card: int = Field(..., ge=0, le=1, example=0)


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    label: str
    confidence: str
