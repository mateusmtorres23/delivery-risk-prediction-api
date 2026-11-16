from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .predict import load_model, predict_late_probability
from .schemas import OrderFeatures, Prediction

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    application.state.model = load_model()
    yield


app = FastAPI(title="Olist Delivery Risk", lifespan=lifespan)


@app.post("/predict", response_model=Prediction)
def predict_order(order: OrderFeatures, request: Request) -> Prediction:
    probability = predict_late_probability(request.app.state.model, order.model_dump())
    return Prediction(late_probability=probability)
