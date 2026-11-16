from pydantic import BaseModel, ConfigDict, Field


class Prediction(BaseModel):
    late_probability: float


class OrderFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    days_from_approval_to_estimate: int = Field(ge=0)
    item_count: int = Field(ge=1)
    distinct_product_count: int = Field(ge=1)
    distinct_seller_count: int = Field(ge=1)
    price_sum: float = Field(ge=0)
    freight_sum: float = Field(ge=0)
    total_weight_g: float | None = Field(default=None, ge=0)
    total_volume_cm3: float | None = Field(default=None, ge=0)
    max_seller_distance_km: float | None = Field(default=None, ge=0)
    customer_state: str = Field(pattern=r"^[A-Z]{2}$")