# service.py
import numpy as np
import bentoml
from pydantic import BaseModel, Field

class HousePriceInput(BaseModel):
    OverallQual: int = Field(..., description="Overall material and finish quality (1-10)")
    GrLivArea: int = Field(..., description="Above grade living area in sq ft")
    GarageCars: int = Field(..., description="Garage car capacity")
    GarageArea: int = Field(..., description="Garage size in sq ft")
    TotalBsmtSF: int = Field(..., description="Total basement area in sq ft")
    FullBath: int = Field(..., description="Number of full bathrooms")
    TotRmsAbvGrd: int = Field(..., description="Total rooms above grade")
    YearBuilt: int = Field(..., description="Year the house was built")


# Create the service
@bentoml.service(name="house_price_service")
class HousePriceService:
    bento_model = "house-pricing-model-simple:latest"
    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)
    @bentoml.api
    async def predict(self, input_data: HousePriceInput) -> float:
        # Convert Pydantic model to numpy array
        input_array = np.array([
            input_data.OverallQual,
            input_data.GrLivArea,
            input_data.GarageCars,
            input_data.GarageArea,
            input_data.TotalBsmtSF,
            input_data.FullBath,
            input_data.TotRmsAbvGrd,
            input_data.YearBuilt
        ]).reshape(1, -1)
        
        # Make prediction directly with numpy array
        prediction = self.model.predict(input_array)
        return float(prediction[0])
