# House Prices Prediction

This repository demonstrates how to predict house sale prices using the Ames Housing dataset. The project is part of the **ML Engineering Zoomcamp** capstone and also takes part in the Kaggle competition **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)**. Accurate price predictions are valuable for buyers, sellers, and agents, and this project provides an end-to-end solution—spanning exploratory data analysis, feature engineering, model training, and deployment.

---
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![BentoML](https://img.shields.io/badge/BentoML-%234395FA?logo=bentoml&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23F76F00?logo=xgboost&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-%232496ED?logo=docker&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-%23035AFC?logo=kaggle&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GCP-%234285F4?logo=googlecloud&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/Github%20Actions-%232088FF?logo=githubactions&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243?logo=numpy&logoColor=white)

## Contents
1. [Getting Started](#getting-started)
2. [Project Details](#project-details)
3. [Models Evaluated](#models-evaluated)
4. [Simplified Model](#simplified-model)
5. [Project Structure](#project-structure)
6. [Environment Setup](#environment-setup)

---

## Getting Started

1. **Exploratory Data Analysis & Model Training**
   - Start with `eda.ipynb` to explore the dataset and understand the feature engineering process. Easiest way to configure the environment is to use 'uv sync' to install the dependencies.
   - The notebook will guide you through training various models and selecting the best performing one
   - A simplified XGBoost model using 8 key features will be created for deployment, in the notebook the model is trained and saved as a BentoML model

2. **BentoML Setup & Deployment**
   - Create an account at [BentoML Cloud](https://cloud.bentoml.com)
   - Install the BentoML CLI and login:
     ```bash
     bentoml cloud login
     ```
   - Build and deploy the service:
     ```bash
     bentoml build
     bentoml deploy
     ```
   - Once deployed, you'll receive an endpoint URL for making predictions

3. **Testing the Deployed Model**
   ```bash
   curl -X POST https://your-endpoint-url/predict \
     -H "Content-Type: application/json" \
     -d '{
       "input_data": {
         "OverallQual": 7,
         "GrLivArea": 2000,
         "GarageCars": 2,
         "GarageArea": 400,
         "TotalBsmtSF": 1000,
         "FullBath": 2,
         "TotRmsAbvGrd": 8,
         "YearBuilt": 2000
       }
     }'
   ```

---

## Project Details

### Models Evaluated
- Linear Regression
- Ridge and Lasso Regression
- Random Forest
- XGBoost (with hyperparameter tuning)

### Simplified Model
The deployed model uses 8 key features:
- OverallQual
- GrLivArea
- GarageCars
- GarageArea
- TotalBsmtSF
- FullBath
- TotRmsAbvGrd
- YearBuilt

### Project Structure
- `eda.ipynb`: Interactive notebook for data analysis and model training
- `service.py`: BentoML service definition for model deployment
- `bentofile.yaml`: BentoML configuration
- `pyproject.toml`: Project dependencies

### Environment Setup
- Use `uv` or your preferred Python virtual environment manager
- Install dependencies from `pyproject.toml`

---

**Note:** If you encounter any issues during deployment or have questions, feel free to open an issue in this repository. Happy predicting!