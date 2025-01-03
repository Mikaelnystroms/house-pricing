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
1. [Project Objectives](#project-objectives)
2. [Models and Training](#models-and-training)
3. [Deployment](#deployment)
4. [Environment and Reproducibility](#environment-and-reproducibility)
5. [EDA and Scripts](#eda-and-scripts)

---

## Project Objectives
1. **Perform Exploratory Data Analysis (EDA) and Feature Engineering**  
   Gain insights into the data and transform features (e.g., encoding categorical variables, handling missing values).

2. **Train and Evaluate Multiple Models**  
   Evaluate a range of regression models, including:
   - **Linear Regression**  
   - **Ridge** and **Lasso**  
   - **Random Forest**  
   - **XGBoost** (with grid search tuning for hyperparameters such as learning rate, max depth, and number of estimators)

3. **Implement a Simplified XGBoost Model**  
   A stripped-down version of the XGBoost model uses only 8 features—OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FullBath, TotRmsAbvGrd, and YearBuilt—to balance accuracy with ease of deployment.

4. **Ensure Reproducibility**  
   Provide training scripts and clear instructions to replicate results.

5. **Deploy the Model**  
   Containerize and deploy the model (using **BentoML** and **Docker**) on a cloud platform, with underlying infrastructure managed by GCP.

---

## Models and Training

1. **Model Training and Hyperparameter Tuning**  
   - **XGBoost** was the main model tuned using grid search over parameters such as `learning_rate`, `max_depth`, and `n_estimators`.
   - **Simplified XGBoost** was additionally created for quick user inference, requiring only 8 features instead of the full ~270.

2. **Performance Evaluation**  
   - Each model was evaluated using standard metrics like RMSE (Root Mean Squared Error).  
   - Feature selection and hyperparameter tuning were iteratively refined to balance performance and model complexity.

---

## Deployment

### BentoML
- **Build a Bento**:  
  ```bash
  bentoml build
  ```
  This command creates a Bento with all necessary files and dependencies.

- **Deploy** (Command-Line Interface):  
  ```bash
  bentoml deploy [bento_name] -n [deployment_name]
  ```
  Attempts to deploy the Bento to the configured cloud platform.  
  *Note:* In some cases, deployment can remain stuck in a pending state. Troubleshooting steps include verifying your BentoML configuration, cloud credentials, and checking logs for errors.

### Docker Containerization
1. **Generate a Docker Image**:  
   ```bash
   bentoml containerize <bento-name>
   ```
   This builds a Docker image containing your Bento.

2. **Run the Container Locally**:  
   ```bash
   docker run -it --rm -p 3000:3000 <container-image>
   ```

3. **Test the Containerized Model**:  
   ```bash
   curl -X POST http://localhost:3000/predict \
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
   When successful, the model will return a predicted house price.

---

## Environment and Reproducibility
- **Virtual Environment**: Created via `uv` (or any Python virtual environment manager of choice).
- **Dependencies**: Listed in `pyproject.toml`.
- **BentoML Configuration**: Managed by `bentofile.yaml` and `service.py`.

---

## EDA and Scripts

1. **eda.ipynb**  
   - Interactive Jupyter Notebook for data analysis and experimentation.  
   - Useful for visualizations and exploratory work.

2. **eda.py**  
   - Script-based EDA and training.  
   - Allows automated or command-line execution without the notebook environment.

Both approaches demonstrate how the model (and its variations) are trained and evaluated. Comparing their outputs can help you understand trade-offs between the simpler 8-feature model and a more comprehensive model using the full dataset.

---

**Thank you for checking out this project!** If you have any questions or suggestions, feel free to open an issue or submit a pull request. Happy predicting!