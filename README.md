# Household Power Consumption Estimation

## Project Overview

This project aims to estimate household active power consumption using electrical measurements and time-based features from the UCI Individual Household Electric Power Consumption dataset.

The project follows an end-to-end Machine Learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tuning, evaluation, and deployment using Flask and Docker.

**Important:** This project estimates power consumption using measurements recorded at the same time. It does not predict future energy consumption.

## Problem Description

Accurately estimating household electricity consumption is useful for understanding energy usage patterns and supporting energy monitoring applications.

The objective is to estimate:

`Global_active_power`

Using electrical measurements and temporal features recorded at the same minute.

This is a supervised regression problem.

## Dataset

The project uses the [Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from the UCI Machine Learning Repository.

### Dataset Information

- Original observations: 2,075,259
- Original features: 9
- Time period: December 2006 – November 2010
- Sampling frequency: One observation per minute
- Target variable: `Global_active_power`

### Data Preparation

The following preprocessing steps were performed:

- Converted missing-value markers (`?`) into NaN.
- Removed records with missing electrical measurements.
- Converted electrical measurements into numeric data types.
- Combined Date and Time into a DatetimeIndex.
- Extracted and engineered temporal features.
- Applied cyclical encoding to periodic time features.
- Created additional features such as weekend indicators, time periods, and seasons.

After cleaning, the dataset contained **2,049,280 observations**.

## Exploratory Data Analysis (EDA)

The exploratory analysis investigated:

- Global active power distribution.
- Hourly electricity consumption patterns.
- Daily and weekly consumption patterns.
- Monthly and seasonal consumption trends.
- Relationships between electrical measurements.
- Outlier detection using IQR and Isolation Forest.

The analysis revealed that electricity consumption generally increases during evening hours and tends to be higher during winter months.

Outlier analysis was used to investigate unusual consumption patterns. These observations were retained because unusual values do not necessarily represent incorrect measurements.

## Feature Engineering

Several features were created to improve model performance:

### Electrical Features

- `Apparent_Power`
- `Power Factor` (excluded from model inputs because it is derived using the target variable).

### Temporal Features

- Hour cyclical encoding (`hour_sin`, `hour_cos`)
- Day-of-week cyclical encoding
- Month cyclical encoding
- Day cyclical encoding
- Weekend indicator (`is_weekend`)
- Time period (`time_period`)
- Season (`season`)
- Year

The final model uses 19 input features.

## Model Development

Two regression algorithms were evaluated:

1. Linear Regression
2. HistGradientBoostingRegressor

A preprocessing pipeline was implemented using:

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- ColumnTransformer to combine preprocessing steps.
- Scikit-learn Pipeline to integrate preprocessing and model training.

### Data Splitting

The dataset was divided chronologically:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

This approach evaluates model performance on later time periods.

### Feature Importance

Permutation Importance was used to evaluate the contribution of each feature to the final HistGradientBoostingRegressor model. The analysis was performed on the validation set using Mean Absolute Error (MAE) as the evaluation metric.

The results showed that **Apparent_Power** was the most influential feature, followed by **Global_intensity**. Other electrical measurements, including Sub_metering_3, Global_reactive_power, and Sub_metering_1, provided smaller contributions.

Temporal features had relatively low importance compared with the electrical measurements, indicating that the model relies primarily on the electrical characteristics of each observation.

These findings are consistent with the project's objective of estimating Global_active_power using measurements collected at the same time. In particular, Apparent_Power and Global_intensity have strong relationships with the target variable.

Permutation Importance measures the increase in prediction error after randomly shuffling a feature. Therefore, lower importance indicates a smaller contribution under this evaluation, rather than proving that a feature is entirely useless.

### Hyperparameter Tuning

RandomizedSearchCV with TimeSeriesSplit was used to explore HistGradientBoostingRegressor hyperparameters.

The original HistGradientBoosting model achieved slightly better validation results than the tuned model, so the original configuration was selected as the final model.

## Model Evaluation

The final model was evaluated on the unseen test dataset.

| Metric | Test Result |
|---|---:|
| MAE | 0.019277 |
| RMSE | 0.030413 |
| R² | 0.998755 |

### Results Interpretation

- MAE: The average absolute prediction error is approximately 0.0193 kW.
- RMSE: The root mean squared error is approximately 0.0304 kW.
- R²: The model explains approximately 99.88% of the variance in the target variable.

The model achieved consistent results across validation and test datasets.

**Note:** The high performance is partly expected because the model uses contemporaneous electrical measurements, particularly Global_intensity, which is strongly correlated with Global_active_power. Therefore, these results demonstrate same-minute power estimation rather than future forecasting.

## 6. Project Structure

```text
Power-Consumption-Estimation/
│
├── data/
│
├── notebooks/
│   └── power_consumption.ipynb
│
├── deployment/
│   └── code/
│       ├── train.py
│       ├── predict.py
│       ├── Dockerfile
│       ├── .dockerignore
│       ├── Pipfile
│       ├── Pipfile.lock
│       ├── requirements.txt
│       ├── pyproject.toml
│       └── README.md
│
├── screenshots/
│   ├── deployment_demo.txt
│   └── ...
│
├── .gitignore
└── README.md
```
## Installation and Setup

### Requirements

- Python 3.11
- Pipenv
- Docker (for containerized deployment)

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Power-Consumption-Estimation
```

### 2. Install Dependencies

```bash
pip install pipenv
pipenv install
pipenv shell
```

### 3. Download the Dataset

Download the dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Place the downloaded file inside the `data/` directory.

### 4. Train the Model

```bash
python train.py
```

This script trains the final model and saves the trained pipeline.

### 5. Run the API

```bash
python predict.py
```

The API will be available at:

```text
http://localhost:9696
```

## API Endpoints

### Health Check

**GET** `/health`

Checks whether the API and model are available.

### Prediction

**POST** `/predict`

Receives electrical measurements and temporal features in JSON format and returns the estimated Global Active Power in kW.

The request must contain all 19 model input features.

A complete request example will be added after finalizing the API input format.

## Docker Deployment

Build the Docker image:

```bash
docker build -t power-consumption-estimation .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 power-consumption-estimation
```

The API will be accessible at:

```text
http://localhost:9696
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Flask
- Gunicorn
- Pipenv
- Docker

## Future Improvements

- Add a simple web interface for submitting electrical measurements.
- Improve API input validation and error handling.
- Explore feature importance and feature ablation.
- Deploy the API to a cloud hosting platform.

## Author

**Fayyad Salahat**

- GitHub: [eng-fayyadsalahat](https://github.com/eng-fayyadsalahat)
- Portfolio: [fayyad-salahat.webflow.io](https://fayyad-salahat.webflow.io)
- LinkedIn: [Fayyad Salahat](https://www.linkedin.com/in/fayyad-salahat/)
