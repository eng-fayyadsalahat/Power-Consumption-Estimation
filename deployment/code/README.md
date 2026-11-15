# Power Consumption Estimation

An end-to-end Machine Learning project for estimating household electricity consumption using electrical measurements and temporal features. The project covers data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and deployment through a Flask REST API and Docker.

## 1. Project Overview

The main objective is to estimate **Global Active Power** based on electrical measurements and time-related features collected at the same minute.

The project uses the **HistGradientBoostingRegressor** model and provides a REST API that accepts input measurements and returns an estimated electricity consumption value.

> **Note:** This project focuses on same-time consumption estimation, not future electricity consumption forecasting.

## 2. Dataset

The project uses the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from the UCI Machine Learning Repository.

| Property | Description |
|---|---|
| Original Records | 2,075,259 |
| Processed Records | 2,049,280 |
| Original Features | 9 |
| Target Variable | Global_active_power |
| Missing Values | 25,979 rows removed |
| Data Frequency | One-minute measurements |
| Data Period | December 2006 – November 2010 |

### Data Preprocessing

- Converted electrical measurement columns to numeric values.
- Replaced invalid values (`?`) with missing values.
- Removed rows containing missing electrical measurements.
- Created a datetime column and sorted observations chronologically.
- Checked for duplicate records.

## 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand electricity consumption patterns and relationships between electrical measurements.

The analysis included:

- Descriptive statistics and distribution analysis.
- Correlation analysis.
- Consumption patterns across hours, weekdays, and months.
- Analysis of electricity consumption under different weather and time conditions.
- Investigation of relationships between electrical measurements and the target variable.

## 4. Feature Engineering

Several features were created to improve model performance.

| Feature | Description |
|---|---|
| Apparent_Power | Voltage × Global_intensity |
| hour_sin / hour_cos | Cyclical encoding of the hour |
| dayofweek_sin / dayofweek_cos | Cyclical encoding of the weekday |
| month_sin / month_cos | Cyclical encoding of the month |
| day_sin / day_cos | Cyclical encoding of the day |
| is_weekend | Weekend indicator |
| time_period | Time-of-day category |
| season | Seasonal category |
| year | Calendar year |

**Power Factor** was calculated during feature engineering but excluded from the model inputs because it is derived from the target variable.

## 5. Model Development

The project uses a chronological data split to preserve the time order of observations.

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

### Selected Model

**HistGradientBoostingRegressor**

Configuration:

```python
HistGradientBoostingRegressor(
    max_iter=250,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)
```

A preprocessing pipeline was used with:

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.

### Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validation | To be updated | To be updated | To be updated |
| Test | To be updated | To be updated | To be updated |

Replace these values with the final evaluation results obtained using the selected model configuration.

### Feature Importance

Permutation Feature Importance was used to understand the contribution of individual features.

The analysis identified:

1. Apparent_Power as the most influential feature.
2. Global_intensity as the second most influential feature.
3. Sub_metering_3 and other electrical measurements as additional contributors.

Temporal features generally had smaller contributions compared with the main electrical measurements.

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

**Note:** The trained model file (`power_model.pkl`) is not included in the GitHub repository because of its size. It is generated locally by running `train.py`.

## 7. Installation and Model Training

### Prerequisites

- Python 3.13
- Pipenv
- Docker (for containerized deployment)

### Clone the Repository

```bash
git clone <repository-url>
cd Power-Consumption-Estimation
```

### Install Dependencies

Navigate to the deployment directory:

```bash
cd deployment/code
```

Install the dependencies:

```bash
pipenv install
```

### Download the Dataset

Download the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

Extract the dataset and place `household_power_consumption.txt` in the location expected by `train.py`.

### Train the Model

Run:

```bash
pipenv run python train.py
```

After successful training, the following file will be generated:

```text
power_model.pkl
```

This file is required by the prediction API and Docker deployment.

## 8. API Deployment with Flask

The project provides a Flask REST API for interacting with the trained model.

Start the API locally:

```bash
pipenv run python predict.py
```

The service runs on:

```text
http://localhost:9696
```

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks the service status |
| POST | `/predict` | Returns an electricity consumption estimate |

### Health Check

Request:

```http
GET /health
```

Example response:

```json
{
    "status": "OK",
    "model": "HistGradientBoostingRegressor"
}
```

### Prediction

Request:

```http
POST /predict
Content-Type: application/json
```

The prediction endpoint requires the following 19 features:

```text
Global_reactive_power
Voltage
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
year
Apparent_Power
hour_sin
hour_cos
dayofweek_sin
dayofweek_cos
month_sin
month_cos
day_sin
day_cos
is_weekend
time_period
season
```

Example response:

```json
{
    "global_active_power": 0.9026952075451284
}
```

## 9. Docker Deployment

Docker is used to package the API and its dependencies into a container.

**Important:** Since `power_model.pkl` is excluded from GitHub, you must generate the model locally by running `train.py` before building the Docker image.

### Build the Docker Image

```bash
docker build -t power-consumption .
```

### Run the Container

```bash
docker run -d \
    --name power-api \
    -p 9696:9696 \
    power-consumption:latest
```

### Check Running Containers

```bash
docker ps
```

### Check Container Logs

```bash
docker logs power-api
```

### Test the Service

```bash
curl http://localhost:9696/health
```

The API can also be tested using Postman by sending GET and POST requests to the endpoints described above.

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Flask
- Gunicorn
- Docker
- Jupyter Notebook
- Pipenv

## 11. Future Improvements

- Experiment with additional regression models.
- Improve model generalization and evaluation.
- Explore future electricity consumption forecasting using historical observations.
- Extend the API with additional monitoring and validation capabilities.

## 12. Author

**Fayyad Salahat**

- GitHub: [eng-fayyadsalahat](https://github.com/eng-fayyadsalahat)
- Portfolio: [fayyad-salahat.webflow.io](https://fayyad-salahat.webflow.io)
- LinkedIn: [Fayyad Salahat](https://www.linkedin.com/in/fayyad-salahat/)

---

If you find this project useful, feel free to explore the repository and share your feedback.
