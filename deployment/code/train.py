import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}")
    print("-" * 35)
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")


data = pd.read_csv("data/household_power_consumption.txt", sep = ";", low_memory=False)


# Feature Engineering

# Create a working copy
data_clean = data.copy(deep=True)

# Define numerical columns
numeric_cols = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

# Replace invalid values and convert columns to numeric
data_clean[numeric_cols] = (
    data_clean[numeric_cols]
    .replace('?', np.nan)
    .apply(pd.to_numeric, errors='coerce')
)

# Remove records with missing electrical measurements
data_clean.dropna(subset=numeric_cols, inplace=True)

# Convert Datetime column to datetime format
data_clean['Datetime'] = pd.to_datetime(
    data_clean['Date'] + ' ' + data_clean['Time'],
    format='%d/%m/%Y %H:%M:%S'
)

# Set Datetime as index
data_clean.set_index('Datetime', inplace=True)

# Sort records chronologically
data_clean.sort_index(inplace=True)

# Extract time-based features from Datetime index
data_clean['hour'] = data_clean.index.hour
data_clean['day'] = data_clean.index.day
data_clean['month'] = data_clean.index.month
data_clean['year'] = data_clean.index.year
data_clean['day_of_week'] = data_clean.index.dayofweek


# Calculate apparent power (VA)
data_clean['Apparent_Power'] = (
    data_clean['Voltage'] * data_clean['Global_intensity']
)

# Calculate power factor
# Global_active_power is measured in kW,
# while Apparent_Power is measured in VA.
data_clean['Power Factor'] = (
    data_clean['Global_active_power'] * 1000
    / data_clean['Apparent_Power'].replace(0, np.nan)
)


# Hour encoding
data_clean['hour_sin'] = np.sin(
    2 * np.pi * data_clean['hour'] / 24
)
data_clean['hour_cos'] = np.cos(
    2 * np.pi * data_clean['hour'] / 24
)

# Day of week encoding
data_clean['dayofweek_sin'] = np.sin(
    2 * np.pi * data_clean['day_of_week'] / 7
)
data_clean['dayofweek_cos'] = np.cos(
    2 * np.pi * data_clean['day_of_week'] / 7
)

# Month encoding
data_clean['month_sin'] = np.sin(
    2 * np.pi * (data_clean['month'] - 1) / 12
)
data_clean['month_cos'] = np.cos(
    2 * np.pi * (data_clean['month'] - 1) / 12
)

# Day of month encoding
data_clean['day_sin'] = np.sin(
    2 * np.pi * (data_clean['day'] - 1) / 31
)
data_clean['day_cos'] = np.cos(
    2 * np.pi * (data_clean['day'] - 1) / 31
)


# Remove original time components
cols_to_drop = [
    'Date', 'Time',
    'hour', 'day', 'month', 'day_of_week'
]

data_clean.drop(columns=cols_to_drop, inplace=True)

# Weekend feature
data_clean['is_weekend'] = (
    data_clean.index.dayofweek >= 5
).astype(int)

# Time period feature
hour = data_clean.index.hour

data_clean['time_period'] = pd.cut(
    hour,
    bins=[-1, 5, 11, 17, 23],
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

# Season feature
month = data_clean.index.month

season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}

data_clean['season'] = pd.Series(
    month.map(season_map),
    index=data_clean.index
)

target = "Global_active_power"

X = data_clean.drop(
    columns=[target, "Power Factor"],
    errors="ignore"
)

y = data_clean[target]


categorical_features = [
    "time_period",
    "season"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ],
    sparse_threshold=0
)



hgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    ))
])

hgb_pipeline.fit(X_train, y_train)

y_pred_hgb = hgb_pipeline.predict(X_val)

evaluate = evaluate_model(y_val, y_pred_hgb, "HistGradientBoosting")

output_file = f"power_model.pkl"

with open(output_file, 'wb') as f_out: 
    pickle.dump(hgb_pipeline, f_out)