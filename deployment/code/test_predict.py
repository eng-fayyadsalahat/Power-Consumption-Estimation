import requests

url = "http://localhost:9696/predict"

power = {
    "Global_reactive_power": 0.1,
    "Voltage": 240.5,
    "Global_intensity": 4.2,
    "Sub_metering_1": 0.0,
    "Sub_metering_2": 1.0,
    "Sub_metering_3": 6.0,
    "year": 2008,
    "Apparent_Power": 1010.1,
    "hour_sin": -0.5,
    "hour_cos": 0.866,
    "dayofweek_sin": 0.0,
    "dayofweek_cos": 1.0,
    "month_sin": 0.5,
    "month_cos": 0.866,
    "day_sin": 0.2,
    "day_cos": 0.98,
    "is_weekend": 0,
    "time_period": "Morning",
    "season": "Spring"
}

response = requests.post(url, json=power)

print("Status Code:", response.status_code)
print("Prediction:", response.json())