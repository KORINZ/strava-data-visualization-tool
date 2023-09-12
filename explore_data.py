import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility import load_matplotlib_local_fonts


# Load local fonts
load_matplotlib_local_fonts("fonts/ipaexg.ttf", 12)
DPI = 300

if not os.path.exists("images"):
    os.makedirs("images")

try:
    df = pd.read_csv(
        "strava_data.csv", parse_dates={"datetime": ["date", "time_of_day"]}
    )
except FileNotFoundError:
    print("strava_data.csvが見つかりません。get_strava_data.pyを実行してください。")
    print("strava_data.csv not found. Please run get_strava_data.py.")
    sys.exit(1)


# Filter out activities that are not "Run" type
df = df[df["type"] == "Run"]

# Drop columns that are not needed for this analysis
df = df.drop(
    ["id", "type", "datetime", "start_location",
        "humidity_pct", "air_pressure_hpa"],
    axis=1,
)

# Mapping from English to Japanese
column_name_mapping = {
    "distance_km": "距離 (km)\nDistance",
    "average_pace_min_km": "ペース (min/km)\nPace",
    "duration_min": "時間 (min)\nDuration",
    "altitude_gains_m": "標高 (m)\nAltitude",
    "average_heart_rate_bpm": "心拍数 (bpm)\nHeart Rate",
    "calories_kcal": "カロリー (kcal)\nCalories",
    "kudos": "kudos",
    "temperature_c": "気温 (℃)\nTemperature",
    "humidity_pct": "湿度 (%)\nHumidity",
    "air_pressure_hpa": "気圧 (hPa)\nAir Pressure",
}

# Rename the columns
df = df.rename(columns=column_name_mapping)

# Drop rows with missing values (indoor runs will be excluded due to missing location data/temperature data)
df = df.dropna()

# Calculate the correlation matrix of the numeric columns
correlation_matrix = df.corr()

# Generate a heatmap of the correlation matrix
plt.figure(1, figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="crest", vmin=-1, vmax=1)
plt.title("相関行列 (Correlation Matrix)")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png", dpi=DPI)

# Generate a scatter matrix of the numeric columns
sns.pairplot(df, diag_kind="hist")
plt.subplots_adjust(bottom=0.075)
plt.savefig("images/scatter_matrix.png", dpi=DPI)

plt.show()
