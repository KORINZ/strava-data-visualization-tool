import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

"""Plot the correlation matrix and scatter matrix of the numeric columns"""


mpl.rcParams["font.family"] = "Meiryo"
mpl.rcParams["font.size"] = 12
DPI = 300

if not os.path.exists("images"):
    os.makedirs("images")

try:
    df = pd.read_csv(
        "strava_data.csv", parse_dates={"datetime": ["date", "time_of_day"]}
    )
except FileNotFoundError:
    print("strava_data.csvが見つかりません。get_strava_data.pyを実行してください。")
    sys.exit(1)


# Filter out activities that are not "Run" type
df = df[df["type"] == "Run"]

# Drop columns that are not needed for this analysis
df = df.drop(
    ["id", "type", "datetime", "start_location", "humidity_pct", "air_pressure_hpa"],
    axis=1,
)

# Mapping from English to Japanese
column_name_mapping = {
    "distance_km": "距離",
    "average_pace_min_km": "平均ペース",
    "duration_min": "持続時間",
    "altitude_gains_m": "獲得標高",
    "average_heart_rate_bpm": "平均心拍数",
    "calories_kcal": "カロリー",
    "kudos": "kudos",
    "temperature_c": "気温",
    "humidity_pct": "湿度",
    "air_pressure_hpa": "気圧",
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
plt.title("相関行列")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png", dpi=DPI)

# Generate a scatter matrix of the numeric columns
sns.pairplot(df, diag_kind="hist")
plt.subplots_adjust(bottom=0.075)
plt.savefig("images/scatter_matrix.png", dpi=DPI)

plt.show()
