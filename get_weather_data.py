from ast import literal_eval
import datetime
import sys

from meteostat import Stations, Hourly
import pandas as pd

"""
Some weather data may fail to be fetched due to no nearby weather station being found or the weather station being too far away (20 km).
"""


def parse_date(date_str, time_str) -> datetime.datetime | None:
    """Parse date and time strings into a datetime object"""
    try:
        return datetime.datetime.strptime(
            date_str + " " + time_str, "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        print(f'Could not parse date and time from "{date_str} {time_str}"')
        return None


def get_temperature_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch temperature data from Meteostat and merge it into the DataFrame"""
    # Create a copy of the DataFrame to avoid modifying the original one
    df = df.copy()

    # Convert "start_location" string to a tuple where possible
    df.loc[df["start_location"].notna(), "start_location"] = df.loc[
        df["start_location"].notna(), "start_location"
    ].apply(lambda x: literal_eval(x))

    # Create list to hold the fetched weather data
    weather_data_list = []

    # Only consider rows where "start_location" is not NaN
    for i, row in df.loc[df["start_location"].notna()].iterrows():
        lat, lon = row["start_location"]
        # Get nearest weather station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if not station.empty and station["distance"].values[0] <= 20000:
            # Fetch weather data
            start_time = parse_date(row["date"], row["time_of_day"])
            if start_time is None:
                continue
            # Round end_time to the next hour
            end_time = start_time + datetime.timedelta(hours=1)
            hour_data = Hourly(station, start_time, end_time)
            weather_data = hour_data.fetch()

            if (
                "temp" in weather_data.columns
                and not weather_data.empty
                and "rhum" in weather_data.columns
                and not weather_data.empty
            ):
                # Append fetched data to the list
                weather_data_list.append(
                    {
                        "id": row["id"],
                        "temperature_c": weather_data["temp"].mean(),
                        "humidity_pct": weather_data["rhum"].mean(),
                        "air_pressure_hpa": weather_data["pres"].mean(),
                        "station_distance_m": station["distance"].values[0],
                    }
                )

    # Convert the list into a DataFrame
    weather_df = pd.DataFrame(weather_data_list)
    weather_df.to_csv("weather_data.csv", index=False)
    weather_df = weather_df.drop(columns=["station_distance_m"], axis=1)

    # Merge the weather data into the original DataFrame
    df = df.drop(columns=["temperature_c", "humidity_pct", "air_pressure_hpa"])
    df = pd.merge(df, weather_df, on="id", how="left")

    return df


if __name__ == "__main__":
    try:
        df = pd.read_csv("strava_data.csv")
    except FileNotFoundError:
        print("strava_data.csvが見つかりません。get_strava_data.pyを実行してください。")
        sys.exit(1)
    df = get_temperature_from_df(df)
    # Save all data together
    try:
        df.to_csv("strava_data.csv", index=False)
    except PermissionError:
        print("strava_data.csvを閉じてから再度実行してください。")
        sys.exit(1)
