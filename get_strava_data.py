import sys
import json

try:
    import pandas as pd
    from stravalib.client import Client
    from stravalib.exc import RateLimitExceeded
except ImportError:
    print(
        "Some required packages are missing. Please run: pip install -r requirements.txt"
    )
    sys.exit(1)

"""
Install necessary packages with:
pip install -r requirements.txt

Update all packages in Windows PowerShell with (may take a while):
pip list --outdated | foreach {pip install --upgrade $_.split()[0]}

Calories data is not available for summary level activities; therefore, we need to fetch the detailed activity data for each activity.
"""


def get_credentials() -> tuple:
    """Get client ID and client secret from credentials.json or prompt the user to enter them"""
    try:
        with open("credentials.json", "r") as f:
            credentials = json.load(f)
            client_id = credentials["client_id"]
            client_secret = credentials["client_secret"]
    except FileNotFoundError:
        client_id = input("クライアントIDを入力してください: ")
        client_secret = input("クライアントシークレットを入力してください: ")
        with open("credentials.json", "w") as f:
            json.dump({"client_id": client_id, "client_secret": client_secret}, f)
    return client_id, client_secret


def get_authorized_client(
    client_id: str, client_secret: str, redirect_uri: str
) -> Client:
    """Get an authorized client object"""
    client = Client()
    url = client.authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=["read", "activity:read_all"],
    )

    print("Authorize the app on: ", url)
    redirected_url = input("↑上に表示されたURLをクリックし、最終的に表示されたURLを入力してください↑: ")

    try:
        code = redirected_url.split("code=")[1].split("&")[0]
        token_response = client.exchange_code_for_token(
            client_id=client_id, client_secret=client_secret, code=code
        )
        client.access_token = token_response["access_token"]
    except IndexError:
        print(
            "最終的に表示されたURLが正しくありません。フォーマットはhttp://localhost/?state=&code=...&scope=read,activity:read_allです。"
        )
        sys.exit(1)

    print("認証が成功しました！")

    return client


def fetch_activities(client: Client) -> pd.DataFrame:
    """Fetch activities from Strava and return a DataFrame"""
    try:
        activities = client.get_activities()
        data = []
        for activity in activities:
            if activity is not None:
                row = {
                    "id": activity.id,
                    "date": activity.start_date_local.date(),  # yyyy-mm-dd
                    "time_of_day": activity.start_date_local.time(),  # hh:mm:ss
                    "type": activity.sport_type,
                    "distance_km": activity.distance.num / 1000,  # km
                    "average_pace_m_s": activity.average_speed,  # m/s
                    "duration": activity.moving_time,  # days hh:mm:ss
                    "altitude_gains_m": activity.total_elevation_gain,  # m
                    "average_heart_rate_bpm": activity.average_heartrate,  # bpm
                    "start_location": activity.start_latlng,  # (lat, lng)
                    "kudos": activity.kudos_count,
                    "temperature_c": None,  # Placeholder
                    "humidity_pct": None,  # Placeholder
                    "air_pressure_hpa": None,  # Placeholder
                }
                data.append(row)
    except RateLimitExceeded:
        print("APIの呼び出し回数が上限に達しました。15分後に再度実行してください。(200回15分毎のリクエスト、 2,000回デイリー)")
        sys.exit(1)
    df = pd.DataFrame(data)

    # Convert all average_pace data to strings and get numeric values
    df["average_pace_m_s"] = df["average_pace_m_s"].astype(str)
    df["average_pace_m_s"] = df["average_pace_m_s"].str.split(" ").str[0]
    df["average_pace_m_s"] = pd.to_numeric(
        df["average_pace_m_s"], errors="coerce"
    ).round(2)

    # Convert average_pace from m/s to min/km only where the value is not 0 or NaN
    mask = df["average_pace_m_s"].notna() & (df["average_pace_m_s"] != 0)
    df.loc[mask, "average_pace_m_s"] = (
        1 / df.loc[mask, "average_pace_m_s"]
    ) * 16.6666667
    df = df.rename(columns={"average_pace_m_s": "average_pace_min_km"})
    df["average_pace_min_km"] = df["average_pace_min_km"].round(2)

    # Extract numeric altitude gains value and convert to float
    df["altitude_gains_m"] = df["altitude_gains_m"].astype(str)
    df["altitude_gains_m"] = df["altitude_gains_m"].str.split(" ").str[0].astype(float)

    # Reformat location data
    df["start_location"] = df["start_location"].astype(str)
    df["start_location"] = (
        df["start_location"].str.replace("[", "(").str.replace("]", ")")
    )
    df["start_location"] = df["start_location"].str.split("=").str[1]

    # Convert duration from string formatted timedelta to total minutes
    df["duration"] = (df["duration"].dt.total_seconds() / 60).round(2)
    df = df.rename(columns={"duration": "duration_min"})

    # Round distance to 2 decimal places
    df["distance_km"] = df["distance_km"].round(2)

    return df


def fetch_calories(client: Client, df: pd.DataFrame) -> pd.DataFrame:
    """Fetch calories data from Strava and merge it into the DataFrame"""
    # Check for existing calories data
    try:
        df_calories = pd.read_csv("strava_data_calories.csv")
    except FileNotFoundError:
        df_calories = pd.DataFrame(columns=["id", "calories_kcal"])

    warning_message = "カロリーデータを取得しますか？（API呼び出し回数が制限されているため、大量のデータ(200回15分毎のリクエスト、 2,000回デイリー)を取得するとエラーが発生する可能性があります。）[Y/N]: "

    user_input = input(warning_message)
    while user_input not in ["Y", "N"]:
        print("無効な入力です。もう一度入力してください。")
        user_input = input(warning_message)

    if user_input == "Y":
        new_calories_data = []
        fetch_count = df_calories.shape[0]

        # Only consider rows of type "Run" or "Ride" for total rows
        total_rows = df[df["type"].isin(["Run", "Ride"])].shape[0]
        for _, row in df.iterrows():
            if row["id"] not in df_calories["id"].values and row["type"] in {
                "Run",
                "Ride",
            }:
                try:
                    detailed_activity = client.get_activity(row["id"])
                    new_calories_data.append(
                        {"id": row["id"], "calories_kcal": detailed_activity.calories}
                    )
                    fetch_count += 1  # Increment counter
                    print(
                        f"Fetched calories data for id {row['id']}... ({fetch_count}/{total_rows})"
                    )
                except AttributeError as e:
                    print(f"Error fetching detailed activity for id {row['id']}: {e}")
                except RateLimitExceeded:
                    print("APIの呼び出し回数が制限を超えました。15分待ってから再度実行してください。今まで取得したデータは保存されます。")
                    df_new_calories = pd.DataFrame(new_calories_data)
                    df_calories = pd.concat([df_calories, df_new_calories])
                    df_calories.to_csv("strava_data_calories.csv", index=False)

                    # Exit the program to avoid losing data
                    break
        if new_calories_data:
            df_new_calories = pd.DataFrame(new_calories_data)
            df_calories = pd.concat([df_calories, df_new_calories])
            df_calories = df_calories.drop_duplicates(subset=["id"])
            df_calories.to_csv("strava_data_calories.csv", index=False)
    return df.merge(df_calories, on="id", how="left")


if __name__ == "__main__":
    client_id, client_secret = get_credentials()
    redirect_uri = "http://localhost"
    client = get_authorized_client(client_id, client_secret, redirect_uri)
    df = fetch_activities(client)
    df.to_csv("strava_data.csv", index=False)

    # Read the data back in and fetch the calories
    df = pd.read_csv("strava_data.csv")
    df = fetch_calories(client, df)

    # Save all data together
    try:
        df.to_csv("strava_data.csv", index=False)
    except PermissionError:
        print("strava_data.csvを閉じてから再度実行してください。")
        sys.exit(1)
    else:
        print("完了。")
