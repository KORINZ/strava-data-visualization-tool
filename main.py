import sys
import os
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib as mpl

from typing import Literal, Tuple
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ペースの最小値と最大値を設定し、それより速いペースと遅いペースのアクティビティを除外する (min/km)
MIN_PACE = 3
MAX_PACE = 10

# 距離の最小値を設定し、それより短い距離のアクティビティを除外する (km)
MIN_DISTANCE = 0.5

# 丸のサイズを設定する
mpl.rcParams["lines.markersize"] = 3.9

# フォントを設定する
mpl.rcParams["font.family"] = "Meiryo"
mpl.rcParams["font.size"] = 12

# 他の設定
mpl.rcParams["axes.axisbelow"] = True
mpl.rcParams["legend.fontsize"] = "small"
SAVE_FIG_DPI = 300


def preprocess_df(df: pd.DataFrame, activity_type="Run") -> pd.DataFrame:
    """Preprocess the DataFrame to prepare for plotting."""

    # Filter by activity type
    df = df[df["type"] == activity_type].copy()

    # Convert date to datetime object
    df["date"] = pd.to_datetime(df["date"])

    df.loc[:, "time_of_day"] = (
        pd.to_datetime(df["time_of_day"], format="%H:%M:%S").dt.hour * 3600
        + pd.to_datetime(df["time_of_day"], format="%H:%M:%S").dt.minute * 60
        + pd.to_datetime(df["time_of_day"], format="%H:%M:%S").dt.second
    )

    # !Filter out average pace that is too fast or too slow
    df = df.query("@MIN_PACE <= average_pace_min_km <= @MAX_PACE")

    # !Filter out distance that is too short
    df = df.query("@MIN_DISTANCE <= distance_km")

    return df


def fit_data(
    df: pd.DataFrame, y_col: str, deg: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.poly1d]:
    """Fit data with polynomial of degree 'deg'."""
    date_as_number = (df["date"] - df["date"].min()).dt.days
    coefficients = np.polyfit(date_as_number, df[y_col], deg)
    polynomial = np.poly1d(coefficients)
    xfit = np.linspace(date_as_number.min(), date_as_number.max(), 1000)
    yfit = polynomial(xfit)

    return xfit, yfit, polynomial


def scatter_plots_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Generate scatter plots from the DataFrame."""
    fig, axs = plt.subplots(2, 1, figsize=[10, 10])

    # Scatter plot distance vs average pace, colored by average heart rate
    scatter0 = axs[0].scatter(
        df["distance_km"],
        df["average_pace_min_km"],
        c=df["average_heart_rate_bpm"],
        cmap="Reds",
    )
    axs[0].set_xlabel("距離 (km)")
    axs[0].set_ylabel("平均ペース (min/km)")
    axs[0].grid(True, linewidth=0.75)
    axs[0].invert_yaxis()
    cbar0 = fig.colorbar(scatter0, ax=axs[0])
    cbar0.set_label("平均心拍数")

    # Scatter plot time of day vs distance, colored by kudos
    default_marker_size = mpl.rcParams["lines.markersize"]
    scatter1 = axs[1].scatter(
        df["time_of_day"],
        df["distance_km"],
        c=df["kudos"],
        s=np.maximum(df["kudos"] * 5, default_marker_size**2),
        cmap="rainbow",
    )
    axs[1].set_xlabel("時間帯 (hh:mm)")
    axs[1].set_ylabel("距離 (km)")
    axs[1].xaxis.set_major_locator(MaxNLocator(nbins=9))
    axs[1].xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: "{:02}:{:02}".format(int(x // 3600), int((x % 3600) // 60))
        )
    )
    axs[1].set_xlim([0, 24 * 3600])
    axs[1].grid(True, linewidth=0.75)
    cbar1 = fig.colorbar(scatter1, ax=axs[1])
    cbar1.set_label("kudos")

    fig.tight_layout()

    return fig


def plot_basic_stat_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot basic statistics from a dataframe of activities."""

    fig, axs = plt.subplots(4, 1, figsize=[15, 30], sharex=True)

    # Plot distance (bar chart)
    axs[0].bar(
        df["date"],
        df["distance_km"],
        color="#34ACE4",
        align="center",
    )
    axs[0].set_ylabel("距離 (km)")
    axs[0].tick_params("x", labelbottom=False)
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[0].grid(True, axis="y", linestyle="--", linewidth=0.75)

    # Annotate the bars with the number of kudos
    for i, v in enumerate(df["distance_km"]):
        axs[0].text(
            df["date"].iloc[i],
            v + 0.1,
            str(df["kudos"].iloc[i]),
            color="black",
            ha="center",
            fontsize=4.5,
        )

    # Plot distance (line chart)
    axs[1].plot_date(
        df["date"], df["distance_km"], "#FC4C02", marker="o", markerfacecolor="white"
    )
    axs[1].fill_between(df["date"], df["distance_km"], color="#FC4C02", alpha=0.1)
    axs[1].set_ylabel("距離 (km)")
    axs[1].tick_params("x", labelbottom=False)
    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[1].grid(True, linestyle="--", linewidth=0.75)

    # Plot altitude gains (bar chart)
    axs[2].bar(
        df["date"],
        df["altitude_gains_m"],
        color="#617A55",
        align="center",
    )
    axs[2].set_ylabel("獲得標高 (m)")
    axs[2].tick_params("x", labelbottom=False)
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[2].yaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:.0f}".format(x)))
    axs[2].grid(True, axis="y", linestyle="--", linewidth=0.75)

    # Plot duration (scatter chart)
    axs[3].scatter(
        df["date"],
        df["duration_min"],
        color="#AB68FF",
    )
    axs[3].set_ylabel("時間 (min)")

    # Fit the data
    xfit, yfit, polynomial = fit_data(df, "duration_min")
    xfit_dates = df["date"].min() + pd.to_timedelta(xfit, "D")
    fitting_label = f"カーブフィッティング (次数 {len(polynomial)})"
    axs[3].plot(
        xfit_dates,
        yfit,
        color="#AB68FF",
        alpha=0.75,
        linestyle="--",
        linewidth=1.5,
        label=fitting_label,
    )
    axs[3].fill_between(xfit_dates, yfit, color="#AB68FF", alpha=0.1)
    axs[3].grid(True, linestyle="--", linewidth=0.75)
    axs[3].legend()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.035)

    return fig


def plot_detailed_stat_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot detailed statistics from a dataframe of activities."""
    fig, axs = plt.subplots(4, 1, figsize=[15, 30], sharex=True)

    # Plot distance (line chart)
    axs[0].scatter(df["date"], df["distance_km"], color="#FC4C02", marker="o")
    axs[0].set_ylabel("距離 (km)")
    axs[0].tick_params("x", labelbottom=False)
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[0].grid(True, linestyle="--", linewidth=0.75)
    # Fit the data
    xfit, yfit, polynomial = fit_data(df, "distance_km")
    xfit_dates = df["date"].min() + pd.to_timedelta(xfit, "D")
    fitting_label = f"カーブフィッティング (次数 {len(polynomial)})"
    axs[0].plot(
        xfit_dates,
        yfit,
        color="#FC4C02",
        linestyle="--",
        alpha=0.75,
        linewidth=1.5,
        label=fitting_label,
    )
    axs[0].fill_between(xfit_dates, yfit, color="#FC4C02", alpha=0.1)
    axs[0].grid(True, linestyle="--", linewidth=0.75)
    axs[0].legend()

    # Plot calories burned (bar chart)
    axs[1].bar(df["date"], df["calories_kcal"], color="#EBD944", align="center")
    axs[1].set_ylabel("カロリー (kcal)")
    axs[1].tick_params("x", labelbottom=False)
    axs[1].grid(True, axis="y", linestyle="--", linewidth=0.75)

    # Plot average heart rate (scatter plot)
    axs[2].plot_date(df["date"], df["average_heart_rate_bpm"], color="#D6324B")
    axs[2].set_ylabel("平均心拍数 (bpm)")
    axs[2].tick_params("x", labelbottom=False)
    axs[2].grid(True, linestyle="--", linewidth=0.75)

    # Plot average pace (scatter plot)
    axs[3].scatter(df["date"], df["average_pace_min_km"], color="#19A7CE")
    axs[3].set_ylabel("平均ペース (min/km)")
    axs[3].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axs[3].grid(True, linestyle="--", linewidth=0.75)

    # Fit the data
    xfit, yfit, polynomial = fit_data(df, "average_pace_min_km")
    xfit_dates = df["date"].min() + pd.to_timedelta(xfit, "D")
    fitting_label = f"カーブフィッティング (次数 {len(polynomial)})"
    axs[3].plot(
        xfit_dates,
        yfit,
        color="#FC4C02",
        alpha=0.75,
        linestyle="-",
        linewidth=1.5,
        label=fitting_label,
    )
    axs[3].legend()
    axs[3].invert_yaxis()

    # Formatting date
    date_format = DateFormatter("%Y-%m")
    axs[3].xaxis.set_major_formatter(date_format)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.035)

    return fig


def scatter_3d_plot_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot a 3D scatter plot from a dataframe of activities."""
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111, projection="3d")

    # Check if 'temperature_c' column is all NaN
    if df["temperature_c"].isnull().all():
        plt.title("get_weather_data.pyを実行してください。")
        # If all values are NaN, return the empty plot
        return fig

    scatter = ax.scatter(
        df["average_pace_min_km"],
        df["duration_min"],
        df["temperature_c"],
        c=df["average_heart_rate_bpm"],
        s=40,
        cmap="Reds",
    )
    ax.set_xlabel("平均ペース (min/km)")
    ax.set_ylabel("時間 (min)")
    ax.set_zlabel("気温 (℃)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("平均心拍数")
    fig.tight_layout()

    return fig


def categorize_time_of_day(
    time_in_seconds: int,
) -> Literal["朝: 5:00 - 12:00", "昼: 12:00 - 18:00", "夜: 18:00 - 5:00"]:
    """Categorize time of day into morning, afternoon, and night."""
    if 5 * 3600 <= time_in_seconds < 12 * 3600:
        return "朝: 5:00 - 12:00"  # Morning: 5:00 - 12:00
    elif 12 * 3600 <= time_in_seconds < 18 * 3600:
        return "昼: 12:00 - 18:00"  # Afternoon: 12:00 - 18:00
    else:
        return "夜: 18:00 - 5:00"  # Night: 18:00 - 5:00


def plot_pie_chart_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot a pie chart from a dataframe of activities."""
    df["period_of_day"] = df["time_of_day"].apply(categorize_time_of_day)

    # Group by period of day
    group = df.groupby("period_of_day").size()

    # Set colors
    color_map = {
        "朝: 5:00 - 12:00": "#E6DF44",
        "昼: 12:00 - 18:00": "#F0810F",
        "夜: 18:00 - 5:00": "#063852",
    }
    colors = [color_map[i] for i in group.index]

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))

    # Pie chart
    wedges, labels, autopct_texts = ax[0, 0].pie(
        group,
        labels=group.index.tolist(),
        autopct=lambda p: f"{p:.1f}%  ({int(p * sum(group) / 100)})",
        startangle=90,
        colors=colors,
        textprops={"color": "black"},
        wedgeprops={"edgecolor": "white"},
    )

    ax[0, 0].axis("equal")
    ax[0, 0].set_title("一日の活動時間割合")

    # Change color of text in 'Night' section to white
    for label, pct in zip(labels, autopct_texts):
        if label.get_text() == "夜: 18:00 - 5:00":
            pct.set_color("white")

    # Bar plot for Day of Week
    # Translate weekday names to Japanese
    days_english_to_japanese = {
        "Monday": "月",
        "Tuesday": "火",
        "Wednesday": "水",
        "Thursday": "木",
        "Friday": "金",
        "Saturday": "土",
        "Sunday": "日",
    }
    days = ["月", "火", "水", "木", "金", "土", "日"]
    df["day_of_week"] = df["date"].dt.weekday.apply(
        lambda x: list(calendar.day_name)[x]
    )
    df["day_of_week"] = df["day_of_week"].map(days_english_to_japanese)
    day_counts = df["day_of_week"].value_counts().reindex(days)

    ax[0, 1].bar(day_counts.index, day_counts.values.tolist(), color="#18A545")
    ax[0, 1].set_title("日別頻度")
    ax[0, 1].set_xlabel("曜日")
    ax[0, 1].set_ylabel("活動数")
    ax[0, 1].grid(True, axis="y", linestyle="--", linewidth=0.75)

    # Bar plot for Month
    df["month"] = df["date"].dt.month
    month_counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0)

    ax[1, 1].bar(month_counts.index, month_counts.values.tolist(), color="#457B9D")
    ax[1, 1].set_title("月別頻度")
    ax[1, 1].set_xlabel("月")
    ax[1, 1].set_xticks(range(1, 13))
    ax[1, 1].set_ylabel("活動数")
    ax[1, 1].grid(True, axis="y", linestyle="--", linewidth=0.75)

    # Bar plot for Year
    df["year"] = df["date"].dt.year
    year_counts = df["year"].value_counts().sort_index()

    ax[1, 0].bar(year_counts.index, year_counts.values.tolist(), color="#FC4C02")
    ax[1, 0].set_title("年別頻度")
    ax[1, 0].set_xlabel("年")
    ax[1, 0].set_ylabel("活動数")
    ax[1, 0].set_xticks(year_counts.index)
    ax[1, 0].grid(True, axis="y", linestyle="--", linewidth=0.75)

    plt.tight_layout()
    return fig


def plot_histograms_from_df(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot histograms of distance and duration from a dataframe of activities."""

    fig, axs = plt.subplots(3, 1, figsize=[8, 10])

    # distance (histogram)
    axs[0].hist(df["distance_km"], bins="auto", color="#FC4C02", edgecolor="white")
    axs[0].set_xlabel("距離 (km)")
    axs[0].set_ylabel("頻度")
    axs[0].set_title("距離の分布")
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True, axis="y", linestyle="--")

    # duration (histogram)
    axs[1].hist(df["duration_min"], bins="auto", color="#AB68FF", edgecolor="white")
    axs[1].set_xlabel("時間 (min)")
    axs[1].set_ylabel("頻度")
    axs[1].set_title("時間の分布")
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True, axis="y", linestyle="--")

    # alitude_gains (histogram)
    axs[2].hist(df["altitude_gains_m"], bins="auto", color="#617A55", edgecolor="white")
    axs[2].set_xlabel("獲得標高 (m)")
    axs[2].set_ylabel("頻度")
    axs[2].set_title("獲得標高の分布")
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].grid(True, axis="y", linestyle="--")

    fig.tight_layout()

    return fig


def plot_weather_data(df: pd.DataFrame) -> matplotlib.figure.Figure:
    fig, axs = plt.subplots(2, 2, figsize=[12, 10])

    # Check if 'temperature_c' column is all NaN
    if df["temperature_c"].isnull().all():
        plt.suptitle("get_weather_data.pyを実行してください。")
        # If all values are NaN, return the empty plot
        return fig

    scatter0 = axs[0, 0].scatter(
        df["average_pace_min_km"],
        df["temperature_c"],
        c=df["humidity_pct"],
        cmap="Blues",
    )
    axs[0, 0].set_xlabel("ペース (min/km)")
    axs[0, 0].set_ylabel("気温 (℃)")
    axs[0, 0].grid(True, linewidth=0.75)
    axs[0, 0].invert_xaxis()
    cbar0 = fig.colorbar(scatter0, ax=axs[0, 0])
    cbar0.set_label("湿度 (%)")

    scatter1 = axs[0, 1].scatter(
        df["distance_km"],
        df["temperature_c"],
        c=df["humidity_pct"],
        cmap="Blues",
    )
    axs[0, 1].set_xlabel("距離 (km)")
    axs[0, 1].set_ylabel("気温 (℃)")
    axs[0, 1].grid(True, linewidth=0.75)
    cbar1 = fig.colorbar(scatter1, ax=axs[0, 1])
    cbar1.set_label("湿度 (%)")

    scatter2 = axs[1, 0].scatter(
        df["average_pace_min_km"],
        df["humidity_pct"],
        c=df["temperature_c"],
        cmap="Reds",
    )
    axs[1, 0].set_xlabel("ペース (min/km)")
    axs[1, 0].set_ylabel("湿度 (%)")
    axs[1, 0].grid(True, linewidth=0.75)
    axs[1, 0].invert_xaxis()
    cbar2 = fig.colorbar(scatter2, ax=axs[1, 0])
    cbar2.set_label("気温 (℃)")

    scatter3 = axs[1, 1].scatter(
        df["average_pace_min_km"],
        df["air_pressure_hpa"],
        c=df["temperature_c"],
        cmap="Reds",
    )
    axs[1, 1].set_xlabel("ペース (min/km)")
    axs[1, 1].set_ylabel("気圧 (hPa)")
    axs[1, 1].grid(True, linewidth=0.75)
    axs[1, 1].invert_xaxis()
    cbar3 = fig.colorbar(scatter3, ax=axs[1, 1])
    cbar3.set_label("気温 (℃)")

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    try:
        df = pd.read_csv("strava_data.csv")
    except FileNotFoundError:
        print("strava_data.csvが見つかりません。get_strava_data.pyを実行してください。")
        sys.exit(1)
    df = preprocess_df(df, activity_type="Run")
    fig1 = plot_basic_stat_from_df(df)
    fig2 = plot_detailed_stat_from_df(df)
    fig3 = scatter_plots_from_df(df)
    fig4 = scatter_3d_plot_from_df(df)
    fig5 = plot_pie_chart_from_df(df)
    fig6 = plot_histograms_from_df(df)
    fig7 = plot_weather_data(df)

    plt.show()

    # Save figures
    if not os.path.exists("images"):
        os.makedirs("images")
    fig1.savefig("images/basic_stat.png", dpi=SAVE_FIG_DPI)
    fig2.savefig("images/detailed_stat.png", dpi=SAVE_FIG_DPI)
    fig3.savefig("images/scatter_plots.png", dpi=SAVE_FIG_DPI)
    fig4.savefig("images/scatter_3d_plot.png", dpi=SAVE_FIG_DPI)
    fig5.savefig("images/pie_chart.png", dpi=SAVE_FIG_DPI)
    fig6.savefig("images/histograms.png", dpi=SAVE_FIG_DPI)
    fig7.savefig("images/weather_data.png", dpi=SAVE_FIG_DPI)


# TODO: Plot data based on different activities (running, climbing, etc.) in different colors and legends
# TODO: Analyze data with different metrics (e.g. time, location, weather, temperature, etc.)
# TODO: With machine learning, identify correlation between different metrics and predict future performance (e.g. speed, distance, etc.) with data from the past and new data (e.g. food intake, sleep, mood, weather, temperature, etc.)
