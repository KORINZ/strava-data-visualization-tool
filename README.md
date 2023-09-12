# Strava Data Visualization Tool

This project uses Python scripts to visualize your Strava activity data using pandas and matplotlib. You can use it to see patterns, compare different types of workouts, understand your performance over time, and more.

## Sample Images

![Figure_1](https://github.com/KORINZ/strava-data-visualization-tool/assets/111611023/2d78093d-4e38-4503-b0ad-ebb90fa6b914)
![Figure_3](https://github.com/KORINZ/strava-data-visualization-tool/assets/111611023/07457b1d-92b5-4ab2-9d04-55fed39e07f1)
![Figure_7](https://github.com/KORINZ/strava-data-visualization-tool/assets/111611023/cdb6cd18-1f86-4b54-8ed6-37631112024c)

## Summary

* Fetches your Strava activity data using Strava's API.
* Visualizes the data using various plots like scatter plots, line plots, bar charts, pie charts, 3D plots, and histograms.
* Plots include distance vs. average pace, altitude gains, duration, heart rate, etc.
* Fetches and merges weather data like temperature, humidity, and air pressure into your activity data using the Meteostat API.
* Generates your data's correlation matrix and scatter matrix to help you find interesting relationships.

## Requirements

You will need Python 3.11 or later to run this tool.

Additionally, the following Python libraries are used:

* `stravalib`
* `pandas`
* `matplotlib`
* `numpy`
* `meteostat`
* `seaborn`

You can install these with `pip` by running: `pip install -r requirements.txt`

You will also need a Strava API key, which you can get by creating an application on [Strava Developers](https://developers.strava.com/).

## Usage

1. Clone this repository: `git clone https://github.com/KORINZ/strava-data-visualization-tool.git`
2. Install the required Python libraries: `pip install -r requirements.txt`
3. Run `get_strava_data.py`; enter your ID and API token.
4. Choose whether to fetch all calorie data or not (check your API call limit).
5. Run the `get_weather_data.py` to fetch temperature, humanity, and air pressure data based on your start location and time of the activity.
6. Run the `main.py` to visualize the data. Images will be saved to the images folder.
7. You can also run `explore_data.py` to see correlation matrix plots.

## Features

1. **Scatter plots**: Two scatter plots are available, one plotting distance vs. average pace, colored by average heart rate, and the other plotting time of day vs. distance, colored by kudos.

2. **Basic statistics plots**: This includes a bar chart of distance, a line chart of distance, a bar chart of altitude gains, and a scatter chart of duration.

3. **Detailed statistics plots**: This includes a scatter plot of distance, a bar chart of calories burned, a scatter plot of average heart rate, and a scatter plot of the average pace.

4. **3D scatter plot**: You can generate a 3D scatter plot of average pace vs. duration vs. temperature, colored by average heart rate.

5. **Pie chart**: This tool can create a pie chart showing the proportion of activities during different periods of the day.

6. **Bar plots**: It also provides bar plots for Day of Week, Month, and Year, showing the frequency of activities.

7. **Histograms**: You can create histograms of distance and duration.

Please note that some of the labels and comments in the code are in Japanese.
