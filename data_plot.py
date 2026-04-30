import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pm25_trend(df: pd.DataFrame):
    
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    daily_avg = df.groupby("date")["PM2.5"].mean()

    plt.figure(figsize=(14, 7))
    plt.plot(daily_avg.index, daily_avg.values)
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.title("Daily Average PM2.5")
    plt.tight_layout()
    plt.savefig("eda_pm25_trend.pdf")
    plt.show()


def plot_correlation(df: pd.DataFrame):
    
    columns = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "hour", "month"
    ]

    cols = [i for i in columns]
    df_ = df[cols]
    correlation = df_.corr(numeric_only=True)
    
    plt.figure(figsize=(14, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm",center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.pdf")
    plt.show()


def plot_histogram_pm25(df: pd.DataFrame):
        
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    pm25_freq = df.groupby("date")["PM2.5"].mean()

    plt.figure(figsize=(14, 7))
    plt.hist(pm25_freq, bins = 30)
    plt.grid()
    plt.xlabel("PM2.5")
    plt.ylabel("Frequency")
    plt.ylim(0, 240)
    plt.title("PM2.5 Histogram")
    plt.tight_layout()
    plt.savefig("eda_pm25_histogram.pdf")
    plt.show()



if __name__ == "__main__":
    path = r"C:\Users\hussa\air_quality_cleaned.csv"
    df = pd.read_csv(path)
    plot_pm25_trend(df)