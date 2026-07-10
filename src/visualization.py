"""
visualization.py

Creates all graphs used in the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class Visualizer:

    def __init__(self):

        self.output = Path("outputs/figures")

        self.output.mkdir(parents=True, exist_ok=True)

    # --------------------------------

    def traffic_distribution(self, train):

        plt.figure(figsize=(8, 5))

        sns.histplot(train["Vehicles"], bins=40, kde=True)

        plt.title("Traffic Distribution")

        plt.tight_layout()

        plt.savefig(self.output / "traffic_distribution.png", dpi=300)

        plt.close()

    # --------------------------------

    def hourly_pattern(self, train):

        hourly = train.groupby("Hour")["Vehicles"].mean()

        plt.figure(figsize=(10, 5))

        plt.plot(hourly.index, hourly.values, marker="o")

        plt.title("Average Hourly Traffic")

        plt.xlabel("Hour")

        plt.ylabel("Vehicles")

        plt.tight_layout()

        plt.savefig(self.output / "hourly_pattern.png", dpi=300)

        plt.close()

    # --------------------------------

    def weekday_pattern(self, train):

        weekday = train.groupby("Weekday")["Vehicles"].mean()

        plt.figure(figsize=(8, 5))

        plt.bar(weekday.index, weekday.values)

        plt.title("Weekday Traffic")

        plt.xlabel("Weekday")

        plt.ylabel("Vehicles")

        plt.tight_layout()

        plt.savefig(self.output / "weekday_pattern.png", dpi=300)

        plt.close()

    # --------------------------------

    def junction_comparison(self, train):

        plt.figure(figsize=(8, 5))

        sns.boxplot(x="Junction", y="Vehicles", data=train)

        plt.tight_layout()

        plt.savefig(self.output / "junction_comparison.png", dpi=300)

        plt.close()

    # --------------------------------

    def correlation_heatmap(self, train):

        numeric = train.select_dtypes(include="number")

        plt.figure(figsize=(12, 8))

        sns.heatmap(numeric.corr(), cmap="coolwarm")

        plt.tight_layout()

        plt.savefig(self.output / "correlation_heatmap.png", dpi=300)

        plt.close()

    # --------------------------------

    def generate_all(self, train):

        print("=" * 60)
        print("Generating Visualizations")
        print("=" * 60)

        self.traffic_distribution(train)

        self.hourly_pattern(train)

        self.weekday_pattern(train)

        self.junction_comparison(train)

        self.correlation_heatmap(train)

        print("Visualizations Generated.")
