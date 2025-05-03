from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
import geojson
import json
import datetime
import numpy as np
import urllib.request as urllib2
from matplotlib.backends.backend_pdf import PdfPages
import os
import logging

"""
Which regions had the highest average flat prices in 2023?
Which regions had the largest gap between house and flat prices?
Which regions saw the greatest percentage increase in property prices over time?
Correlation heatmap between different price bands (e.g., 2k–10k, 10k–50k) across regions
Which regions consistently had high prices across years (top 5 every year)?
Comparison of price trend stability across regions
Which regions show the highest disparity between house and flat prices?
Which regions have the most volatile property prices?
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    return df


def save_figure_to_pdf(fig, filename):
    """
    Save a figure to a PDF file.
    """
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_flat_prices_2023(prices_df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Plots a choropleth map of the Czech Republic showing average flat (Byty) prices in 2023.
    Adds missing Praha if needed and displays price labels.
    """
    # Filter only flat prices for 2023
    df = prices_df[(prices_df["type"] == "Byty") & (prices_df["year"] == 2023)].copy()
    df = df.rename(columns={"region": "name", "price": "TOTAL"})

    # Add Praha if missing
    if "Hlavní město Praha" not in df["name"].values:
        value = df.loc[df["name"] == "Středočeský kraj", "TOTAL"].iloc[0]
        df = pd.concat([df, pd.DataFrame([{"name": "Hlavní město Praha", "TOTAL": value}])], ignore_index=True)

    # Load Czech region GeoJSON
    url = "https://raw.githubusercontent.com/Plavit/Simple-Dash-Plotly-Map-Czech-Regions/main/maps/czech-regions-low-res.json"
    with urllib2.urlopen(url) as f:
        geo = gpd.read_file(f)

    # Merge prices with GeoDataFrame
    geo = geo.merge(df, on="name", how="left")

    # Plot map
    fig, ax = plt.subplots(figsize=(10, 6))
    geo.plot(column="TOTAL", ax=ax, legend=True, cmap="OrRd", edgecolor="black")

    # Add value labels
    for _, row in geo.iterrows():
        if pd.notna(row["TOTAL"]):
            point = row["geometry"].representative_point()
            label_color = "gray" if row["name"] == "Hlavní město Praha" else "black"
            ax.text(
                point.x,
                point.y,
                f'{int(row["TOTAL"]):,}'.replace(",", " "),
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
                weight="bold",
            )

    ax.set_title("Average Flat Prices in Czech Regions (2023)", fontsize=14)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def main():
    avg_region_prices_df = load_csv_to_dataframe("data/csv/avg_prices_regions.csv")
    ooh_df = load_csv_to_dataframe("data/csv/ooh_data.csv")
    regional_prices_df = load_csv_to_dataframe("data/csv/regional_prices.csv")

    q1_plot = plot_flat_prices_2023(avg_region_prices_df, show=False)

    os.makedirs("output", exist_ok=True)
    save_figure_to_pdf(q1_plot, "output/flat_prices_2023.pdf")


if __name__ == "__main__":
    main()
