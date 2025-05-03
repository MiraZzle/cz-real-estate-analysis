from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
import geojson
import json
import seaborn as sns
import numpy as np
import urllib.request as urllib2
from matplotlib.backends.backend_pdf import PdfPages
import os
import logging
import sys

"""
Which regions had the highest average flat prices in 2023?
Which regions had the largest gap between house and flat prices?
Which regions saw the greatest percentage increase in property prices over time?
Correlation heatmap between different price bands (e.g., 2k–10k, 10k–50k) across regions
Which regions consistently had high prices across years (top 5 every year)?
Comparison of price trend stability across regions
Which regions show the highest disparity between house and flat prices?
Which regions have the most volatile property prices?

----

Otazky primo na regionalni ceny:
Jake regiony mely rust cen bytu v letech 2019-2021?
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


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import urllib.request as urllib2


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import urllib.request as urllib2
import numpy as np


def plot_kraj_disparity_from_avg_prices(data: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Computes and plots the intra-kraj disparity in flat prices (max - min across districts)
    for each Czech region based on avg_prices_regions.csv data.
    Praha is shown but has no value.
    """
    # --- District to kraj mapping ---
    kraj_districts_raw = """
    Hlavní město Praha
    Středočeský kraj
    Benešov
    Beroun
    Kladno
    Kolín
    Kutná Hora
    Mělník
    Mladá Boleslav
    Nymburk
    Praha-východ
    Praha-západ
    Příbram
    Rakovník
    Jihočeský kraj
    České Budějovice
    Český Krumlov
    Jindřichův Hradec
    Písek
    Prachatice
    Strakonice
    Tábor
    Plzeňský kraj
    Domažlice
    Klatovy
    Plzeň-jih
    Plzeň-město
    Plzeň-sever
    Rokycany
    Tachov
    Karlovarský kraj
    Cheb
    Karlovy Vary
    Sokolov
    Ústecký kraj
    Děčín
    Chomutov
    Litoměřice
    Louny
    Most
    Teplice
    Ústí nad Labem
    Liberecký kraj
    Česká Lípa
    Jablonec nad Nisou
    Liberec
    Semily
    Královéhradecký kraj
    Hradec Králové
    Jičín
    Náchod
    Rychnov nad Kněžnou
    Trutnov
    Pardubický kraj
    Chrudim
    Pardubice
    Svitavy
    Ústí nad Orlicí
    Kraj Vysočina
    Havlíčkův Brod
    Jihlava
    Pelhřimov
    Třebíč
    Žďár nad Sázavou
    Jihomoravský kraj
    Blansko
    Brno-město
    Brno-venkov
    Břeclav
    Hodonín
    Vyškov
    Znojmo
    Olomoucký kraj
    Jeseník
    Olomouc
    Prostějov
    Přerov
    Šumperk
    Zlínský kraj
    Kroměříž
    Uherské Hradiště
    Vsetín
    Zlín
    Moravskoslezský kraj
    Bruntál
    Frýdek-Místek
    Karviná
    Nový Jičín
    Opava
    Ostrava-město
    """
    lines = [line.strip() for line in kraj_districts_raw.strip().splitlines()]
    district_to_kraj = {}
    current_kraj = None
    for line in lines:
        if line == "Kraj Vysočina":
            current_kraj = line
        elif "kraj" in line or "Praha" in line:
            current_kraj = line
        else:
            district_to_kraj[line] = current_kraj

    # --- Prepare data ---
    data = data.copy()
    data["kraj"] = data["region"].map(district_to_kraj)

    # Filter flats in 2023
    flats = data[(data["type"] == "Byty") & (data["year"] == 2023)]

    # Compute disparity
    disparity = (
        flats[flats["kraj"].notna()]
        .groupby("kraj")["price"]
        .agg(lambda x: x.max() - x.min())
        .reset_index()
        .rename(columns={"kraj": "name", "price": "disparity"})
    )

    # Add Praha with NaN
    disparity = pd.concat(
        [disparity, pd.DataFrame([{"name": "Hlavní město Praha", "disparity": np.nan}])], ignore_index=True
    )

    # --- Load Czech region map ---
    url = "https://raw.githubusercontent.com/Plavit/Simple-Dash-Plotly-Map-Czech-Regions/main/maps/czech-regions-low-res.json"
    with urllib2.urlopen(url) as f:
        geo = gpd.read_file(f)

    geo = geo.merge(disparity, on="name", how="left")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    geo.plot(
        column="disparity",
        ax=ax,
        cmap="OrRd",
        legend=True,
        edgecolor="black",
        missing_kwds={"color": "lightgray", "label": "No data"},
    )

    ax.set_title("Intra-Kraj Flat Price Disparity (2023)", fontsize=14)
    ax.axis("off")

    # Annotate values
    for _, row in geo.iterrows():
        if pd.notna(row["disparity"]):
            point = row["geometry"].representative_point()
            ax.text(
                point.x,
                point.y,
                f'{int(row["disparity"]):,}'.replace(",", " "),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_growth_comparison_kraje(data: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Filters raw price data, computes normalized growth (2019–2023), and plots a grouped bar chart
    for Czech kraje and Prague comparing flats and houses. Returns the Figure object.
    """
    # Define official kraje and Prague
    kraje_names = [
        "Hlavní město Praha",
        "Středočeský kraj",
        "Jihočeský kraj",
        "Plzeňský kraj",
        "Karlovarský kraj",
        "Ústecký kraj",
        "Liberecký kraj",
        "Královéhradecký kraj",
        "Pardubický kraj",
        "Vysočina",
        "Jihomoravský kraj",
        "Olomoucký kraj",
        "Zlínský kraj",
        "Moravskoslezský kraj",
    ]

    # Filter for relevant years and types
    df = data[data["year"].isin([2019, 2023]) & data["type"].isin(["Byty", "Rodinné domy"])]

    # Pivot to get 2019 and 2023 prices as columns
    pivot = df.pivot_table(index=["region", "type"], columns="year", values="price")

    # Drop rows missing either year
    pivot = pivot.dropna(subset=[2019, 2023])

    # Calculate normalized growth index
    pivot["growth_index"] = (pivot[2023] / pivot[2019]) * 100
    pivot = pivot.rename(columns={2019: "price_2019", 2023: "price_2023"}).reset_index()

    # Filter only kraje + Prague
    pivot = pivot[pivot["region"].isin(kraje_names)]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=pivot, x="region", y="growth_index", hue="type", ax=ax)

    ax.set_title("Percentual Price Growth by Region (2019–2023)")
    ax.set_ylabel("Index (2019 = 100%)")
    ax.set_xlabel("Region")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Property Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_price_growth_over_years(
    data: pd.DataFrame, property_type: str = "Byty", top_n: int = 10, show: bool = True
) -> plt.Figure:
    """
    Plots the yearly price growth from 2019 to 2023 for the top N regions with the highest 2023 price
    for a given property type ("Byty" or "Rodinné domy").
    """
    # Filter for the desired property type and years
    df_filtered = data[(data["type"] == property_type) & (data["year"].isin(range(2019, 2024)))]

    # Get top N regions by 2023 price
    top_regions = df_filtered[df_filtered["year"] == 2023].sort_values("price", ascending=False).head(top_n)["region"]

    # Filter data to only include top regions
    df_top = df_filtered[df_filtered["region"].isin(top_regions)]

    # Pivot to get years as index and regions as columns
    pivot = df_top.pivot_table(index="year", columns="region", values="price")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax, marker="o")

    ax.set_title(f"{property_type} Price Evolution (2019–2023) in Top {top_n} Regions", fontsize=14)
    ax.set_ylabel("Price (CZK per m²)")
    ax.set_xlabel("Year")
    ax.grid(True)
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_flat_prices_2023(prices_df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Plots a choropleth map of the Czech Republic showing average flat (Byty) prices in 2023.
    Adds missing Praha if needed and displays price labels.
    """
    # Filter only flat prices for 2023
    df = prices_df[(prices_df["type"] == "Byty") & (prices_df["year"] == 2023)].copy()
    df = df.rename(columns={"region": "name", "price": "TOTAL"})

    if "Hlavní město Praha" not in df["name"].values:
        value = df.loc[df["name"] == "Středočeský kraj", "TOTAL"].iloc[0]
        df = pd.concat([df, pd.DataFrame([{"name": "Hlavní město Praha", "TOTAL": value}])], ignore_index=True)

    # Load GeoJSON
    url = "https://raw.githubusercontent.com/Plavit/Simple-Dash-Plotly-Map-Czech-Regions/main/maps/czech-regions-low-res.json"
    with urllib2.urlopen(url) as f:
        geo = gpd.read_file(f)

    geo = geo.merge(df, on="name", how="left")

    # Plot map
    fig, ax = plt.subplots(figsize=(10, 6))
    geo.plot(column="TOTAL", ax=ax, legend=True, cmap="OrRd", edgecolor="black")

    # Add labels with adjustments
    for _, row in geo.iterrows():
        if pd.notna(row["TOTAL"]):
            point = row["geometry"].representative_point()
            x, y = point.x, point.y
            label = f'{int(row["TOTAL"]):,}'.replace(",", " ")
            color = "red" if row["name"] == "Hlavní město Praha" else "black"

            if row["name"] == "Hlavní město Praha":
                # Offset upward and draw pointer line
                ax.plot([x, x], [y, y + 0.15], color="gray", linewidth=0.6)
                ax.text(x, y + 0.17, label, ha="center", va="bottom", fontsize=8, color=color, weight="bold")

            elif row["name"] == "Olomoucký kraj":
                # Offset downward
                ax.text(x + 0.23, y - 0.12, label, ha="center", va="top", fontsize=8, color=color, weight="bold")

            else:
                ax.text(x, y, label, ha="center", va="center", fontsize=8, color=color, weight="bold")

    ax.set_title("Average Flat Prices in Czech Regions (2023)", fontsize=14)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def main(show_plots: bool):
    avg_region_prices_df = load_csv_to_dataframe("data/csv/avg_prices_regions.csv")
    ooh_df = load_csv_to_dataframe("data/csv/ooh_data.csv")
    regional_prices_df = load_csv_to_dataframe("data/csv/regional_prices.csv")

    q1_plot = plot_flat_prices_2023(avg_region_prices_df, show=show_plots)
    q2_plot = plot_price_growth_over_years(avg_region_prices_df, property_type="Byty", top_n=10, show=show_plots)
    q3_plot = plot_growth_comparison_kraje(avg_region_prices_df, show=show_plots)
    q4_plot = plot_kraj_disparity_from_avg_prices(avg_region_prices_df, show=show_plots)

    os.makedirs("output", exist_ok=True)
    save_figure_to_pdf(q1_plot, "output/flat_prices_2023.pdf")
    save_figure_to_pdf(q2_plot, "output/price_growth_byty.pdf")
    save_figure_to_pdf(q3_plot, "output/price_growth_comparison_kraje.pdf")
    save_figure_to_pdf(q4_plot, "output/kraj_flat_price_disparity.pdf")


if __name__ == "__main__":
    show_plots = False
    try:
        show_plots = bool(sys.argv[1])
    except IndexError:
        logging.info("No command line argument provided for show_plots. Defaulting to False.")
    main(show_plots)
