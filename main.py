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
from utils import load_csv_to_dataframe, save_figure_to_pdf, save_figure_to_png, KRAJ_DISTRICTS_RAW, KRAJE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_top_growing_districts(data: pd.DataFrame) -> pd.DataFrame:
    """
    From raw avg_prices_regions.csv data, computes the top-growing district in each kraj
    based on percentual flat price growth (2019–2023).
    """
    lines = [line.strip() for line in KRAJ_DISTRICTS_RAW.strip().splitlines()]
    district_to_kraj = {}
    current_kraj = None
    for line in lines:
        if "kraj" in line or "Praha" in line:
            current_kraj = "Vysočina" if line == "Kraj Vysočina" else line
        else:
            district_to_kraj[line] = current_kraj

    # Map kraje
    data = data.copy()
    data["kraj"] = data["region"].map(district_to_kraj)

    # Filter for flats and years
    filtered = data[(data["type"] == "Byty") & (data["year"].isin([2019, 2023]))]

    # Pivot to get 2019 and 2023 columns
    pivot = filtered.pivot_table(index=["region", "kraj"], columns="year", values="price")
    pivot = pivot.dropna(subset=[2019, 2023])
    pivot["growth_index"] = (pivot[2023] / pivot[2019]) * 100
    pivot = pivot.reset_index()

    # Get top-growing district per kraj
    top_districts = pivot.loc[pivot.groupby("kraj")["growth_index"].idxmax()]
    return top_districts


def plot_growth_scatter_with_adjustments(df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Q: Which district grew the fastest in each kraj from 2019 to 2023??

    Parameters:
        df (pd.DataFrame): DataFrame containing average flat prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated scatter plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    df = prepare_top_growing_districts(df)

    x = df[2019]
    y = df["growth_index"]
    labels = df["kraj"] + ": " + df["region"]
    prices_2023 = df[2023]

    # Top 3 by growth
    top3 = df.nlargest(3, "growth_index")

    # Plot all points
    ax.scatter(x, y, color="royalblue", s=60, zorder=2)

    # Highlight top 3 in red
    ax.scatter(top3[2019], top3["growth_index"], color="crimson", s=80, zorder=3, edgecolors="black")
    x_range = x.max() - x.min()

    # Annotate each point
    for xi, yi, label, price, region in zip(x, y, labels, prices_2023, df["region"]):
        dx, dy = 0, 1.5  # general upward shift
        dx = 0.01 * x_range  # e.g., shift by ~100–200 CZK
        if region == "Prachatice":
            dy = 2.5
        if region == "Jeseník":
            dy = -4
            dx = 0.02 * x_range
        if region == "Kolín":
            dx = -0.08 * x_range
            dy = 0
        if region == "Blansko":
            dx = -0.03 * x_range

            # Draw pointer line from original position to label
        if region in ["Prachatice", "Jeseník"]:
            ax.plot([xi, xi + dx], [yi, yi + dy], color="blue", linewidth=0.8, linestyle="--", zorder=1)

        ax.text(xi + dx, yi + dy, label, fontsize=8, ha="center", va="bottom")
        ax.text(xi, yi - 0.85, f"{int(price):,} CZK", fontsize=7, ha="center", va="top", color="gray")

    ax.set_xlabel("Flat Price in 2019 (CZK/m²)")
    ax.set_ylabel("Normalized Growth (2019 = 100%)")
    ax.set_title("Growth vs. Starting Price for Top-Growing Districts per Kraj (2019–2023)\n2023 Prices Shown")
    buffer_x = (x.max() - x.min()) * 0.05
    buffer_y = (y.max() - y.min()) * 0.05
    ax.set_xlim(x.min() - buffer_x, x.max() + buffer_x)
    ax.set_ylim(y.min() - buffer_y, y.max() + buffer_y + 3)

    ax.grid(True)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_kraj_disparity_from_avg_prices(data: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Q: Which regions have the largest price disparities between their districts in 2023?

    Parameters:
        data (pd.DataFrame): DataFrame containing average flat prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated choropleth map figure
    """
    lines = [line.strip() for line in KRAJ_DISTRICTS_RAW.strip().splitlines()]
    district_to_kraj = {}
    current_kraj = None
    for line in lines:
        if line == "Kraj Vysočina":
            current_kraj = line
        elif "kraj" in line or "Praha" in line:
            current_kraj = line
        else:
            district_to_kraj[line] = current_kraj

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

    url = "https://raw.githubusercontent.com/Plavit/Simple-Dash-Plotly-Map-Czech-Regions/main/maps/czech-regions-low-res.json"
    with urllib2.urlopen(url) as f:
        geo = gpd.read_file(f)

    geo = geo.merge(disparity, on="name", how="left")

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
    How do flat vs house price growth compare across kraje from 2019 to 2023?

    Parameters:
        data (pd.DataFrame): DataFrame containing average flat prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated bar plot figure
    """

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
    pivot = pivot[pivot["region"].isin(KRAJE_NAMES)]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=pivot, x="region", y="growth_index", hue="type", ax=ax)

    ax.set_title("Percentual Price Growth by Region (2019 -> 2023)")
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
    Q: Which regions saw the largest flat price growth year-by-year?

    Parameters:
        data (pd.DataFrame): DataFrame containing average flat prices by region and year
        property_type (str): Type of property to filter for (default: "Byty")
        top_n (int): Number of top regions to display (default: 10)
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated line plot figure
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
    Q: Which regions had the highest average flat prices in 2023?

    Parameters:
        prices_df (pd.DataFrame): DataFrame containing average flat prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated choropleth map figure
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

    ax.set_title("Average Flat Price in 2023 by Region", fontsize=14)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def plot_house_price_shift_by_city_size(regional_df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Q: Have house prices grown faster in small vs large cities after COVID?

    Parameters:
        regional_df (pd.DataFrame): DataFrame containing house prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated boxplot figure
    """
    # Filter for the two time periods
    df_early = regional_df[regional_df["time_period"] == "2019-2021"]
    df_late = regional_df[regional_df["time_period"] == "2021-2023"]

    # Select relevant house price columns by population group
    melt_cols = ["house_price_2k_10k", "house_price_10k_50k", "house_price_50k_plus"]

    # Melt into long format
    melted_early = df_early[melt_cols].melt(var_name="pop_group", value_name="price").dropna()
    melted_late = df_late[melt_cols].melt(var_name="pop_group", value_name="price").dropna()

    # Rename population groups for readability
    group_map = {"house_price_2k_10k": "2k–10k", "house_price_10k_50k": "10k–50k", "house_price_50k_plus": "50k+"}
    melted_early["pop_group"] = melted_early["pop_group"].map(group_map)
    melted_late["pop_group"] = melted_late["pop_group"].map(group_map)

    # Add period labels
    melted_early["period"] = "2019–2021"
    melted_late["period"] = "2021–2023"

    # Combine and clean
    combined = pd.concat([melted_early, melted_late], ignore_index=True)
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")
    combined = combined.dropna(subset=["price"])

    # Create boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.boxplot(data=combined[combined["period"] == "2019–2021"], x="pop_group", y="price", ax=axes[0])
    axes[0].set_title("House Prices by City Size (2019–2021)")
    axes[0].set_xlabel("City Size")
    axes[0].set_ylabel("Price per m²")
    axes[0].grid(True)

    sns.boxplot(data=combined[combined["period"] == "2021–2023"], x="pop_group", y="price", ax=axes[1])
    axes[1].set_title("House Prices by City Size (2021–2023)")
    axes[1].set_xlabel("City Size")
    axes[1].set_ylabel("Price per m²")
    axes[1].grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kraje_price_rankings(avg_df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Q: How does the average flat price ranking of regions change over time?

    Parameters:
        avg_df (pd.DataFrame): DataFrame containing average flat prices by region and year
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated line plot figure
    """

    # Filter for flats and only kraj-level regions
    kraje_df = (
        avg_df[(avg_df["region"].isin(KRAJE_NAMES)) & (avg_df["type"] == "Byty") & (avg_df["year"].between(2019, 2023))]
        .groupby(["region", "year"])["price"]
        .mean()
        .reset_index()
    )

    # Compute rank per year
    kraje_df["rank"] = kraje_df.groupby("year")["price"].rank(ascending=False, method="min")

    # Pivot to wide format
    rank_pivot = kraje_df.pivot(index="region", columns="year", values="rank").dropna()

    # Assign unique colors to each region
    palette = sns.color_palette("tab20", len(rank_pivot))
    color_map = {region: palette[i] for i, region in enumerate(rank_pivot.index)}

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for region, row in rank_pivot.iterrows():
        ax.plot(row.index, row.values, marker="o", label=region, color=color_map[region], linewidth=2)

    ax.set_title("Kraje Ranking by Average Flat Prices (2019–2023)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rank (1 = Most Expensive)")
    ax.set_yticks(range(1, len(KRAJE_NAMES) + 1))
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", title="Kraj")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_population_price_slope_chart(df: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Q: How have average flat and house prices changed for different population groups in Středočeský kraj?

    Parameters:
        df (pd.DataFrame): Input regional dataset with detailed price columns
        show (bool): Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated slope chart figure
    """
    flat_cols = ["flat_price_2k", "flat_price_2k_10k", "flat_price_10k_50k", "flat_price_50k_plus"]
    house_cols = ["house_price_2k", "house_price_2k_10k", "house_price_10k_50k", "house_price_50k_plus"]
    all_cols = flat_cols + house_cols

    # Filter and clean
    region_df = df[
        (df["region_name"] == "Středočeský kraj") & df["time_period"].isin(["2019-2021", "2021-2023"])
    ].copy()
    region_df[all_cols] = region_df[all_cols].replace("x", pd.NA).apply(pd.to_numeric, errors="coerce")

    # Aggregate
    avg = region_df.groupby("time_period")[all_cols].mean().T
    avg.index = [
        "Flat <2k",
        "Flat 2k–10k",
        "Flat 10k–50k",
        "Flat 50k+",
        "House <2k",
        "House 2k–10k",
        "House 10k–50k",
        "House 50k+",
    ]

    # Reshape
    slope_data = avg.T.reset_index().rename(columns={"index": "time_period"})
    slope_data = slope_data.melt(id_vars="time_period", var_name="group", value_name="price")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in slope_data["group"].unique():
        subset = slope_data[slope_data["group"] == group]
        ax.plot(subset["time_period"], subset["price"], marker="o", label=group)

    ax.set_title(f"Price Change by Population Group in Středočeský kraj (2019–2023)", fontsize=14)
    ax.set_ylabel("Average Price (CZK/m²)")
    ax.set_xlabel("Time Period")
    ax.set_xticks(["2019-2021", "2021-2023"])
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Group")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def main(show_plots: bool):
    avg_region_prices_df = load_csv_to_dataframe("data/csv/avg_prices_regions.csv")
    regional_prices_df = load_csv_to_dataframe("data/csv/regional_prices.csv")

    q1_plot = plot_flat_prices_2023(avg_region_prices_df, show=show_plots)
    q2_plot = plot_price_growth_over_years(avg_region_prices_df, property_type="Byty", top_n=10, show=show_plots)
    q3_plot = plot_growth_comparison_kraje(avg_region_prices_df, show=show_plots)
    q4_plot = plot_kraj_disparity_from_avg_prices(avg_region_prices_df, show=show_plots)
    q5_plot = plot_growth_scatter_with_adjustments(avg_region_prices_df, show=show_plots)
    q6_plot = plot_house_price_shift_by_city_size(regional_prices_df, show=show_plots)
    q7_plot = plot_kraje_price_rankings(avg_region_prices_df, show=show_plots)
    q8_plot = plot_population_price_slope_chart(regional_prices_df, show=show_plots)

    pdf_dir = "output/pdfs"
    png_dir = "output/pngs"

    os.makedirs(pdf_dir, exist_ok=True)
    save_figure_to_pdf(q1_plot, f"{pdf_dir}/flat_prices_2023.pdf")
    save_figure_to_pdf(q2_plot, f"{pdf_dir}/price_growth_byty.pdf")
    save_figure_to_pdf(q3_plot, f"{pdf_dir}/price_growth_comparison_kraje.pdf")
    save_figure_to_pdf(q4_plot, f"{pdf_dir}/kraj_flat_price_disparity.pdf")
    save_figure_to_pdf(q5_plot, f"{pdf_dir}/growth_scatter.pdf")
    save_figure_to_pdf(q6_plot, f"{pdf_dir}/house_price_shift_by_city_size.pdf")
    save_figure_to_pdf(q7_plot, f"{pdf_dir}/kraje_price_rankings.pdf")
    save_figure_to_pdf(q8_plot, f"{pdf_dir}/population_price_slope_chart.pdf")

    os.makedirs(png_dir, exist_ok=True)
    save_figure_to_png(q1_plot, f"{png_dir}/flat_prices_2023.png")
    save_figure_to_png(q2_plot, f"{png_dir}/price_growth_byty.png")
    save_figure_to_png(q3_plot, f"{png_dir}/price_growth_comparison_kraje.png")
    save_figure_to_png(q4_plot, f"{png_dir}/kraj_flat_price_disparity.png")
    save_figure_to_png(q5_plot, f"{png_dir}/growth_scatter.png")
    save_figure_to_png(q6_plot, f"{png_dir}/house_price_shift_by_city_size.png")
    save_figure_to_png(q7_plot, f"{png_dir}/kraje_price_rankings.png")
    save_figure_to_png(q8_plot, f"{png_dir}/population_price_slope_chart.png")


if __name__ == "__main__":
    show_plots = False
    try:
        show_plots = bool(sys.argv[1])
    except IndexError:
        logging.info("No command line argument provided for show_plots. Defaulting to False.")
    main(show_plots)
