import pandas as pd
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_as_csv(df: pd.DataFrame, filename: str) -> None:
    try:
        df.to_csv(filename, index=False, encoding="utf-8")
        logging.info(f"File successfully saved as {filename}")
    except Exception as e:
        logging.error(f"Error saving data as CSV file: {e}")
        raise


def transform_regional_prices_to_csv(file_names):
    output_dataset = pd.DataFrame(
        columns=[
            "time_period",
            "region_name",
            "house_price_avg",
            "house_price_2k",
            "house_price_2k_10k",
            "house_price_10k_50k",
            "house_price_50k_plus",
            "flat_price_avg",
            "flat_price_2k",
            "flat_price_2k_10k",
            "flat_price_10k_50k",
            "flat_price_50k_plus",
        ]
    )

    row_idx = 0
    for file_name, time_period in file_names:
        try:
            data = pd.read_excel(file_name, skiprows=7, engine="openpyxl")
            data = data.iloc[:, 2:13]
            data = data[:-4]
            data.insert(0, "time_period", time_period)

            logging.info(f"Processing file: {file_name}")
            for _, row in data.iterrows():
                output_dataset.loc[row_idx] = row.values
                row_idx += 1

        except Exception as e:
            logging.error(f"Error reading Excel content: {e}")
            raise

    return output_dataset


def transform_avg_regional_prices_to_csv(file_path):
    try:
        data = pd.read_excel(file_path, skiprows=5, engine="openpyxl")
        data = data.iloc[:, 1:12]
        data = data[:-4]

        year_headers = data.iloc[0]
        data = data[1:]

        new_columns = ["region"]
        valid_column_indices = []

        for idx, label in enumerate(year_headers[1:], start=1):
            if isinstance(label, str) and re.match(r"^\D*(\d{4})\D*$", label):
                new_columns.append(label)
                valid_column_indices.append(idx)

        selected_columns = [0] + valid_column_indices
        data = data.iloc[:, selected_columns]
        data.columns = new_columns

        long_df = pd.melt(data, id_vars="region", var_name="year", value_name="price")
        long_df = long_df.dropna(subset=["price"])
        long_df["year"] = long_df["year"].astype(str).str.extract(r"(\d{4})")
        long_df["price"] = pd.to_numeric(long_df["price"], errors="coerce")
        long_df = long_df.dropna(subset=["year", "price"])

        rodinne_domy_years = year_headers[1:9].dropna().astype(str).str.extract(r"(\d{4})")[0].dropna().tolist()
        byty_years = year_headers[9:].dropna().astype(str).str.extract(r"(\d{4})")[0].dropna().tolist()

        def determine_type(year):
            if year in rodinne_domy_years:
                return "Rodinné domy"
            elif year in byty_years:
                return "Byty"
            return None

        long_df["type"] = long_df["year"].apply(determine_type)
        long_df = long_df.dropna(subset=["type"])

        return long_df

    except Exception as e:
        logging.error(f"Error reading Excel content: {e}")
        raise


def transform_ooh_data_to_csv(file_names):
    output_dataset = pd.DataFrame(columns=["region", "type", "year", "quarter", "index_value"])

    for file_name, year in file_names:
        try:
            data = pd.read_excel(file_name, skiprows=7, engine="openpyxl")
            data = data.iloc[:, 1:12]
            data = data[:-6]

            logging.info(f"Processing file: {file_name}")
            header_row = data.iloc[0].astype(str).str.strip()
            data = data[1:]
            data.columns = ["region"] + list(header_row[1:])

            region_col = data["region"]
            domy_values = data.iloc[:, 1:6]
            byty_values = data.iloc[:, 6:11]
            domy_quarters = header_row[1:6]
            byty_quarters = header_row[6:11]

            def normalize_quarter(q):
                if isinstance(q, str) and q.strip().startswith("Q"):
                    return q.strip()[1]
                return "year"

            domy_df = pd.concat([region_col.reset_index(drop=True), domy_values.reset_index(drop=True)], axis=1)
            domy_df.columns = ["region"] + domy_quarters.tolist()
            domy_df = domy_df.melt(id_vars="region", var_name="quarter", value_name="index_value")
            domy_df["quarter"] = domy_df["quarter"].apply(normalize_quarter)
            domy_df["type"] = "Domy"
            domy_df["year"] = str(year)

            byty_df = pd.concat([region_col.reset_index(drop=True), byty_values.reset_index(drop=True)], axis=1)
            byty_df.columns = ["region"] + byty_quarters.tolist()
            byty_df = byty_df.melt(id_vars="region", var_name="quarter", value_name="index_value")
            byty_df["quarter"] = byty_df["quarter"].apply(normalize_quarter)
            byty_df["type"] = "Byty"
            byty_df["year"] = str(year)

            combined = pd.concat([domy_df, byty_df])
            combined = combined.dropna(subset=["index_value"])
            combined = combined[["region", "type", "year", "quarter", "index_value"]]
            output_dataset = pd.concat([output_dataset, combined], ignore_index=True)

        except Exception as e:
            logging.error(f"Error reading Excel content from {file_name}: {e}")
            raise

    return output_dataset


if __name__ == "__main__":
    # Input Excel files
    input_files = [
        ("data/excel/regional_prices_19-21.xlsx", "2019-2021"),
        ("data/excel/regional_prices_20-22.xlsx", "2020-2022"),
        ("data/excel/regional_prices_21-23.xlsx", "2021-2023"),
    ]

    ooh_input_files = [
        ("data/excel/ooh_19.xlsx", "2019"),
        ("data/excel/ooh_20.xlsx", "2020"),
        ("data/excel/ooh_21.xlsx", "2021"),
        ("data/excel/ooh_22.xlsx", "2022"),
        ("data/excel/ooh_23.xlsx", "2023"),
    ]

    avg_prices_input_file = "data/excel/avg_prices_regions.xlsx"

    # Output paths
    os.makedirs("data/csv", exist_ok=True)

    # Transformations
    regional_prices = transform_regional_prices_to_csv(input_files)
    save_as_csv(regional_prices, "data/csv/regional_prices.csv")

    regional_avg = transform_avg_regional_prices_to_csv(avg_prices_input_file)
    save_as_csv(regional_avg, "data/csv/avg_prices_regions.csv")

    ooh_data = transform_ooh_data_to_csv(ooh_input_files)
    save_as_csv(ooh_data, "data/csv/ooh_data.csv")
