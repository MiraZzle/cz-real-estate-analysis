# Czech Real Estate Analysis

## Introduction

This project was created to better understand long-term trends and regional disparities in the Czech property market. The analysis covers both aggregate and segmented price indicators (e.g., price bands), including quarterly and annual trends. Various visualization techniques such as heatmaps, choropleth maps, and statistical plots were used to gain insight into how different regions of the country compare and evolve over time.

Data Source: [ČSÚ (Czech Statistical Office)](https://vdb.czso.cz/vdbvo2/faces/cs/index.jsf?page=statistiky&katalog=31782)

## Questions to Answer

The following questions are explored in this analysis and visualized using maps, heatmaps, and statistical plots:

- Which regions had the highest average flat prices in 2023?
- Which regions had the largest gap between house and flat prices?
- Which regions saw the greatest percentage increase in property prices over time?
- Correlation heatmap between different price bands (e.g., 2k–10k, 10k–50k) across regions
- Which regions consistently had high prices across years (top 5 every year)?
- Comparison of price trend stability across regions
- Which regions show the highest disparity between house and flat prices?
- Which regions have the most volatile property prices?

## System Requirements

- Python 3.10+
- pip
- Recommended: virtual environment (`venv`)

## Installation

### Clone the repository

```bash
git clone https://github.com/MiraZzle/cz-real-estate-analysis
cd cz-real-estate-analysis
```

### Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

1. If needed, transform the raw Excel files to CSV format by running:

   ```bash
   python transform_data_format.py
   ```

   (Transformed CSVs are already included in the repository under `data/csv`.)

2. Run the analysis:

   ```bash
   python main.py
   ```

3. Outputs (plots, graphs) will be saved to the `graphs/` directory.

## Results

All visual outputs are saved in the `graphs/` directory as `.png` or `.svg` files. These include:

- Choropleth maps of property prices across the Czech Republic
- Heatmaps of quarterly and annual trends
- Line plots showing temporal trends per region
- Correlation matrices and volatility measures

## License

This project is licensed under the MIT License.
