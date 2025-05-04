# Czech Real Estate Analysis

## Introduction

This project was created to better understand long-term trends and regional disparities in the Czech property market. The analysis covers both aggregate and segmented price indicators (e.g., price bands), including quarterly and annual trends. Various visualization techniques such as heatmaps, choropleth maps, and statistical plots were used to gain insight into how different regions of the country compare and evolve over time.

Data Source: [ČSÚ (Czech Statistical Office)](https://vdb.czso.cz/vdbvo2/faces/cs/index.jsf?page=statistiky&katalog=31782)

## Questions to Answer

The following questions are explored in this analysis and visualized using maps, heatmaps, and statistical plots:

| #   | Question                                                              |
| --- | --------------------------------------------------------------------- |
| Q1  | **Where was it most expensive to buy a flat in 2023?**                |
| Q2  | **Which district grew the fastest in each region from 2019 to 2023?** |
| Q3  | **Is post-COVID demand favouring smaller cities for houses?**         |
| Q4  | **In which regions is the flat market the most uneven?**              |
| Q5  | **Which regions became more or less expensive since 2019?**           |
| Q6  | **How have prices changed across city sizes in Středočeský kraj?**    |
| Q7  | **Was there a peak in flat prices — and if so, when?**                |
| Q8  | **Are houses or flats gaining more value since 2019?**                |

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
   python main.py <optional_show_flag>
   ```

3. Outputs (plots, graphs) will be saved to the `graphs/` directory.

## Results

All visual outputs are saved in the `outputs/` directory as `.png` or `.svg` files.

## License

This project is licensed under the MIT License.
