# GDELT GKG to Inflation Modeling

This repository contains the end-to-end pipeline to process GDELT Global Knowledge Graph (GKG) data, extract media indicators, and model their predictive impact on macroeconomic trends.

## 1. Load Data

The ingestion and initial loading of GDELT GKG records (covering 2015 to the present, totaling over 400 GB) are orchestrated by two core files:

*   **`0_load_data.py`**:
    *   Downloads raw GDELT GKG `.zip` archives (both standard and translingual streams) into a temporary buffer directory.
    *   Parses CSV tables in parallel using a multi-process architecture with PyArrow IPC streaming to eliminate Python serialization overhead.
    *   Maps textual source domains (`SourceCommonName`) to compact integer IDs (`SourceCommonName_ID`) using `gdelt_sources_mapping.json`.
    *   Compresses and stores the cleaned tables into partitioned Parquet files (`gdelt_YYYY-MM.parquet`) via asynchronous zstd-compressed writes.

*   **`1_gdelt_first_explo.ipynb`**:
    *   Mounts the full collection of monthly Parquet files into **DuckDB** using `read_parquet()` without loading the entire 400+ GB dataset into RAM.
    *   Loads and registers `gdelt_sources_mapping.json` as a relational lookup table (`src_map`) inside DuckDB.
    *   Creates baseline SQL views (`gkg`, `gkg_clean`) to verify table schemas, validate date/record formats, and prepare the dataset for out-of-core querying and downstream quality filtering.


## 2. Compute Whitelist of Sources

To ensure data quality and eliminate noise, the dataset is filtered down to reliable publishers. The pipeline maps article URLs to standard source IDs and applies historical thresholds (e.g., minimum total articles, active publishing across multiple years) to generate a strict whitelist of valid media sources.

* **Whitelist Generation Notebook `2_gdelt_to_total_sources_optimal.ipynb` **: Processes the raw GDELT data to extract source-level features, including years of activity, publication volume, thematic entropy, and economic focus. It applies strict thresholds (e.g., $\ge$ 4 active years, high thematic entropy) to filter out noise and exports the final, optimized list of approved source IDs to `gold_standard_whitelist_v2.txt`.

## 3. Compute Indicators

Using the filtered whitelist, the scripts extract and aggregate key media signals from the GDELT data. This step computes daily or weekly time-series indicators based on thematic volume, emotional tone, and specific entity mentions.

### Key Pipeline Scripts

* **`3_gdelt_indicators_general.py`:** Generates aggregated, multi-scale indicators across configurable time buckets (15-minute, daily, or monthly) and geographic scopes (global or country-level). It processes data in memory-optimized batches using DuckDB to compute sectoral and category-level attention, continuous/binary sentiment, and attention-weighted sentiment indices.


* **`3_gdelt_indicators_sources_geo_monthly.py`:** Computes granular monthly indicators disaggregated by both geographic region and individual media source. It tracks publisher-specific volume, attention shares, and sentiment metrics, mapping source IDs back to human-readable domain names for detailed comparative analysis.

## 4. Visualize Indicators

The computed media indicators are plotted. This step generates charts to observe narrative trends.

* **Long-Term & Sectoral Visualization Notebook `4_indicators.ipynb`:** Loads regional (`France`, `US`, `UK`) and global daily/monthly Parquet indicators. It features an academic plotting suite (`plot_academic_subsectors_zscore_grid` and `plot_single_subsector_zscore`) to generate rolling-window smoothed, cross-regional Z-score grids formatted for anomaly and shock detection.

* **Granular Event-Study Notebook `4_indicators_15min.ipynb`:** Analyzes high-frequency 15-minute data around acute geopolitical and market shocks. It calculates normalized abnormal attention (`att_weight`) and continuous sentiment (`sent_cont`) metrics to plot short-term narrative dispersion.

## 5. Download Data from FRED and Model (SGL)

The final step pulls official macroeconomic data from FRED (Federal Reserve Economic Data), specifically targeting inflation metrics. The GDELT media indicators are then aligned with the FRED data to train a Sparse Group Lasso (SGL) model, evaluating how specific media narratives and sentiment indicators affect or predict inflation.

* **Macroeconomic Data Ingestion & Transformation:** Downloads monthly headline inflation benchmarks and macroeconomic controls (policy interest rates, unemployment, industrial production, oil prices, FX rates, money supply) using FRED tickers (e.g., `CPIAUCSL` for the US, `CP0000FRM086NEST` for France, `GBRCPIALLMINMEI` for the UK), supplemented with the ECB API for Eurozone M3 and Bank of England CSVs. Applies percentage-change or differencing transforms and performs additive seasonal decomposition where appropriate.
* **Stationarity & Preprocessing Pipeline (`strict_stationarize`):** Executes Augmented Dickey-Fuller (ADF) unit-root tests on all GDELT sentiment and macro series, applying first-differencing and 5th/95th percentile winsorization to non-stationary variables to prevent spurious correlations.
* **Sparse Group Lasso Modeling (`run_cv_sparse_group_lasso`):** Groups GDELT narrative subcategories by overarching economic sectors (agriculture, commodities, energy, finance, industry) while isolating macroeconomic autoregressive lags. Tunes the $\ell_1$ sparsity ratio ($\alpha$) and group penalty ($\lambda$) via `TimeSeriesSplit` cross-validation to perform both group-level and within-group feature selection.
* **Post-Selection Inference & Diagnostics (`run_post_lasso_ols`):** Fits an OLS regression on SGL-selected features to obtain unbiased coefficients, p-values, and confidence intervals. Evaluates residual autocorrelation using Ljung-Box diagnostic tests ($p \ge 0.05$) under dynamic lag augmentation, benchmarking explanatory power ($R^2$ / Adjusted $R^2$) directly against macro-only ARX specifications.