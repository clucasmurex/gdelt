import gc
import glob
import json
import os
import time
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import HTML, display

# -----------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION & PLOTTING SETUP
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Paths configuration
DATA_DIR = Path("/data/gdelt/gdelt_parquet_db")
SOURCE_MAP_PATH = Path("/data/gdelt/gdelt_sources_mapping.json")
DOMAINS_PARQUET_PATTERN = "data/domains/domains_*.parquet"

# Pipeline thresholds
MIN_ACTIVE_YEARS = 2


# =============================================================================
# PART 1: DATABASE SETUP & PREPROCESSING PIPELINE
# =============================================================================
def setup_gdelt_pipeline() -> duckdb.DuckDBPyConnection:
  """Initializes DuckDB connection, sets hardware limits, loads reference maps,

  and creates preprocessing views.
  """
  print("========================================================")
  print("🛠️  INITIALIZING DUCKDB & PREPROCESSING PIPELINE")
  print("========================================================")

  parquet_files = sorted(glob.glob(str(DATA_DIR / "gdelt_*.parquet")))
  total_size_gb = sum(os.path.getsize(f) for f in parquet_files) / (1024**3)
  print(f"✔ Found {len(parquet_files)} parquet files ({total_size_gb:.2f} GB).")

  con = duckdb.connect()
  # Set memory limit to prevent OS freezing; DuckDB will manage RAM strictly
  con.execute("PRAGMA memory_limit='32GB'")
  con.execute("PRAGMA threads=4")

  glob_pattern = str(DATA_DIR / "gdelt_*.parquet")
  con.execute(
      f"CREATE OR REPLACE VIEW gkg AS SELECT * FROM read_parquet('{glob_pattern}')"
  )

  with open(SOURCE_MAP_PATH, "r", encoding="utf-8") as f:
    source_map = json.load(f)

  src_df = pd.DataFrame({
      "SourceCommonName_ID": [
          int(k) for k in source_map["id_to_source"].keys()
      ],
      "SourceCommonName": list(source_map["id_to_source"].values()),
  })
  con.register("src_map", src_df)
  print(
      "✔ Parquet base view and JSON dictionary (src_map) loaded into DuckDB."
  )

  # 1. gkg_inter: Noise filtering and ID repair
  print(
      "⏳ Creating 'gkg_inter' view (Filtering and repairing sources)..."
  )
  con.execute(r"""
        CREATE OR REPLACE VIEW gkg_inter AS
        WITH raw_filtered AS (
            SELECT * FROM gkg
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{14}$')
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
        )
        SELECT 
            r.* EXCLUDE (SourceCommonName_ID),
            CASE 
                WHEN COALESCE(r.SourceCommonName_ID, 0) = 0 THEN m.SourceCommonName_ID
                ELSE r.SourceCommonName_ID
            END AS SourceCommonName_ID
        FROM raw_filtered r
        LEFT JOIN src_map m 
          ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '.') = m.SourceCommonName
        WHERE COALESCE(r.SourceCommonName_ID, 0) != 0 OR m.SourceCommonName_ID IS NOT NULL
    """)

  # 2. gkg_year: Filter by minimum active lifespan
  print(
      f"⏳ Creating 'gkg_year' view (Filtering: >= {MIN_ACTIVE_YEARS} active"
      " years)..."
  )
  con.execute(f"""
        CREATE OR REPLACE VIEW gkg_year AS
        WITH ValidSources AS (
            SELECT SourceCommonName_ID FROM gkg_inter
            GROUP BY SourceCommonName_ID
            HAVING COUNT(DISTINCT substr(CAST(DATE AS VARCHAR), 1, 4)) >= {MIN_ACTIVE_YEARS}
        )
        SELECT s.* FROM gkg_inter s
        INNER JOIN ValidSources v ON s.SourceCommonName_ID = v.SourceCommonName_ID
    """)

  # 3. gkg_wiki: Sources present in Wikidata
  print("⏳ Creating 'gkg_wiki' view (In gkg_year AND in Wikidata)...")
  con.execute(f"""
        CREATE OR REPLACE VIEW gkg_wiki AS
        SELECT s.*, w.medialabel, w.typelabel, w.countrylabel, w.inception
        FROM gkg_year s
        INNER JOIN (
            SELECT id AS Src_ID, medialabel, typelabel, countrylabel, inception
            FROM read_parquet('{DOMAINS_PARQUET_PATTERN}')
            QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY inception ASC, countrylabel ASC) = 1
        ) w ON s.SourceCommonName_ID = w.Src_ID;
    """)

  # 4. gkg_not_in_wiki: Sources NOT present in Wikidata
  print(
      "⏳ Creating 'gkg_not_in_wiki' view (In gkg_year but NOT in Wikidata)..."
  )
  con.execute(f"""
        CREATE OR REPLACE VIEW gkg_not_in_wiki AS
        SELECT s.*, NULL AS medialabel, NULL AS typelabel, NULL AS countrylabel, NULL AS inception
        FROM gkg_year s
        LEFT JOIN (
            SELECT DISTINCT id AS Src_ID FROM read_parquet('{DOMAINS_PARQUET_PATTERN}')
        ) w ON s.SourceCommonName_ID = w.Src_ID
        WHERE w.Src_ID IS NULL;
    """)
  print("✔ All preprocessing views created successfully!")
  return con


# =============================================================================
# PART 2: MODULAR, DECOMPOSED FEATURE EXTRACTION ENGINE
# =============================================================================
def cleanup_temp_files(*file_paths):
  """Safely removes temporary Parquet files from disk."""
  for fp in file_paths:
    if os.path.exists(fp):
      try:
        os.remove(fp)
      except Exception as e:
        print(f"⚠️ Could not remove temp file {fp}: {e}")


def extract_gkg_features_decomposed(
    con: duckdb.DuckDBPyConnection,
    source_view: str,
    output_parquet: str,
    is_wiki_flag: int,
):
  """Decomposes feature extraction into isolated Parquet checkpoints to prevent

  RAM explosion and temporary `.tmp` disk spills.
  """
  print(f"\n========================================================")
  print(f"🚀 Starting DECOMPOSED extraction for: '{source_view}'")
  print(f"========================================================")
  start_time = time.time()

  # Define temporary checkpoint filenames
  f_subset = f"temp_0_subset_{is_wiki_flag}.parquet"
  f_base = f"temp_1_base_{is_wiki_flag}.parquet"
  f_daily = f"temp_2_daily_{is_wiki_flag}.parquet"
  f_temporal = f"temp_3_temporal_{is_wiki_flag}.parquet"
  f_themes_raw = f"temp_4_themes_raw_{is_wiki_flag}.parquet"
  f_entropy = f"temp_5_entropy_{is_wiki_flag}.parquet"

  try:
    # ---------------------------------------------------------------------
    # STEP 0: MATERIALIZE ESSENTIAL COLUMNS ONLY
    # This prevents DuckDB from scanning the massive raw GKG files 4+ times!
    # ---------------------------------------------------------------------
    print(
        "⏳ [Step 0/5] Extracting essential columns to compressed checkpoint..."
    )
    t0 = time.time()
    con.execute(f"""
            COPY (
                SELECT 
                    SourceCommonName_ID,
                    WordCount,
                    EnhancedThemes,
                    strptime(substr(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period
                FROM {source_view}
            ) TO '{f_subset}' (FORMAT PARQUET, CODEC 'ZSTD');
        """)
    print(
        f"   ✔ Subset materialized in {time.time()-t0:.2f}s"
        f" ({os.path.getsize(f_subset)/(1024**2):.2f} MB)"
    )

    # ---------------------------------------------------------------------
    # STEP 1: BASE STATS & WORDCOUNT
    # ---------------------------------------------------------------------
    print("⏳ [Step 1/5] Computing base aggregations and wordcount...")
    t1 = time.time()
    con.execute(f"""
            COPY (
                SELECT 
                    SourceCommonName_ID,
                    COUNT(*) AS total_articles,
                    AVG(WordCount) AS mean_wordcount,
                    CASE WHEN AVG(WordCount) > 0 THEN STDDEV(WordCount) / AVG(WordCount) ELSE 0 END AS cv_wordcount,
                    AVG(
                        CASE 
                            WHEN EnhancedThemes IS NULL OR EnhancedThemes = '' THEN 0 
                            ELSE len(list_distinct(list_transform(string_split(EnhancedThemes, ';'), x -> split_part(x, ',', 1))))
                        END
                    ) AS mean_themes_per_art
                FROM read_parquet('{f_subset}')
                GROUP BY SourceCommonName_ID
            ) TO '{f_base}' (FORMAT PARQUET);
        """)
    print(f"   ✔ Base features saved in {time.time()-t1:.2f}s")

    # ---------------------------------------------------------------------
    # STEP 2: TEMPORAL DISPERSION (Daily -> Monthly/Yearly)
    # ---------------------------------------------------------------------
    print("⏳ [Step 2/5] Computing temporal dynamics and dispersion...")
    t2 = time.time()

    # Intermediate daily counts
    con.execute(f"""
            COPY (
                SELECT SourceCommonName_ID, period, COUNT(*) AS n_articles
                FROM read_parquet('{f_subset}')
                GROUP BY 1, 2
            ) TO '{f_daily}' (FORMAT PARQUET);
        """)

    # Compute CVs and inactive days rate from daily checkpoint
    con.execute(f"""
            COPY (
                WITH daily_stats AS (
                    SELECT SourceCommonName_ID, COUNT(*) AS active_days, MIN(period) AS first_seen, MAX(period) AS last_seen
                    FROM read_parquet('{f_daily}') GROUP BY 1
                ),
                monthly_calc AS (
                    SELECT SourceCommonName_ID,
                        date_diff('month', MIN(month_date), MAX(month_date)) + 1 AS lifespan_months,
                        SUM(n_articles)::DOUBLE / (date_diff('month', MIN(month_date), MAX(month_date)) + 1) AS avg_monthly,
                        (SUM(n_articles * n_articles)::DOUBLE / (date_diff('month', MIN(month_date), MAX(month_date)) + 1)) 
                        - POW(SUM(n_articles)::DOUBLE / (date_diff('month', MIN(month_date), MAX(month_date)) + 1), 2) AS var_monthly
                    FROM (SELECT SourceCommonName_ID, date_trunc('month', period) AS month_date, SUM(n_articles) AS n_articles FROM read_parquet('{f_daily}') GROUP BY 1, 2)
                    GROUP BY 1
                ),
                yearly_calc AS (
                    SELECT SourceCommonName_ID,
                        COUNT(DISTINCT year_date) AS years_active,
                        date_diff('year', MIN(year_date), MAX(year_date)) + 1 AS lifespan_years,
                        SUM(n_articles)::DOUBLE / (date_diff('year', MIN(year_date), MAX(year_date)) + 1) AS avg_yearly,
                        (SUM(n_articles * n_articles)::DOUBLE / (date_diff('year', MIN(year_date), MAX(year_date)) + 1)) 
                        - POW(SUM(n_articles)::DOUBLE / (date_diff('year', MIN(year_date), MAX(year_date)) + 1), 2) AS var_yearly
                    FROM (SELECT SourceCommonName_ID, date_trunc('year', period) AS year_date, SUM(n_articles) AS n_articles FROM read_parquet('{f_daily}') GROUP BY 1, 2)
                    GROUP BY 1
                )
                SELECT 
                    d.SourceCommonName_ID,
                    yc.years_active,
                    1.0 - (d.active_days::DOUBLE / NULLIF(date_diff('day', d.first_seen, d.last_seen) + 1, 0)) AS inactive_days_rate,
                    CASE WHEN mc.avg_monthly > 0 THEN SQRT(GREATEST(mc.var_monthly, 0)) / mc.avg_monthly ELSE 0 END AS cv_monthly,
                    CASE WHEN yc.avg_yearly > 0 THEN SQRT(GREATEST(yc.var_yearly, 0)) / yc.avg_yearly ELSE 0 END AS cv_yearly
                FROM daily_stats d
                JOIN monthly_calc mc ON d.SourceCommonName_ID = mc.SourceCommonName_ID
                JOIN yearly_calc yc ON d.SourceCommonName_ID = yc.SourceCommonName_ID
            ) TO '{f_temporal}' (FORMAT PARQUET);
        """)
    cleanup_temp_files(f_daily)  # Delete daily counts immediately
    print(f"   ✔ Temporal features saved in {time.time()-t2:.2f}s")

    # ---------------------------------------------------------------------
    # STEP 3: THEME UNNESTING & SHANNON ENTROPY (Heaviest Memory Step)
    # ---------------------------------------------------------------------
    print("⏳ [Step 3/5] Unnesting themes and calculating Shannon entropies...")
    t3 = time.time()

    # Dump raw aggregated theme frequencies to disk to break RAM limits
    con.execute(f"""
            COPY (
                SELECT 
                    SourceCommonName_ID,
                    split_part(raw_theme, ',', 1) AS theme,
                    COUNT(*) AS freq,
                    CASE WHEN split_part(raw_theme, ',', 1) LIKE '%ECON%' THEN 1 ELSE 0 END AS is_eco
                FROM (
                    SELECT SourceCommonName_ID, unnest(string_split(EnhancedThemes, ';')) AS raw_theme
                    FROM read_parquet('{f_subset}')
                    WHERE EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
                )
                WHERE raw_theme != ''
                GROUP BY 1, 2
            ) TO '{f_themes_raw}' (FORMAT PARQUET, CODEC 'ZSTD');
        """)
    # We no longer need the base subset parquet! Free up disk space now.
    cleanup_temp_files(f_subset)

    # Compute entropy strictly from the lightweight theme frequencies parquet
    con.execute(f"""
            COPY (
                WITH source_totals AS (
                    SELECT SourceCommonName_ID, SUM(freq) AS total_freq, SUM(CASE WHEN is_eco = 1 THEN freq ELSE 0 END) AS eco_total_freq
                    FROM read_parquet('{f_themes_raw}') GROUP BY 1
                ),
                global_max AS (
                    SELECT COUNT(DISTINCT theme) AS T_global, COUNT(DISTINCT CASE WHEN is_eco = 1 THEN theme END) AS T_global_eco
                    FROM read_parquet('{f_themes_raw}')
                )
                SELECT 
                    tc.SourceCommonName_ID,
                    st.eco_total_freq::DOUBLE / NULLIF(st.total_freq, 0) AS ratio_eco,
                    SUM(- (tc.freq * 1.0 / st.total_freq) * ln(tc.freq * 1.0 / st.total_freq)) / ln(NULLIF((SELECT T_global FROM global_max), 1)) AS h_norm_themes,
                    COALESCE(
                        SUM(CASE WHEN tc.is_eco = 1 AND st.eco_total_freq > 0 THEN - (tc.freq * 1.0 / st.eco_total_freq) * ln(tc.freq * 1.0 / st.eco_total_freq) ELSE 0 END) 
                        / ln(NULLIF((SELECT T_global_eco FROM global_max), 1)), 0.0
                    ) AS h_norm_eco_themes
                FROM read_parquet('{f_themes_raw}') tc
                JOIN source_totals st ON tc.SourceCommonName_ID = st.SourceCommonName_ID
                GROUP BY tc.SourceCommonName_ID, st.total_freq, st.eco_total_freq
            ) TO '{f_entropy}' (FORMAT PARQUET);
        """)
    cleanup_temp_files(f_themes_raw)
    print(f"   ✔ Thematic features saved in {time.time()-t3:.2f}s")

    # ---------------------------------------------------------------------
    # STEP 4: FINAL ASSEMBLY & EXPORT
    # ---------------------------------------------------------------------
    print("⏳ [Step 4/5] Merging checkpoints into final Parquet deliverable...")
    t4 = time.time()
    con.execute(f"""
            COPY (
                SELECT 
                    b.SourceCommonName_ID,
                    {is_wiki_flag} AS is_wiki,
                    t.years_active,
                    LN(1.0 + (b.total_articles::DOUBLE / NULLIF(t.years_active, 0))) AS log_articles_per_year,
                    t.cv_monthly,
                    t.cv_yearly,
                    t.inactive_days_rate,
                    b.mean_wordcount,
                    b.cv_wordcount,
                    b.mean_themes_per_art,
                    COALESCE(e.ratio_eco, 0.0) AS ratio_eco,
                    COALESCE(e.h_norm_themes, 0.0) AS h_norm_themes,
                    COALESCE(e.h_norm_eco_themes, 0.0) AS h_norm_eco_themes
                FROM read_parquet('{f_base}') b
                JOIN read_parquet('{f_temporal}') t ON b.SourceCommonName_ID = t.SourceCommonName_ID
                LEFT JOIN read_parquet('{f_entropy}') e ON b.SourceCommonName_ID = e.SourceCommonName_ID
                ORDER BY b.total_articles DESC
            ) TO '{output_parquet}' (FORMAT PARQUET);
        """)
    print(f"   ✔ Final export complete in {time.time()-t4:.2f}s")

  finally:
    # ---------------------------------------------------------------------
    # STEP 5: GUARANTEED CLEANUP
    # ---------------------------------------------------------------------
    print("🧹 Cleaning up all temporary checkpoint files...")
    cleanup_temp_files(f_subset, f_base, f_daily, f_temporal, f_themes_raw, f_entropy)
    gc.collect()

  elapsed_time = time.time() - start_time
  print(
      f"✔ SUCCESS: Extracted features safely saved to '{output_parquet}' in"
      f" {elapsed_time:.2f}s!"
  )


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
  db_conn = setup_gdelt_pipeline()

  try:
    # Extract features for sources NOT in Wikipedia
    extract_gkg_features_decomposed(
        con=db_conn,
        source_view="gkg_not_in_wiki",
        output_parquet="features_sources_not_in_wiki.parquet",
        is_wiki_flag=0,
    )

    # Extract features for Wikipedia sources
    extract_gkg_features_decomposed(
        con=db_conn,
        source_view="gkg_wiki",
        output_parquet="features_sources_wiki.parquet",
        is_wiki_flag=1,
    )

    print("\n🎉 Entire decomposed pipeline completed successfully!")

  finally:
    db_conn.close()
    print("🧹 Database connection safely closed.")