import json
import time
from pathlib import Path
import duckdb
import pandas as pd

def _fmt(n: int) -> str:
    return f"{n:>14,}"

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m{int(s%60):02d}s"

def make_connection(threads: int, memory_gb: int, tmp_dir: Path = Path("./duckdb_tmp")) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    
    tmp_dir.mkdir(exist_ok=True, parents=True)
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    
    return con

def compute_global_whitelist(
    con: duckdb.DuckDBPyConnection,
    parquet_dir: Path,
    source_map_path: Path,
    whitelist_out_path: Path,
    min_articles_per_source: int,   # <-- Seuil modulable
    min_active_years: int           # <-- Seuil modulable
) -> None:
    """Génère la liste des sources valides selon le nombre d'articles et d'années d'activité."""
    glob_pattern = str(parquet_dir / "gdelt_*.parquet")
    
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
        
    src_df = pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    })
    con.register("src_map", src_df)

    con.execute(fr"""
        CREATE TABLE global_valid_sources AS
        WITH raw_src AS (
            SELECT 
                COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID,
                substr(CAST(DATE AS VARCHAR), 1, 4) AS pub_year
            FROM read_parquet('{glob_pattern}') r
            LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
        )
        SELECT Src_ID
        FROM raw_src
        WHERE Src_ID IS NOT NULL
        GROUP BY Src_ID
        HAVING COUNT(*) >= {min_articles_per_source}
           AND COUNT(DISTINCT pub_year) >= {min_active_years};
    """)
    
    con.execute(f"COPY global_valid_sources TO '{whitelist_out_path}' (FORMAT PARQUET)")
    
    n = con.execute("SELECT COUNT(*) FROM global_valid_sources").fetchone()[0]
    print(f"  ✓ Whitelist globale générée : {n:,} sources respectent les critères (>= {min_active_years} ans, >= {min_articles_per_source} articles).")

def build_materialized_clean_table(
    con: duckdb.DuckDBPyConnection,
    glob_pattern: str,
    source_map_path: Path,
    whitelist_path: Path,
    min_words: int,                 # <-- Seuil modulable
    max_words: int,                 # <-- Seuil modulable
    min_themes: int,                # <-- Seuil modulable
) -> None:
    """Crée la table `gkg_clean` filtrée de manière dynamique."""
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
        
    src_df = pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    })
    con.register("src_map", src_df)

    con.execute(fr"""
        CREATE TEMPORARY TABLE temp_mapped AS
        WITH raw AS (
            SELECT * FROM read_parquet('{glob_pattern}')
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL
              AND EnhancedThemes != ''
              AND WordCount BETWEEN {min_words} AND {max_words}
        )
        SELECT 
            r.* EXCLUDE (SourceCommonName_ID),
            COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
        FROM raw r
        LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
        WHERE COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) IS NOT NULL;

        CREATE TABLE gkg_clean AS
        SELECT 
            m.GKGRECORDID,
            strptime(substr(CAST(m.DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period,
            CAST(m.Tone AS DOUBLE) AS tone,
            SIGN(CAST(m.Tone AS DOUBLE)) AS tone_bin,
            ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) AS total_themes_count,
            list_transform(
                string_split(m.EnhancedThemes, ';'),
                x -> upper(trim(split_part(trim(x), ',', 1)))
            ) AS themes_list
        FROM temp_mapped m
        INNER JOIN read_parquet('{whitelist_path}') v ON m.Src_ID = v.Src_ID
        WHERE ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes};
        
        DROP TABLE temp_mapped;
    """)
    n = con.execute("SELECT COUNT(*) FROM gkg_clean").fetchone()[0]
    print(f"  ✓ Table `gkg_clean` matérialisée ({n:,} articles | mots: [{min_words}-{max_words}] | thèmes >= {min_themes}).")

def compute_total_news(con: duckdb.DuckDBPyConnection) -> None:
    """Crée la table de référence journalière du total d'articles."""
    con.execute("""
        CREATE TABLE total_news_tbl AS
        SELECT period, COUNT(DISTINCT GKGRECORDID) AS total_news
        FROM gkg_clean
        GROUP BY 1 ORDER BY 1
    """)