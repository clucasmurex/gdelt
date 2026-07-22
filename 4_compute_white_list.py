"""
4_compute_white_list.py
=======================
Pipeline GDELT dédié à la génération de la whitelist des sources.
Le script ne calcule que la liste des sources valides et écrit le fichier parquet associé.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import duckdb
import pandas as pd


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m{int(s%60):02d}s"

def make_connection(threads: int, memory_gb: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    
    tmp_dir = Path("./duckdb_tmp")
    tmp_dir.mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def compute_global_whitelist(con, parquet_dir, source_map_path, min_years):
    glob_pattern = str(parquet_dir / "gdelt_*.parquet")
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    }))
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
        SELECT Src_ID FROM raw_src WHERE Src_ID IS NOT NULL
        GROUP BY Src_ID HAVING COUNT(DISTINCT pub_year) >= {min_years};
    """)
    con.execute("COPY global_valid_sources TO 'valid_sources_whitelist.parquet' (FORMAT PARQUET)")
    print(f"  ✓ Whitelist globale générée : {con.execute('SELECT COUNT(*) FROM global_valid_sources').fetchone()[0]:,} sources.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map", type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    # p.add_argument("--min_articles_per_source", type=int, default=30)
    p.add_argument("--min_active_years", type=int, default=2)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--memory_gb", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    t_total = time.time()

    whitelist_file = Path("valid_sources_whitelist.parquet")
    tmp_dir = Path("./duckdb_tmp")

    print("\n[ÉTAPE 0] Calcul de la Whitelist globale des sources...")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)

    con = make_connection(args.threads, args.memory_gb)
    try:
        compute_global_whitelist(
            con,
            args.parquet_dir,
            args.source_map,
            args.min_active_years,
        )
    finally:
        con.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nWhitelist enregistrée dans {whitelist_file.resolve()}\n")
    print(f"Pipeline terminé en {_elapsed(t_total)}")

if __name__ == "__main__":
    main()