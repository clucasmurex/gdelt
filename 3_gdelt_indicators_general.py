"""
build_gdelt_indicators_unified.py
=================================
Pipeline GDELT Universel (Global & Geo / 15min, Journalier, Mensuel).
Optimisé pour minimiser l'empreinte mémoire et les fichiers temporaires.
"""

import argparse
import json
import time
from pathlib import Path
import shutil 
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION PRINCIPALE (À MODIFIER ICI)
# ══════════════════════════════════════════════════════════════════════════════

# Mode de calcul : "global" (monde entier) ou "geo" (restreint par pays)
MODE = "geo" 

# Liste des pays (Format FIPS 10-4). Ignoré si MODE = "global".
REGIONS = {
    "US": ["US"],
    "France": ["FR"],
    "UK" :["UK"]
}

# Fréquence désirée : "15min", "daily", ou "monthly"
FREQUENCY = "15min" 

# Période de calcul (Format: "YYYYMMDD"). Mettre à None pour tout l'échantillon.
START_DATE = "20230901"
END_DATE = "20230930"

# Dossier de sortie
OUTPUT_DIR = f"./data/indicators_{MODE}_{FREQUENCY}"

# ══════════════════════════════════════════════════════════════════════════════


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


# ── 1. MATÉRIALISATION ET GÉOGRAPHIE ──────────────────────────────────────────

def build_materialized_clean_table(con, glob_pattern, source_map_path, retained_ids_path, min_words, max_words, min_themes, start_str, end_str):
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    }))
    
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE retained_ids AS 
        SELECT column0::BIGINT AS id 
        FROM read_csv('{retained_ids_path}', header=False)
    """)

    # Définition de l'expression temporelle selon la fréquence configurée
    if FREQUENCY == "15min":
        period_expr = "time_bucket(INTERVAL '15 Minutes', strptime(CAST(DATE AS VARCHAR), '%Y%m%d%H%M%S'))"
    elif FREQUENCY == "daily":
        period_expr = "strptime(substr(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE"
    elif FREQUENCY == "monthly":
        period_expr = "date_trunc('month', strptime(substr(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE)::DATE"
    else:
        raise ValueError("FREQUENCY doit être '15min', 'daily', ou 'monthly'")

    # Filtre de date (si défini)
    date_filter = f"AND substr(CAST(DATE AS VARCHAR), 1, 8) BETWEEN '{start_str}' AND '{end_str}'" if start_str and end_str else ""

    # 🔥 OPTIMISATION : Suppression de temp_mapped. Tout est fait en une passe pour éviter les écritures disques inutiles.
    con.execute(fr"""
        CREATE TABLE gkg_clean AS
        WITH raw_filtered AS (
            SELECT 
                GKGRECORDID, DATE, Tone, EnhancedThemes, EnhancedLocations,
                COALESCE(NULLIF(SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
            FROM read_parquet('{glob_pattern}') r
            LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              {date_filter}
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
              AND WordCount BETWEEN {min_words} AND {max_words}
        )
        SELECT 
            r.GKGRECORDID,
            {period_expr} AS period,
            CAST(r.Tone AS DOUBLE) AS tone,
            SIGN(CAST(r.Tone AS DOUBLE)) AS tone_bin,
            ARRAY_LENGTH(string_split(r.EnhancedThemes, ';')) AS total_themes_count,
            list_transform(string_split(r.EnhancedThemes, ';'), x -> upper(trim(split_part(trim(x), ',', 1)))) AS themes_list,
            list_filter(list_distinct(list_transform(string_split(r.EnhancedLocations, ';'), x -> split_part(x, '#', 3))), c -> c != '') AS countries_list
        FROM raw_filtered r
        INNER JOIN retained_ids rid ON r.Src_ID = rid.id
        WHERE r.Src_ID IS NOT NULL
          AND ARRAY_LENGTH(string_split(r.EnhancedThemes, ';')) >= {min_themes};
    """)

def map_articles_to_regions(con):
    if MODE == "global":
        # Mode global : on affecte tous les articles à une fausse région "GLOBAL"
        con.execute("""
            CREATE TABLE article_regions AS
            SELECT GKGRECORDID, period, 'GLOBAL' AS region_key
            FROM gkg_clean
        """)
    else:
        # Mode geo : croisement avec la configuration REGIONS
        region_rows = [{"region_key": k, "country_code": c} for k, codes in REGIONS.items() for c in codes]
        con.register("regions_map", pd.DataFrame(region_rows))
        
        con.execute("""
            CREATE TABLE article_regions AS
            SELECT DISTINCT g.GKGRECORDID, g.period, rm.region_key
            FROM gkg_clean g, unnest(g.countries_list) AS c(code)
            INNER JOIN regions_map rm ON c.code = rm.country_code
        """)

def compute_total_news(con):
    con.execute("""
        CREATE TABLE total_news_region_tbl AS
        SELECT period, region_key, COUNT(DISTINCT GKGRECORDID) AS total_news_region
        FROM article_regions
        GROUP BY 1, 2 ORDER BY 1, 2
    """)

# ── 2. CALCUL DES INDICATEURS ─────────────────────────────────────────────────

def compute_sector_indicators(con, sector_key, sector_cfg):
    categories = sector_cfg["categories"]
    con.register("sector_themes_tbl", pd.DataFrame([
        {"cat_key": cat_key, "theme": theme.upper()}
        for cat_key, cat_cfg in categories.items() for theme in cat_cfg["themes"]
    ]))

    # 🔥 Requête unifiée utilisant l'approche de dépilage optimisée
    query = """
    WITH 
    article_theme_unnest AS (
        SELECT GKGRECORDID, period, tone, tone_bin, total_themes_count, unnest(themes_list) as theme FROM gkg_clean
    ),
    matched_themes AS (
        SELECT u.GKGRECORDID, u.period, u.tone, u.tone_bin, u.total_themes_count, st.cat_key, COUNT(*) AS theme_hits
        FROM article_theme_unnest u INNER JOIN sector_themes_tbl st ON u.theme = st.theme
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    matched_with_regions AS (
        SELECT mt.GKGRECORDID, mt.period, ar.region_key, mt.tone, mt.tone_bin, mt.total_themes_count, mt.cat_key, mt.theme_hits
        FROM matched_themes mt INNER JOIN article_regions ar ON mt.GKGRECORDID = ar.GKGRECORDID
    ),
    article_cat AS (
        SELECT GKGRECORDID, period, region_key, cat_key, tone, tone_bin, (theme_hits::DOUBLE / total_themes_count) AS w
        FROM matched_with_regions
    ),
    aggr_cat AS (
        SELECT period, region_key, cat_key AS granularity,
               COUNT(*) AS N, SUM(w) AS sum_w, SUM(tone) AS sum_t_cont, SUM(tone_bin) AS sum_t_bin,
               SUM(tone * w) AS sum_t_cont_w, SUM(tone_bin * w) AS sum_t_bin_w
        FROM article_cat GROUP BY period, region_key, cat_key
    ),
    article_sector AS (
        SELECT GKGRECORDID, period, region_key, ANY_VALUE(tone) AS tone, ANY_VALUE(tone_bin) AS tone_bin,
               SUM(theme_hits)::DOUBLE / ANY_VALUE(total_themes_count) AS w_sector
        FROM matched_with_regions GROUP BY GKGRECORDID, period, region_key
    ),
    aggr_sector AS (
        SELECT period, region_key, '__sector__' AS granularity,
               COUNT(*) AS N, SUM(w_sector) AS sum_w, SUM(tone) AS sum_t_cont, SUM(tone_bin) AS sum_t_bin,
               SUM(tone * w_sector) AS sum_t_cont_w, SUM(tone_bin * w_sector) AS sum_t_bin_w
        FROM article_sector GROUP BY period, region_key
    )
    SELECT * FROM aggr_cat UNION ALL SELECT * FROM aggr_sector
    """

    long_df = con.execute(query).df()
    if long_df.empty: return pd.DataFrame()

    total_news = con.execute("SELECT period, region_key, total_news_region FROM total_news_region_tbl").df()
    total_news["period"] = pd.to_datetime(total_news["period"])
    long_df["period"] = pd.to_datetime(long_df["period"])

    long_df = long_df.merge(total_news, on=["period", "region_key"], how="left")

    long_df["att"]              = long_df["N"] / long_df["total_news_region"]
    long_df["att_weight"]       = long_df["sum_w"] / long_df["total_news_region"]
    long_df["sent_cont"]        = np.where(long_df["N"] > 0, long_df["sum_t_cont"] / long_df["N"], np.nan)
    long_df["sent_bin"]         = np.where(long_df["N"] > 0, long_df["sum_t_bin"] / long_df["N"], np.nan)
    long_df["sent_cont_weight"] = np.where(long_df["sum_w"] > 0, long_df["sum_t_cont_w"] / long_df["sum_w"], np.nan)
    long_df["sent_bin_weight"]  = np.where(long_df["sum_w"] > 0, long_df["sum_t_bin_w"] / long_df["sum_w"], np.nan)
    long_df["axs_cont"]         = long_df["att"] * long_df["sent_cont"]
    long_df["axs_cont_weight"]  = long_df["att_weight"] * long_df["sent_cont_weight"]
    long_df["axs_bin"]          = long_df["att"] * long_df["sent_bin"]
    long_df["axs_bin_weight"]   = long_df["att_weight"] * long_df["sent_bin_weight"]

    metrics = ["att", "att_weight", "sent_cont", "sent_bin", "sent_cont_weight", "sent_bin_weight",
               "axs_cont", "axs_cont_weight", "axs_bin", "axs_bin_weight"]

    pivot_df = long_df.pivot(index=["period", "region_key"], columns="granularity", values=metrics)
    pivot_df.columns = [f"{m}_{sector_key}" if g == "__sector__" else f"{m}_{sector_key}_{g}" for m, g in pivot_df.columns]
    
    return pivot_df.reset_index()


# ── 3. MAIN (GESTION DES BATCHS) ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir",  type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--config",       type=Path, default=Path("./sectors_config.json"))
    p.add_argument("--retained_ids", type=Path, default=Path("gold_standard_whitelist_v2.txt"))
    p.add_argument("--min_words",    type=int, default=150)
    p.add_argument("--max_words",    type=int, default=5500)
    p.add_argument("--min_themes",   type=int, default=2)
    p.add_argument("--threads",      type=int, default=16)
    p.add_argument("--memory_gb",    type=int, default=150)
    return p.parse_args()

def main():
    args = parse_args()
    t_total = time.time()

    with open(args.config, encoding="utf-8") as f:
        sectors = json.load(f)["sectors"]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("./duckdb_tmp")

    # Logique de découpage : Hebdomadaire si 15min (pour préserver la RAM), Mensuelle sinon
    freq_days = 6 if FREQUENCY == "15min" else 30
    
    start_dt = pd.to_datetime(START_DATE) if START_DATE else pd.to_datetime("20150101")
    end_dt = pd.to_datetime(END_DATE) if END_DATE else pd.to_datetime(datetime.now().strftime("%Y%m%d"))
    
    intervals = []
    curr = start_dt
    while curr <= end_dt:
        nxt = min(curr + pd.Timedelta(days=freq_days), end_dt)
        intervals.append((curr.strftime("%Y%m%d"), nxt.strftime("%Y%m%d"), curr.year, nxt.year))
        curr = nxt + pd.Timedelta(days=1)

    print(f"\n[INFO] Mode : {MODE.upper()} | Fréquence : {FREQUENCY} | Batchs : {len(intervals)}")

    collected_results = {sector_key: [] for sector_key in sectors.keys()}

    for i, (w_start, w_end, y_start, y_end) in enumerate(intervals, 1):
        print(f"\n{'═'*50}\n  BATCH {i}/{len(intervals)} : {w_start} -> {w_end}\n{'═'*50}")
        
        glob_pattern = str(args.parquet_dir / f"gdelt_{y_start}*.parquet") if y_start == y_end else str(args.parquet_dir / "gdelt_*.parquet")
        
        if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        
        con = make_connection(args.threads, args.memory_gb)
        
        try:
            t0 = time.time()
            build_materialized_clean_table(con, glob_pattern, args.source_map, args.retained_ids, args.min_words, args.max_words, args.min_themes, w_start, w_end)
            
            count_clean = con.execute("SELECT COUNT(*) FROM gkg_clean").fetchone()[0]
            if count_clean == 0:
                print("  ⚠ Aucun article correspondant, passage au batch suivant.")
                continue
                
            map_articles_to_regions(con)
            compute_total_news(con)

            for sector_key, sector_cfg in sectors.items():
                result_df = compute_sector_indicators(con, sector_key, sector_cfg)
                if not result_df.empty:
                    collected_results[sector_key].append(result_df)

            print(f"  > Batch {w_start}_{w_end} calculé en {_elapsed(t0)}")

        except Exception as e:
            print(f"  [ERREUR] Échec sur le batch {w_start}->{w_end}: {e}")
        finally:
            con.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── SAUVEGARDE FINALE ──
    print(f"\n{'═'*50}\n  [SAUVEGARDE FINALE]\n{'═'*50}")
    date_suffix = f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    
    for sector_key in sectors.keys():
        if collected_results[sector_key]:
            final_df = pd.concat(collected_results[sector_key], ignore_index=True).sort_values(by=["period", "region_key"])
            out_path = out_dir / f"{sector_key}_{date_suffix}_{MODE}_{FREQUENCY}.parquet"
            final_df.to_parquet(out_path, index=False)
            print(f"  ✓ {sector_key} : {len(final_df):,} lignes -> {out_path.name}")

    print(f"\nPipeline terminé en {_elapsed(t_total)}\n")

if __name__ == "__main__":
    main()