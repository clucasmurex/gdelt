"""
build_gdelt_indicators_geo_15min_weekly.py
=========================================
Pipeline GDELT intégrant la dimension géographique.
Calcule les indicateurs par tranches de 15 minutes ET par zone géographique.
Traitement par batchs hebdomadaires.
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


# ── CONFIGURATION GÉOGRAPHIQUE (CODES FIPS 10-4) ──────────────────────────────
REGIONS = {
    "IS": ["IS"]
}
# ──────────────────────────────────────────────────────────────────────────────


def _fmt(n: int) -> str:
    return f"{n:>14,}"

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


# ══════════════════════════════════════════════════════════════════════════════
# 1. MATÉRIALISATION ET GÉOGRAPHIE
# ══════════════════════════════════════════════════════════════════════════════

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
    
    # Filtrage précis entre start_str et end_str (inclus)
    con.execute(fr"""
        CREATE TEMPORARY TABLE temp_mapped AS
        WITH raw AS (
            SELECT * FROM read_parquet('{glob_pattern}')
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND substr(CAST(DATE AS VARCHAR), 1, 8) BETWEEN '{start_str}' AND '{end_str}'
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
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
            -- Bucketing 15 minutes
            time_bucket(INTERVAL '15 Minutes', strptime(CAST(m.DATE AS VARCHAR), '%Y%m%d%H%M%S')) AS period,
            CAST(m.Tone AS DOUBLE) AS tone,
            SIGN(CAST(m.Tone AS DOUBLE)) AS tone_bin,
            ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) AS total_themes_count,
            list_transform(string_split(m.EnhancedThemes, ';'), x -> upper(trim(split_part(trim(x), ',', 1)))) AS themes_list,
            list_filter(
                list_distinct(list_transform(string_split(m.EnhancedLocations, ';'), x -> split_part(x, '#', 3))),
                c -> c != ''
            ) AS countries_list
        FROM temp_mapped m
        INNER JOIN retained_ids rid ON m.Src_ID = rid.id
        WHERE ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes};
        
        DROP TABLE temp_mapped;
    """)

def map_articles_to_regions(con):
    region_rows = [{"region_key": k, "country_code": c} for k, codes in REGIONS.items() for c in codes]
    con.register("regions_map", pd.DataFrame(region_rows))
    
    con.execute("""
        CREATE TABLE article_regions AS
        SELECT DISTINCT g.GKGRECORDID, g.period, rm.region_key
        FROM gkg_clean g, unnest(g.countries_list) AS c(code)
        INNER JOIN regions_map rm ON c.code = rm.country_code
    """)

def compute_total_news_regional(con):
    con.execute("""
        CREATE TABLE total_news_region_tbl AS
        SELECT period, region_key, COUNT(DISTINCT GKGRECORDID) AS total_news_region
        FROM article_regions
        GROUP BY 1, 2 ORDER BY 1, 2
    """)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALCUL DES INDICATEURS GÉOGRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_sector_indicators_geo(con, sector_key, sector_cfg):
    categories = sector_cfg["categories"]
    con.register("sector_themes_tbl", pd.DataFrame([
        {"cat_key": cat_key, "theme": theme.upper()}
        for cat_key, cat_cfg in categories.items() for theme in cat_cfg["themes"]
    ]))

    query = """
    WITH 
    article_theme_unnest AS (
        SELECT GKGRECORDID, period, tone, tone_bin, total_themes_count, unnest(themes_list) as theme
        FROM gkg_clean
    ),
    matched_themes AS (
        SELECT 
            u.GKGRECORDID, u.period, u.tone, u.tone_bin, u.total_themes_count, st.cat_key,
            COUNT(*) AS theme_hits
        FROM article_theme_unnest u
        INNER JOIN sector_themes_tbl st ON u.theme = st.theme
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    matched_with_regions AS (
        SELECT 
            mt.GKGRECORDID, mt.period, ar.region_key, mt.tone, mt.tone_bin, mt.total_themes_count, mt.cat_key, mt.theme_hits
        FROM matched_themes mt
        INNER JOIN article_regions ar ON mt.GKGRECORDID = ar.GKGRECORDID
    ),
    article_cat AS (
        SELECT 
            GKGRECORDID, period, region_key, cat_key, tone, tone_bin,
            (theme_hits::DOUBLE / total_themes_count) AS w
        FROM matched_with_regions
    ),
    daily_cat AS (
        SELECT 
            period, region_key, cat_key AS granularity,
            COUNT(*) AS N, SUM(w) AS sum_w, SUM(tone) AS sum_t_cont, SUM(tone_bin) AS sum_t_bin,
            SUM(tone * w) AS sum_t_cont_w, SUM(tone_bin * w) AS sum_t_bin_w
        FROM article_cat GROUP BY period, region_key, cat_key
    ),
    article_sector AS (
        SELECT 
            GKGRECORDID, period, region_key,
            ANY_VALUE(tone) AS tone, ANY_VALUE(tone_bin) AS tone_bin,
            SUM(theme_hits)::DOUBLE / ANY_VALUE(total_themes_count) AS w_sector
        FROM matched_with_regions
        GROUP BY GKGRECORDID, period, region_key
    ),
    daily_sector AS (
        SELECT 
            period, region_key, '__sector__' AS granularity,
            COUNT(*) AS N, SUM(w_sector) AS sum_w, SUM(tone) AS sum_t_cont, SUM(tone_bin) AS sum_t_bin,
            SUM(tone * w_sector) AS sum_t_cont_w, SUM(tone_bin * w_sector) AS sum_t_bin_w
        FROM article_sector GROUP BY period, region_key
    )

    SELECT * FROM daily_cat UNION ALL SELECT * FROM daily_sector
    """

    long_df = con.execute(query).df()
    if long_df.empty:
        return pd.DataFrame()

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


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir",  type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--config",       type=Path, default=Path("./sectors_config.json"))
    p.add_argument("--output_dir",   type=Path, default=Path("./data/indicators_geo_IS_15min"))
    p.add_argument("--sectors",      nargs="*", default=None)
    p.add_argument("--min_words",    type=int, default=150)
    p.add_argument("--max_words",    type=int, default=5500)
    p.add_argument("--retained_ids", type=Path, default=Path("gold_standard_whitelist_v2.txt"))
    p.add_argument("--min_themes",   type=int, default=2)
    p.add_argument("--threads",      type=int, default=16)
    p.add_argument("--memory_gb",    type=int, default=150)
    # Arguments de dates (YYYYMMDD ou YYYY-MM-DD)
    p.add_argument("--start_date",   type=str, default="20230907", help="Format: YYYYMMDD ou YYYY-MM-DD")
    p.add_argument("--end_date",     type=str, default="20231107", help="Format: YYYYMMDD ou YYYY-MM-DD")
    return p.parse_args()


def main():
    args = parse_args()
    t_total = time.time()

    with open(args.config, encoding="utf-8") as f:
        sectors = json.load(f)["sectors"]
    if args.sectors:
        sectors = {k: sectors[k] for k in args.sectors}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("./duckdb_tmp")

    # Génération des chunks hebdomadaires
    start_dt = pd.to_datetime(args.start_date)
    end_dt = pd.to_datetime(args.end_date)
    
    # Découpage en intervalles de 7 jours
    intervals = []
    curr = start_dt
    while curr <= end_dt:
        nxt = min(curr + pd.Timedelta(days=6), end_dt)
        intervals.append((curr.strftime("%Y%m%d"), nxt.strftime("%Y%m%d"), curr.year, nxt.year))
        curr = nxt + pd.Timedelta(days=1)

    print(f"\n[INFO] Découpage en {len(intervals)} semaines entre {start_dt.date()} et {end_dt.date()}")

    # Stockage des résultats par secteur
    collected_results = {sector_key: [] for sector_key in sectors.keys()}

    for i, (w_start, w_end, y_start, y_end) in enumerate(intervals, 1):
        print(f"\n{'═'*65}\n  SEMAINE {i}/{len(intervals)} : {w_start} -> {w_end}\n{'═'*65}")
        t_week = time.time()
        
        # Sélection des parquets nécessaires pour la période
        if y_start == y_end:
            glob_pattern = str(args.parquet_dir / f"gdelt_{y_start}*.parquet")
        else:
            glob_pattern = str(args.parquet_dir / "gdelt_*.parquet")
        
        if tmp_dir.exists(): 
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        
        con = make_connection(args.threads, args.memory_gb)
        
        try:
            build_materialized_clean_table(
                con, glob_pattern, args.source_map, args.retained_ids, 
                args.min_words, args.max_words, args.min_themes, 
                w_start, w_end
            )
            
            count_clean = con.execute("SELECT COUNT(*) FROM gkg_clean").fetchone()[0]
            if count_clean == 0:
                print(f"  ⚠ Aucun article nettoyé pour {w_start} -> {w_end}.")
                continue
            
            print(f"  --- Données chargées : {count_clean:,} articles ---")

            map_articles_to_regions(con)
            compute_total_news_regional(con)

            for sector_key, sector_cfg in sectors.items():
                result_df = compute_sector_indicators_geo(con, sector_key, sector_cfg)
                if not result_df.empty:
                    collected_results[sector_key].append(result_df)

            print(f"  > Semaine {w_start}_{w_end} terminée en {_elapsed(t_week)}")

        except Exception as e:
            print(f"  [ERREUR] Échec sur la semaine {w_start}->{w_end}: {e}")
        finally:
            con.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Sauvegarde Finale ─────────────────────────────────────────────────────
    print(f"\n{'═'*65}\n  [SAUVEGARDE FINALE DES FICHIERS 15-MIN]\n{'═'*65}")
    date_suffix = f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    
    for sector_key in sectors.keys():
        if collected_results[sector_key]:
            final_df = pd.concat(collected_results[sector_key], ignore_index=True)
            final_df = final_df.sort_values(by=["period", "region_key"])
            
            out_path = args.output_dir / f"{sector_key}_{date_suffix}_15min.parquet"
            final_df.to_parquet(out_path, index=False)
            print(f"  ✓ {sector_key} : {len(final_df):,} lignes écrites dans -> {out_path.name}")
        else:
            print(f"  ⚠ {sector_key} : Aucune donnée générée.")

    print(f"\nPipeline terminé en {_elapsed(t_total)}\n")

if __name__ == "__main__":
    main()