"""
build_gdelt_indicators_geo.py
=============================
Pipeline GDELT intégrant la dimension géographique.
Calcule les 10 indicateurs par jour ET par zone géographique spécifiée.
"""

import argparse
import json
import time
from pathlib import Path
import shutil 
import duckdb
import pandas as pd
import numpy as np


# ── CONFIGURATION GÉOGRAPHIQUE (CODES FIPS 10-4) ──────────────────────────────
# GDELT utilise le format FIPS (ex: Chine = CH, et non CN)
REGIONS = {
    "US": ["US"],
    "China": ["CH"],
    "India": ["IN"],
    "Russia": ["RS"],
    "France": ["FR"],
    "Japan": ["JA"],
    "Lebanon": ["LE"],
    "North America": ["US", "CA", "MX"],
    "South America": ["AR", "BL", "BR", "CI", "CO", "EC", "GY", "PM", "PA", "PE", "UY", "VE", "NS"],
    "European Union": [
        "AU", "BE", "BU", "HR", "CY", "EZ", "DA", "EN", "FI", "FR", "GM", "GR", 
        "HU", "EI", "IT", "LG", "LH", "MT", "NL", "PL", "PO", "RO", "LO", "SI", "SP", "SW", "LU"
    ],
    "Middle East": ["SA", "IR", "IZ", "SY", "JO", "IS", "LE", "KU", "QA", "AE", "YM", "MU", "BA", "TU"],
    "South East Asia": ["ID", "MY", "PH", "SN", "TH", "VM", "CB", "LA", "BM", "BX", "TT"],
    "Africa": [
        "AG", "AO", "BN", "BC", "UV", "BY", "CM", "CV", "CT", "CD", "DJ", "EG", "EK", "ER", "ET", 
        "GB", "GA", "GH", "GV", "PU", "KE", "LT", "LI", "LY", "MA", "MI", "ML", "MR", "MP", "WA", 
        "NG", "NI", "RW", "SG", "SE", "SL", "SO", "SF", "OD", "WZ", "TZ", "TO", "TS", "UG", "ZA", "ZI", "CN", "CF", "IV", "MO", "MZ", "CG", "TP", "SU"
    ]
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

def compute_global_whitelist(con, parquet_dir, source_map_path, min_articles, min_years):
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
        GROUP BY Src_ID HAVING COUNT(*) >= {min_articles} AND COUNT(DISTINCT pub_year) >= {min_years};
    """)
    con.execute("COPY global_valid_sources TO 'valid_sources_whitelist.parquet' (FORMAT PARQUET)")
    print(f"  ✓ Whitelist globale générée : {con.execute('SELECT COUNT(*) FROM global_valid_sources').fetchone()[0]:,} sources.")

def build_materialized_clean_table(con, glob_pattern, source_map_path, min_words, max_words, min_themes):
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    }))
    
    # NOUVEAUTÉ : Extraction propre des codes pays uniques dans "countries_list"
    con.execute(fr"""
        CREATE TEMPORARY TABLE temp_mapped AS
        WITH raw AS (
            SELECT * FROM read_parquet('{glob_pattern}')
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
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
            strptime(substr(CAST(m.DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period,
            CAST(m.Tone AS DOUBLE) AS tone,
            SIGN(CAST(m.Tone AS DOUBLE)) AS tone_bin,
            ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) AS total_themes_count,
            
            list_transform(string_split(m.EnhancedThemes, ';'), x -> upper(trim(split_part(trim(x), ',', 1)))) AS themes_list,
            
            list_filter(
                list_distinct(list_transform(string_split(m.EnhancedLocations, ';'), x -> split_part(x, '#', 3))),
                c -> c != ''
            ) AS countries_list

        FROM temp_mapped m
        INNER JOIN read_parquet('valid_sources_whitelist.parquet') v ON m.Src_ID = v.Src_ID
        WHERE ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes};
        
        DROP TABLE temp_mapped;
    """)

def map_articles_to_regions(con):
    """
    Croise la liste des pays de chaque article avec nos cibles géographiques.
    Un article peut appartenir à plusieurs régions simultanément.
    """
    region_rows = [{"region_key": k, "country_code": c} for k, codes in REGIONS.items() for c in codes]
    con.register("regions_map", pd.DataFrame(region_rows))
    
    con.execute("""
        CREATE TABLE article_regions AS
        SELECT DISTINCT g.GKGRECORDID, g.period, rm.region_key
        FROM gkg_clean g, unnest(g.countries_list) AS c(code)
        INNER JOIN regions_map rm ON c.code = rm.country_code
    """)

def compute_total_news_regional(con):
    """
    Calcule le nombre d'articles TOTAL par jour ET par région.
    Ce sera notre dénominateur pour l'Attention locale.
    """
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

    # La requête groupe désormais par "period", "region_key" et "cat_key"
    query = """
    WITH 
    unnested AS (
        SELECT g.GKGRECORDID, g.period, g.tone, g.tone_bin, g.total_themes_count, unnest(g.themes_list) as theme,
               ar.region_key
        FROM gkg_clean g
        INNER JOIN article_regions ar ON g.GKGRECORDID = ar.GKGRECORDID
    ),
    matched AS (
        SELECT 
            u.GKGRECORDID, u.period, u.region_key, u.tone, u.tone_bin, u.total_themes_count, st.cat_key,
            COUNT(*) AS theme_hits
        FROM unnested u
        INNER JOIN sector_themes_tbl st ON u.theme = st.theme
        GROUP BY 1, 2, 3, 4, 5, 6, 7
    ),
    article_cat AS (
        SELECT 
            GKGRECORDID, period, region_key, cat_key, tone, tone_bin,
            (theme_hits::DOUBLE / total_themes_count) AS w
        FROM matched
    ),
    monthly_cat AS (
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
        FROM matched
        GROUP BY GKGRECORDID, period, region_key
    ),
    monthly_sector AS (
        SELECT 
            period, region_key, '__sector__' AS granularity,
            COUNT(*) AS N, SUM(w_sector) AS sum_w, SUM(tone) AS sum_t_cont, SUM(tone_bin) AS sum_t_bin,
            SUM(tone * w_sector) AS sum_t_cont_w, SUM(tone_bin * w_sector) AS sum_t_bin_w
        FROM article_sector GROUP BY period, region_key
    )

    SELECT * FROM monthly_cat UNION ALL SELECT * FROM monthly_sector
    """

    long_df = con.execute(query).df()
    if long_df.empty:
        return pd.DataFrame()

    total_news = con.execute("SELECT period, region_key, total_news_region FROM total_news_region_tbl").df()
    total_news["period"] = pd.to_datetime(total_news["period"])
    long_df["period"] = pd.to_datetime(long_df["period"])

    # Jointure avec le total régional
    long_df = long_df.merge(total_news, on=["period", "region_key"], how="left")

    # ── Calcul des Indicateurs ──────────────────────────────────────────────
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

    # ── PIVOTAGE PUISSANT (Pandas gère nativement le MultiIndex) ────────────
    # Résultat : une ligne par (Date, Région), et une colonne par (Métrique_Catégorie)
    pivot_df = long_df.pivot(index=["period", "region_key"], columns="granularity", values=metrics)
    
    # Aplatissement des noms de colonnes pour qu'ils soient lisibles
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
    p.add_argument("--output_dir",   type=Path, default=Path("./indicators_geo"))
    p.add_argument("--sectors",      nargs="*", default=None)
    p.add_argument("--min_words",               type=int, default=15)
    p.add_argument("--max_words",               type=int, default=6500)
    p.add_argument("--min_articles_per_source", type=int, default=30)
    p.add_argument("--min_active_years",        type=int, default=2)
    p.add_argument("--min_themes",              type=int, default=2)
    p.add_argument("--threads",    type=int, default=64)
    p.add_argument("--memory_gb",  type=int, default=200) 
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
    
    # ── ÉTAPE 0 : Whitelist (inchangé)
    whitelist_file = Path("valid_sources_whitelist.parquet")
    if not whitelist_file.exists():
        print("\n[ÉTAPE 0] Calcul de la Whitelist globale des sources...")
        if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        con = make_connection(args.threads, args.memory_gb)
        compute_global_whitelist(con, args.parquet_dir, args.source_map, args.min_articles_per_source, args.min_active_years)
        con.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    all_files = list(args.parquet_dir.glob("gdelt_*.parquet"))
    years = sorted(list(set([f.name.split('_')[1][:4] for f in all_files if f.name.split('_')[1][:4].isdigit()])))
    print(f"\n[INFO] Années détectées pour le traitement : {years}")
    print(f"[INFO] Régions configurées : {list(REGIONS.keys())}")

    for year in years:
        print(f"\n{'═'*65}\n  TRAITEMENT BATCH ANNÉE {year}\n{'═'*65}")
        t_year = time.time()
        glob_pattern = str(args.parquet_dir / f"gdelt_{year}*.parquet")
        
        if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        con = make_connection(args.threads, args.memory_gb)

        try:
            print("[1/4] Matérialisation de la base nettoyée ...")
            build_materialized_clean_table(con, glob_pattern, args.source_map, args.min_words, args.max_words, args.min_themes)

            print("[2/4] Cartographie des articles par zones géographiques ...")
            map_articles_to_regions(con)

            print("[3/4] Calcul du référentiel régional (Total News Local) ...")
            compute_total_news_regional(con)

            print(f"[4/4] Calcul des indicateurs géo ({len(sectors)} secteur(s)) ...")
            for i, (sector_key, sector_cfg) in enumerate(sectors.items(), 1):
                t0 = time.time()
                result_df = compute_sector_indicators_geo(con, sector_key, sector_cfg)

                if not result_df.empty:
                    out_path = args.output_dir / f"{sector_key}_{year}.parquet"
                    result_df.to_parquet(out_path, index=False)
                    print(f"    ✓ {sector_cfg.get('label', sector_key)} ({year}) : {len(result_df)} lignes, {len(result_df.columns)-2} cols - {_elapsed(t0)}")
                else:
                    print(f"    ⚠ {sector_cfg.get('label', sector_key)} : Aucune donnée pour l'année {year}.")

        except Exception as e:
            print(f"  [ERREUR] Impossible de traiter l'année {year}: {e}")
        finally:
            con.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  > Année {year} terminée en {_elapsed(t_year)}. Disque purgé.")

    print(f"\n{'═'*65}\nPipeline complet terminé en {_elapsed(t_total)}\n{'═'*65}\n")

if __name__ == "__main__":
    main()