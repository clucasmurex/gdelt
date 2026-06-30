"""
build_gdelt_indicators_fast_v10.py
==================================
Pipeline ultra-optimisé calculant simultanément 10 indicateurs croisés 
(Attention, Sentiments Continus/Binaires, Pondérés/Non-Pondérés et Interactions)
grâce à une matérialisation en RAM DuckDB.
"""

import argparse
import json
import time
from pathlib import Path
import shutil 
import glob   
import duckdb
import pandas as pd
import numpy as np


def _fmt(n: int) -> str:
    return f"{n:>14,}"

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m{int(s%60):02d}s"
def make_connection(threads: int, memory_gb: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    
    # CORRECTION 1 : On crée un dossier temporaire LOCAL au lieu de saturer le /tmp global
    tmp_dir = Path("./duckdb_tmp")
    tmp_dir.mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    
    # OPTIMISATION : On dit à DuckDB de ne pas garder l'ordre d'insertion en mémoire,
    # ce qui permet d'économiser énormément de ressources lors des GROUP BY.
    con.execute("SET preserve_insertion_order=false")
    
    return con


# ══════════════════════════════════════════════════════════════════════════════
# 1. MATÉRIALISATION DE LA TABLE UNIQUE NETTOYÉE
# ══════════════════════════════════════════════════════════════════════════════
def compute_global_whitelist(
    con: duckdb.DuckDBPyConnection,
    parquet_dir: Path,
    source_map_path: Path,
    min_articles: int,
    min_years: int
) -> None:
    """
    Scanne toute la base rapidement (uniquement les colonnes nécessaires) 
    pour identifier les sources globalement valides.
    """
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
            -- On ne lit QUE ce dont on a besoin, ce sera très rapide
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
        HAVING COUNT(*) >= {min_articles}
           AND COUNT(DISTINCT pub_year) >= {min_years};
    """)
    
    # On sauvegarde la liste dans un fichier sur le disque
    con.execute("COPY global_valid_sources TO 'valid_sources_whitelist.parquet' (FORMAT PARQUET)")
    
    n = con.execute("SELECT COUNT(*) FROM global_valid_sources").fetchone()[0]
    print(f"  ✓ Whitelist globale générée : {n:,} sources respectent les critères (>={min_years} ans, >={min_articles} articles).")

def build_materialized_clean_table(
    con: duckdb.DuckDBPyConnection,
    glob_pattern: str,
    source_map_path: Path,
    min_words: int,
    max_words: int,
    min_themes: int,
) -> None:
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
        -- LA MAGIE EST ICI : On croise avec la whitelist globale
        INNER JOIN read_parquet('valid_sources_whitelist.parquet') v ON m.Src_ID = v.Src_ID
        WHERE ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes};
        
        DROP TABLE temp_mapped;
    """)
    n = con.execute("SELECT COUNT(*) FROM gkg_clean").fetchone()[0]
    print(f"  Table `gkg_clean` matérialisée avec {n:,} articles pour ce batch.")

def compute_total_news(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE total_news_tbl AS
        SELECT period, COUNT(DISTINCT GKGRECORDID) AS total_news
        FROM gkg_clean
        GROUP BY 1 ORDER BY 1
    """)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALCUL DES 10 INDICATEURS
# ══════════════════════════════════════════════════════════════════════════════

def compute_sector_indicators(
    con: duckdb.DuckDBPyConnection,
    sector_key: str,
    sector_cfg: dict
) -> pd.DataFrame:
    categories = sector_cfg["categories"]
    
    rows = [
        {"cat_key": cat_key, "theme": theme.upper()}
        for cat_key, cat_cfg in categories.items()
        for theme in cat_cfg["themes"]
    ]
    con.register("sector_themes_tbl", pd.DataFrame(rows))

    query = """
    WITH 
    unnested AS (
        SELECT GKGRECORDID, period, tone, tone_bin, total_themes_count, unnest(themes_list) as theme
        FROM gkg_clean
    ),
    matched AS (
        SELECT 
            u.GKGRECORDID, u.period, u.tone, u.tone_bin, u.total_themes_count,
            st.cat_key,
            COUNT(*) AS theme_hits
        FROM unnested u
        INNER JOIN sector_themes_tbl st ON u.theme = st.theme
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    article_cat AS (
        SELECT 
            GKGRECORDID, period, cat_key, tone, tone_bin,
            (theme_hits::DOUBLE / total_themes_count) AS w
        FROM matched
    ),
    monthly_cat AS (
        SELECT 
            period,
            cat_key AS granularity,
            COUNT(*) AS N,
            SUM(w) AS sum_w,
            SUM(tone) AS sum_t_cont,
            SUM(tone_bin) AS sum_t_bin,
            SUM(tone * w) AS sum_t_cont_w,
            SUM(tone_bin * w) AS sum_t_bin_w
        FROM article_cat
        GROUP BY period, cat_key
    ),
    article_sector AS (
        SELECT 
            GKGRECORDID, period,
            ANY_VALUE(tone) AS tone, ANY_VALUE(tone_bin) AS tone_bin,
            SUM(theme_hits)::DOUBLE / ANY_VALUE(total_themes_count) AS w_sector
        FROM matched
        GROUP BY GKGRECORDID, period
    ),
    monthly_sector AS (
        SELECT 
            period,
            '__sector__' AS granularity,
            COUNT(*) AS N,
            SUM(w_sector) AS sum_w,
            SUM(tone) AS sum_t_cont,
            SUM(tone_bin) AS sum_t_bin,
            SUM(tone * w_sector) AS sum_t_cont_w,
            SUM(tone_bin * w_sector) AS sum_t_bin_w
        FROM article_sector
        GROUP BY period
    )

    SELECT * FROM monthly_cat
    UNION ALL
    SELECT * FROM monthly_sector
    ORDER BY granularity, period
    """

    t0 = time.time()
    long_df = con.execute(query).df()
    print(f"    Requête SQL : {_elapsed(t0)}")

    if long_df.empty:
        return pd.DataFrame()

    total_news = con.execute("SELECT period, total_news FROM total_news_tbl").df()
    total_news["period"] = pd.to_datetime(total_news["period"])
    long_df["period"] = pd.to_datetime(long_df["period"])

    long_df = long_df.merge(total_news, on="period", how="left")

    # ── 1. Les Attentions ───────────────────────────────────────────────────
    long_df["att"]          = long_df["N"] / long_df["total_news"]
    long_df["att_weight"] = long_df["sum_w"] / long_df["total_news"]
    
    # ── 2 & 3. Les Sentiments (Continus et Binaires) ────────────────────────
    long_df["sent_cont"]          = np.where(long_df["N"] > 0, long_df["sum_t_cont"] / long_df["N"], np.nan)
    long_df["sent_bin"]           = np.where(long_df["N"] > 0, long_df["sum_t_bin"] / long_df["N"], np.nan)
    long_df["sent_cont_weight"] = np.where(long_df["sum_w"] > 0, long_df["sum_t_cont_w"] / long_df["sum_w"], np.nan)
    long_df["sent_bin_weight"]  = np.where(long_df["sum_w"] > 0, long_df["sum_t_bin_w"] / long_df["sum_w"], np.nan)

    # ── 4. Les Interactions (Attention x Sentiment) ─────────────────────────
    long_df["axs_cont"]          = long_df["att"] * long_df["sent_cont"]
    long_df["axs_cont_weight"] = long_df["att_weight"] * long_df["sent_cont_weight"]
    long_df["axs_bin"]           = long_df["att"] * long_df["sent_bin"]
    long_df["axs_bin_weight"]  = long_df["att_weight"] * long_df["sent_bin_weight"]

    # ── Pivotage vers format large ──────────────────────────────────────────
    metrics = [
        "att", "att_weight", 
        "sent_cont", "sent_bin", 
        "sent_cont_weight", "sent_bin_weight",
        "axs_cont", "axs_cont_weight",
        "axs_bin", "axs_bin_weight"
    ]
    
    all_periods = total_news["period"].sort_values().unique()
    result = pd.DataFrame({"period": all_periods})

    for gran in long_df["granularity"].unique():
        sub = long_df[long_df["granularity"] == gran].set_index("period")
        suffix = sector_key if gran == "__sector__" else f"{sector_key}_{gran}"

        for metric in metrics:
            result[f"{metric}_{suffix}"] = result["period"].map(sub[metric]).astype(float)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir",  type=Path, default=Path("./gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("./gdelt_sources_mapping.json"))
    p.add_argument("--config",       type=Path, default=Path("./sectors_config.json"))
    p.add_argument("--output_dir",   type=Path, default=Path("./indicators"))
    p.add_argument("--sectors",      nargs="*", default=None)
    
    p.add_argument("--min_words",               type=int, default=15)
    p.add_argument("--max_words",               type=int, default=6500)
    p.add_argument("--min_articles_per_source", type=int, default=30)
    p.add_argument("--min_active_years",        type=int, default=2)
    p.add_argument("--min_themes",              type=int, default=2)

    p.add_argument("--threads",    type=int, default=64)
    p.add_argument("--memory_gb",  type=int, default=150)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    t_total = time.time()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    sectors = {k: config["sectors"][k] for k in args.sectors} if args.sectors else config["sectors"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gérer le dossier temporaire proprement
    tmp_dir = Path("./duckdb_tmp")
    
    # ── ÉTAPE 0 : Génération de la Whitelist globale ─────────────────────────
    whitelist_file = Path("valid_sources_whitelist.parquet")
    if not whitelist_file.exists():
        print("\n[ÉTAPE 0] Calcul de la Whitelist globale des sources...")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        
        con = make_connection(args.threads, args.memory_gb)
        compute_global_whitelist(
            con, args.parquet_dir, args.source_map, 
            args.min_articles_per_source, args.min_active_years
        )
        con.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print("\n[ÉTAPE 0] Whitelist globale détectée. Utilisation du cache existant.")

    # ── Identifier les années à traiter ──────────────────────────────────────
    all_files = list(args.parquet_dir.glob("gdelt_*.parquet"))
    years = sorted(list(set([f.name.split('_')[1][:4] for f in all_files if f.name.split('_')[1][:4].isdigit()])))
    print(f"\n[INFO] Années détectées pour le traitement : {years}")

    # ── 2. Boucle Principale par Année ──────────────────────────────────────
    for year in years:
        print(f"\n{'═'*65}")
        print(f"  TRAITEMENT DU BATCH : ANNÉE {year}")
        print(f"{'═'*65}")
        
        t_year = time.time()
        
        # Cibler uniquement les fichiers de l'année en cours
        glob_pattern = str(args.parquet_dir / f"gdelt_{year}*.parquet")
        
        # Purge agressive du dossier temporaire pour libérer le disque dur
        tmp_dir = Path("./duckdb_tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)

        # Nouvelle connexion propre pour chaque année
        con = make_connection(args.threads, args.memory_gb)

        try:
            print("[1/3] Matérialisation de la base nettoyée ...")
            t0 = time.time()
            build_materialized_clean_table(
                con, glob_pattern, args.source_map, 
                args.min_words, args.max_words, 
                args.min_themes
            )
            print(f"  Temps : {_elapsed(t0)}")

            print("[2/3] Calcul du référentiel journalier ...")
            compute_total_news(con)

            print(f"[3/3] Calcul des indicateurs ({len(sectors)} secteur(s)) ...")
            for i, (sector_key, sector_cfg) in enumerate(sectors.items(), 1):
                label = sector_cfg.get("label", sector_key)
                
                t0 = time.time()
                result_df = compute_sector_indicators(con, sector_key, sector_cfg)

                if not result_df.empty:
                    # Écriture d'un fichier PAR ANNÉE pour ce secteur
                    out_path = args.output_dir / f"{sector_key}_{year}.parquet"
                    result_df.to_parquet(out_path, index=False)
                    print(f"    ✓ {label} ({year}) : {len(result_df)} jrs, {len(result_df.columns)-1} cols - {_elapsed(t0)}")
                else:
                    print(f"    ⚠ {label} : Aucune donnée pour l'année {year}.")

        except Exception as e:
            print(f"  [ERREUR] Impossible de traiter l'année {year}: {e}")
        
        finally:
            # Fermeture de la connexion et purge du disque
            con.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\n  > Année {year} terminée en {_elapsed(t_year)}. Disque dur purgé.")

    print(f"\n{'═'*65}")
    print(f"Pipeline complet terminé en {_elapsed(t_total)}")
    print(f"{'═'*65}\n")

if __name__ == "__main__":
    main()