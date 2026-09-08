#!/usr/bin/env python3
"""
check_categories_emergence.py
==============================================================================
Détermine à partir de quelle date chaque *sous-catégorie* (indicateur) est 
pleinement représentée dans l'échantillon "propre" (filtré par whitelist, 
WordCount, etc.). Fonctionne par lots mensuels pour préserver la RAM.
"""

import argparse
import json
import time
import re
from pathlib import Path
import shutil
import duckdb
import pandas as pd


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m{int(s%60):02d}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Chemins des données
    p.add_argument("--parquet_dir",   type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--config",        type=Path, default=Path("./sectors_config.json"))
    p.add_argument("--source_map",    type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--retained_ids",  type=Path, default=Path("./liste_ids_retenus.txt"))
    p.add_argument("--output_csv",    type=Path, default=Path("./categories_emergence_report.csv"))
    
    # Paramètres du filtre "Base Propre" (les mêmes que votre pipeline global)
    p.add_argument("--min_words",     type=int, default=150)
    p.add_argument("--max_words",     type=int, default=5500)
    p.add_argument("--min_themes",    type=int, default=2)
    
    # Paramètres DuckDB
    p.add_argument("--threads",       type=int, default=16)
    p.add_argument("--memory_gb",     type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()

    # ==========================================
    # 1. LECTURE DU JSON ET MAPPING DES CATÉGORIES
    # ==========================================
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    
    # Construction d'un tableau (Catégorie -> Thèmes) pour DuckDB
    cat_to_sector = {}
    rows_themes = []
    
    for sector_key, sector_cfg in config["sectors"].items():
        categories = sector_cfg.get("categories", {})
        for cat_key, cat_cfg in categories.items():
            cat_to_sector[cat_key] = sector_key
            for theme in cat_cfg.get("themes", []):
                rows_themes.append({
                    "cat_key": cat_key, 
                    "theme": theme.upper().strip()
                })

    df_cat_themes = pd.DataFrame(rows_themes)
    remaining_categories = set(cat_to_sector.keys())
    
    print(f"\n[INFO] Cible : {len(remaining_categories)} sous-catégories à valider.")

    # ==========================================
    # 2. IDENTIFICATION DES MOIS (Robuste)
    # ==========================================
    all_files = list(args.parquet_dir.glob("*.parquet"))
    months_set = set()
    for f in all_files:
        # Accepte les formats YYYYMM (ex: 201501) ou YYYY-MM (ex: 2025-06)
        match = re.search(r'(\d{4}-?\d{2})', f.name)
        if match:
            months_set.add(match.group(1))
            
    months = sorted(list(months_set))
    if not months:
        print(f"\n❌ ERREUR : Aucun fichier Parquet avec date n'a été trouvé dans '{args.parquet_dir}'")
        return
        
    print(f"[INFO] {len(months)} mois détectés pour l'analyse chronologique (de {months[0]} à {months[-1]})\n")

    # ==========================================
    # 3. INITIALISATION DUCKDB ET TABLES DE FILTRAGE
    # ==========================================
    tmp_dir = Path("./duckdb_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")

    # A. Chargement de la whitelist (retained_ids)
    con.execute(f"""
        CREATE TABLE retained_ids AS 
        SELECT column0::BIGINT AS id 
        FROM read_csv('{args.retained_ids}', header=False)
    """)
    
    # B. Chargement du dictionnaire des sources
    with open(args.source_map, "r", encoding="utf-8") as f:
        source_map = json.load(f)
    src_df = pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    })
    con.register("src_map", src_df)
    
    # C. Chargement des correspondances Thèmes <-> Catégories
    con.register("category_themes", df_cat_themes)

    cat_first_seen = {}
    current_year = ""

    try:
        # ==========================================
        # 4. BOUCLE MENSUELLE ET REQUÊTE STRICTE
        # ==========================================
        for month in months:
            if not remaining_categories:
                print("\n[🎉 SUCCÈS] L'univers complet des catégories est apparu ! Arrêt anticipé.")
                break
                
            year_of_month = month[:4]
            if year_of_month != current_year:
                current_year = year_of_month
                print(f"\n{'═'*65}")
                print(f"  ANNÉE {current_year}")
                print(f"{'═'*65}")

            t0 = time.time()
            glob_pattern = str(args.parquet_dir / f"*{month}*.parquet")
            
            cats_sql_list = ", ".join([f"'{c}'" for c in remaining_categories])

            # LA REQUÊTE QUI REPREND STRICTEMENT VOTRE PIPELINE NBER
            query = f"""
                WITH raw AS (
                    SELECT 
                        GKGRECORDID, 
                        strptime(substr(CAST(DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS article_date,
                        SourceCommonName_ID,
                        DocumentIdentifier,
                        EnhancedThemes
                    FROM read_parquet('{glob_pattern}')
                    WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\\d{{14}}$')
                      AND GKGRECORDID != '20210925181500-T1111'
                      AND EnhancedThemes IS NOT NULL
                      AND EnhancedThemes != ''
                      AND WordCount BETWEEN {args.min_words} AND {args.max_words}
                      AND ARRAY_LENGTH(string_split(EnhancedThemes, ';')) >= {args.min_themes}
                ),
                mapped AS (
                    -- Réparation des IDs comme dans build_gdelt_indicators
                    SELECT 
                        r.article_date,
                        r.EnhancedThemes,
                        COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
                    FROM raw r
                    LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
                    WHERE COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) IS NOT NULL
                ),
                clean_articles AS (
                    -- Filtrage par la Whitelist d'IDs
                    SELECT m.article_date, m.EnhancedThemes
                    FROM mapped m
                    INNER JOIN retained_ids rid ON m.Src_ID = rid.id
                ),
                exploded AS (
                    -- Éclatement des thèmes pour la correspondance
                    SELECT 
                        article_date,
                        UPPER(TRIM(split_part(trim(x), ',', 1))) AS theme_name
                    FROM clean_articles, unnest(string_split(EnhancedThemes, ';')) AS t(x)
                ),
                category_hits AS (
                    -- Validation de la sous-catégorie si l'un de ses thèmes est présent
                    SELECT 
                        ct.cat_key,
                        MIN(e.article_date) AS first_date
                    FROM exploded e
                    INNER JOIN category_themes ct ON e.theme_name = ct.theme
                    WHERE ct.cat_key IN ({cats_sql_list})
                    GROUP BY ct.cat_key
                )
                SELECT * FROM category_hits;
            """

            df_batch = con.execute(query).df()
            
            found_in_month = 0
            for _, row in df_batch.iterrows():
                c_name = row["cat_key"]
                f_date = row["first_date"]
                
                if c_name in remaining_categories:
                    if c_name not in cat_first_seen or f_date < cat_first_seen[c_name]:
                        cat_first_seen[c_name] = f_date
                        found_in_month += 1

            for c in list(remaining_categories):
                if c in cat_first_seen:
                    remaining_categories.remove(c)

            print(f"  └─ Mois {month} : {found_in_month} catégories validées | Reste à trouver: {len(remaining_categories)} | {_elapsed(t0)}")

    finally:
        con.close()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\n[INFO] Connexion fermée et cache purgé.")

    # ==========================================
    # 5. EXPORT DU RAPPORT
    # ==========================================
    report_rows = []
    for cat_key, sector in cat_to_sector.items():
        first_date = cat_first_seen.get(cat_key, None)
        status = "Couverture Validée" if first_date else "Jamais Apparue"
        
        report_rows.append({
            "Secteur": sector.capitalize(),
            "Sous_Categorie": cat_key.capitalize(),
            "Premiere_Date_Valide": first_date,
            "Statut": status
        })

    df_report = pd.DataFrame(report_rows)
    # Tri temporel pour que vous voyiez immédiatement les dates qui "tardent" à arriver
    df_report = df_report.sort_values(by=["Premiere_Date_Valide", "Secteur", "Sous_Categorie"], na_position="last")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(args.output_csv, index=False)
    
    print(f"\n{'═'*65}")
    print(f"  BILAN : ANALYSE TERMINÉE en {_elapsed(t_total)}")
    print(f"  Rapport exporté vers : {args.output_csv}")
    print(f"{'═'*65}\n")
    
    # Affichage propre en console des 20 dernières catégories à être apparues (pour repérer les goulets d'étranglement)
    print("Top 20 des sous-catégories apparues en dernier :")
    print(df_report.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()