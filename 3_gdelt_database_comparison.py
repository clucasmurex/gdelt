"""
compare_gdelt_stats.py
======================
Script rapide pour comparer la volumétrie (lignes et sources) 
entre la base GDELT brute et la base nettoyée selon les critères du pipeline.
"""

import duckdb
import pandas as pd
import json
import time
from pathlib import Path

def get_stats(
    parquet_dir="./gdelt_parquet_db", 
    source_map_path="./gdelt_sources_mapping.json",
    min_words=15,
    max_words=6500,
    min_themes=2
):
    # L'ajout de flush=True force Python à afficher le texte immédiatement dans le terminal
    print("⏳ Démarrage de l'analyse des statistiques GDELT (Raw vs Clean)...", flush=True)
    t0 = time.time()
    
    con = duckdb.connect()
    glob_pattern = f"{parquet_dir}/*.parquet"
    whitelist_file = "valid_sources_whitelist.parquet"
    
    # 1. Chargement du mapping des sources
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
        
    src_df = pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    })
    con.register("src_map", src_df)

    # 2. Statistiques RAW (Brutes)
    print("\n📊 Calcul des statistiques BRUTES en cours...", flush=True)
    raw_query = f"""
        SELECT 
            COUNT(*) as raw_lines,
            COUNT(DISTINCT RTRIM(regexp_extract(DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.')) as raw_sources
        FROM read_parquet('{glob_pattern}')
    """
    raw_lines, raw_sources = con.execute(raw_query).fetchone()
    print(f"  ✓ Terminé.", flush=True)

    # 3. Statistiques CLEAN (Nettoyées)
    print("\n🧹 Calcul des statistiques NETTOYÉES en cours...", flush=True)
    if not Path(whitelist_file).exists():
        print("  ⚠ Attention : 'valid_sources_whitelist.parquet' introuvable.", flush=True)
        print("  Veuillez d'abord exécuter l'ÉTAPE 0 du pipeline principal pour le générer.", flush=True)
        return

    clean_query = f"""
        WITH raw AS (
            SELECT * FROM read_parquet('{glob_pattern}')
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
              AND WordCount BETWEEN {min_words} AND {max_words}
        ),
        mapped AS (
            SELECT 
                COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID,
                r.EnhancedThemes
            FROM raw r
            LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
        )
        SELECT 
            COUNT(*) as clean_lines,
            COUNT(DISTINCT m.Src_ID) as clean_sources
        FROM mapped m
        INNER JOIN read_parquet('{whitelist_file}') v ON m.Src_ID = v.Src_ID
        WHERE m.Src_ID IS NOT NULL
          AND ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes}
    """
    clean_lines, clean_sources = con.execute(clean_query).fetchone()
    print(f"  ✓ Terminé.", flush=True)
    
    # 4. Affichage du Bilan
    print("\n" + "="*50, flush=True)
    print("🎯 BILAN DES PERTES (RAW vs CLEAN)", flush=True)
    print("="*50, flush=True)
    print(f" ARTICLES (Lignes) :", flush=True)
    print(f"   - Brut     : {raw_lines:>15,}", flush=True)
    print(f"   - Nettoyé  : {clean_lines:>15,}", flush=True)
    print(f"   - Perte    : {raw_lines - clean_lines:>15,} (-{((raw_lines - clean_lines) / raw_lines) * 100:.1f}%)", flush=True)
    print(f"", flush=True)
    print(f" SOURCES (Uniques) :", flush=True)
    print(f"   - Brut     : {raw_sources:>15,}", flush=True)
    print(f"   - Nettoyé  : {clean_sources:>15,}", flush=True)
    print(f"   - Perte    : {raw_sources - clean_sources:>15,} (-{((raw_sources - clean_sources) / raw_sources) * 100:.1f}%)", flush=True)
    print("="*50, flush=True)
    
    s = time.time() - t0
    print(f"⏱️  Temps d'exécution total : {int(s//60)}m{int(s%60):02d}s", flush=True)


# Lancement automatique de l'analyse
get_stats()