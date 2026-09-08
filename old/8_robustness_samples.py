"""
Script de génération d'échantillons pour Robustness Checks (GDELT)
==================================================================
1. Extrait 100 sources aléatoires (brutes) et indique si elles ont été retenues.
2. Extrait 100 articles aléatoires avec leurs URLs et thèmes pour lecture manuelle.
"""

import argparse
import json
from pathlib import Path
import duckdb
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir",  type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--retained_ids", type=Path, default=Path("liste_ids_retenus.txt"))
    p.add_argument("--output_dir",   type=Path, default=Path("./robustness_checks"))
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialisation de DuckDB
    print("[INFO] Connexion à DuckDB...")
    con = duckdb.connect()
    
    # Récupération d'un fichier Parquet au hasard pour l'échantillonnage rapide
    # (Évite de charger toute la base de données juste pour 100 lignes)
    parquet_files = list(args.parquet_dir.glob("gdelt_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun fichier parquet trouvé dans {args.parquet_dir}")
    sample_file = str(parquet_files[0]) 

    # 2. Chargement du mapping et des identifiants retenus
    print("[INFO] Chargement des dictionnaires de sources...")
    with open(args.source_map, "r", encoding="utf-8") as f:
        source_map = json.load(f)
        
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    }))

    con.execute(f"""
        CREATE TEMPORARY TABLE retained_ids AS 
        SELECT column0::BIGINT AS id 
        FROM read_csv('{args.retained_ids}', header=False)
    """)

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 1 : ÉCHANTILLON DE 100 SOURCES (BRUTES VS NETTOYÉES)
    # ══════════════════════════════════════════════════════════════════════════
    print("[INFO] Extraction de 100 sources aléatoires...")
    
    # Extraction du domaine brut de l'URL comme dans ton script original
    query_sources = f"""
        WITH random_raw_docs AS (
            SELECT DocumentIdentifier 
            FROM read_parquet('{sample_file}') 
            USING SAMPLE 5000 ROWS
        ),
        extracted_domains AS (
            SELECT DISTINCT RTRIM(regexp_extract(DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') AS domain
            FROM random_raw_docs
            WHERE DocumentIdentifier IS NOT NULL
        )
        SELECT 
            d.domain AS source_brute, 
            m.SourceCommonName_ID AS source_id,
            CASE WHEN rid.id IS NOT NULL THEN 'OUI (Retenue)' ELSE 'NON (Rejetée)' END AS status_selection
        FROM extracted_domains d
        LEFT JOIN src_map m ON d.domain = m.SourceCommonName
        LEFT JOIN retained_ids rid ON m.SourceCommonName_ID = rid.id
        WHERE d.domain != ''
        LIMIT 100
    """
    
    sources_df = con.execute(query_sources).df()
    
    out_sources = args.output_dir / "check_100_sources.txt"
    with open(out_sources, "w", encoding="utf-8") as f:
        f.write("=== ROBUSTNESS CHECK : SÉLECTION DES SOURCES ===\n")
        f.write("Vérifiez manuellement si les sources 'Rejetées' auraient dû être gardées et inversement.\n\n")
        for i, row in sources_df.iterrows():
            src_id = f"ID: {int(row['source_id'])}" if pd.notna(row['source_id']) else "ID: Inconnu"
            f.write(f"[{i+1:03d}] {row['status_selection']:>15} | {src_id:>12} | Domaine: {row['source_brute']}\n")
            
    print(f"  ✓ Fichier généré : {out_sources}")

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 2 : ÉCHANTILLON DE 100 ARTICLES ET LEURS THÈMES (SOURCES RETENUES UNIQUEMENT + MULTI-YEAR)
    # ══════════════════════════════════════════════════════════════════════════
    print("[INFO] Extraction de 100 articles (sources filtrées) sur plusieurs années...")
    
    query_articles = f"""
    WITH raw_docs AS (
        SELECT 
            GKGRECORDID,
            DocumentIdentifier AS url,
            EnhancedThemes,
            RTRIM(regexp_extract(DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') AS domain
        FROM read_parquet('{args.parquet_dir}/gdelt_*.parquet')
        WHERE EnhancedThemes IS NOT NULL 
          AND EnhancedThemes != ''
    ),
    filtered_docs AS (
        SELECT d.GKGRECORDID, d.url, d.EnhancedThemes
        FROM raw_docs d
        JOIN src_map m ON d.domain = m.SourceCommonName
        JOIN retained_ids rid ON m.SourceCommonName_ID = rid.id
    )
    SELECT * 
    FROM filtered_docs 
    USING SAMPLE 100 ROWS
    """

    articles_df = con.execute(query_articles).df()
    
    out_articles = args.output_dir / "check_100_articles_themes.txt"
    with open(out_articles, "w", encoding="utf-8") as f:
        f.write("=== ROBUSTNESS CHECK : PERTINENCE DES THÈMES ===\n")
        f.write("Lisez l'article via l'URL et vérifiez si les thèmes GDELT assignés sont cohérents.\n\n")
        for i, row in articles_df.iterrows():
            # Nettoyage et formatage des thèmes comme dans ta pipeline
            raw_themes = row['EnhancedThemes'].split(';')
            clean_themes = list(set([t.split(',')[0].strip().upper() for t in raw_themes if t]))
            
            f.write(f"--- Article {i+1:03d} ---\n")
            f.write(f"ID     : {row['GKGRECORDID']}\n")
            f.write(f"URL    : {row['url']}\n")
            f.write(f"THÈMES : {', '.join(clean_themes[:15])}") # Limité à 15 pour la lisibilité
            if len(clean_themes) > 15:
                f.write(f" ... (+{len(clean_themes)-15} autres)")
            f.write("\n\n")

    print(f"  ✓ Fichier généré : {out_articles}")
    print("\n[SUCCÈS] Génération terminée. Bon courage pour l'annotation manuelle !")

if __name__ == "__main__":
    main()