"""
Script unifié de génération d'échantillons pour Robustness Checks (GDELT - V2)
==============================================================================
1. Charge la whitelist finale (format TXT).
2. Extrait 200 sources aléatoires du JSON pour évaluer les Faux Positifs / Faux Négatifs.
3. Extrait 100 articles aléatoires (sources retenues) pour évaluer la pertinence thématique.
"""

import argparse
import json
import random
from pathlib import Path
import duckdb
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Génération d'échantillons pour Robustness Checks (V2)")
    p.add_argument("--parquet_dir",  type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    # Utilisation du NOUVEAU fichier texte généré par le notebook
    p.add_argument("--retained_ids", type=Path, default=Path("gold_standard_whitelist_v2.txt"))
    p.add_argument("--output_dir",   type=Path, default=Path("./robustness_checks"))
    return p.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Connexion à DuckDB...")
    con = duckdb.connect()
    # On bride un peu la RAM pour éviter de perturber les autres processus du serveur
    con.execute("PRAGMA memory_limit='16GB'")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 1. CHARGEMENT DE LA WHITELIST TXT
    # ══════════════════════════════════════════════════════════════════════════
    print(f"[INFO] Chargement de la whitelist depuis {args.retained_ids}...")
    if not args.retained_ids.exists():
        raise FileNotFoundError(f"Le fichier {args.retained_ids} est introuvable.")
        
    # Création du SET Python (pour le check rapide des sources)
    with open(args.retained_ids, "r", encoding="utf-8") as f:
        retained_set = {int(line.strip()) for line in f if line.strip().isdigit()}
        
    # Création de la table temporaire DuckDB (pour les requêtes sur les articles)
    con.execute(f"""
        CREATE TEMPORARY TABLE retained_ids AS 
        SELECT column0::BIGINT AS id 
        FROM read_csv('{args.retained_ids}', header=False)
    """)
    
    # ══════════════════════════════════════════════════════════════════════════
    # 2. CHARGEMENT DU DICTIONNAIRE JSON
    # ══════════════════════════════════════════════════════════════════════════
    print(f"[INFO] Chargement du mapping des sources {args.source_map}...")
    with open(args.source_map, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        
    source_to_id = mapping_data.get("source_to_id", {})
    
    # Enregistrement du dictionnaire inversé dans DuckDB
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in mapping_data["id_to_source"].keys()],
        "SourceCommonName":    list(mapping_data["id_to_source"].values()),
    }))

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 1 : ÉCHANTILLON DE 200 SOURCES (FAUX POSITIFS / FAUX NÉGATIFS)
    # ══════════════════════════════════════════════════════════════════════════
    print("[INFO] Tirage aléatoire de 200 domaines pour vérification FP/FN...")
    all_sources = list(source_to_id.items())
    sample_size = min(200, len(all_sources))
    sampled_sources = random.sample(all_sources, sample_size)
    
    out_sources = args.output_dir / "check_200_sources_fp_fn.txt"
    with open(out_sources, "w", encoding="utf-8") as f:
        f.write("=== ROBUSTNESS CHECK : FAUX POSITIFS / FAUX NÉGATIFS ===\n")
        f.write("Vérifiez manuellement :\n")
        f.write("- Faux Négatifs : Les sources 'Rejetées' qui auraient dû être gardées.\n")
        f.write("- Faux Positifs : Les sources 'Retenues' qui auraient dû être jetées.\n\n")
        
        for i, (domain, src_id) in enumerate(sampled_sources, 1):
            status = "OUI (Retenue)" if src_id in retained_set else "NON (Rejetée)"
            f.write(f"[{i:03d}] {status:>15} | ID: {src_id:>8} | Domaine: {domain}\n")
            
    print(f"  ✓ Fichier généré : {out_sources}")

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 2 : ÉCHANTILLON DE 100 ARTICLES ET LEURS THÈMES (SOURCES RETENUES)
    # ══════════════════════════════════════════════════════════════════════════
    print("[INFO] Extraction de 100 articles (sources filtrées) sur plusieurs années...")
    
    # Le mot-clé USING SAMPLE de DuckDB est ultra performant pour ce besoin
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
            raw_themes = row['EnhancedThemes'].split(';')
            clean_themes = list(set([t.split(',')[0].strip().upper() for t in raw_themes if t]))
            
            f.write(f"--- Article {i+1:03d} ---\n")
            f.write(f"ID     : {row['GKGRECORDID']}\n")
            f.write(f"URL    : {row['url']}\n")
            f.write(f"THÈMES : {', '.join(clean_themes[:15])}")
            if len(clean_themes) > 15:
                f.write(f" ... (+{len(clean_themes)-15} autres)")
            f.write("\n\n")

    print(f"  ✓ Fichier généré : {out_articles}")
    print("\n[SUCCÈS] Génération terminée. Bon courage pour l'annotation manuelle !")

if __name__ == "__main__":
    main()