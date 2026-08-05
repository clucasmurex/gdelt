"""
plot_tsne_clusters.py
=====================
Script autonome pour générer la figure t-SNE des articles (Façon Bybee et al. 2024).
Extrait un échantillon à la volée depuis les parquets bruts, calcule les poids, et trace.
"""

import argparse
import json
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec

# ── CONFIGURATION GÉOGRAPHIQUE ────────────────────────────────────────────────
REGIONS = {
    "France": ["FR"],
    "US": ["US"],
    # Vous pouvez recoller ici votre dictionnaire REGIONS complet si besoin
}

def make_connection(memory_gb: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    return con

def extract_article_sample(con, parquet_dir, source_map_path, retained_ids_path, config_path, target_region, sample_size):
    """
    Extrait un échantillon aléatoire d'articles et calcule le poids (att_weight) 
    de chaque sous-secteur au niveau de l'article individuel.
    """
    print(f"1. Extraction d'un échantillon de {sample_size} articles pour la région '{target_region}'...")
    
    # 1. Chargement des référentiels (Whitelist et Sources)
    with open(source_map_path, "r", encoding="utf-8") as f:
        source_map = json.load(f)
    con.register("src_map", pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in source_map["id_to_source"]],
        "SourceCommonName":    list(source_map["id_to_source"].values()),
    }))
    con.execute(f"CREATE TEMPORARY TABLE retained_ids AS SELECT column0::BIGINT AS id FROM read_csv('{retained_ids_path}', header=False)")
    
    # 2. Chargement des référentiels Régions
    region_rows = [{"region_key": target_region, "country_code": c} for c in REGIONS[target_region]]
    con.register("regions_map", pd.DataFrame(region_rows))
    
    # 3. Chargement du dictionnaire de thèmes (Secteurs)
    with open(config_path, encoding="utf-8") as f:
        sectors = json.load(f)["sectors"]
    
    theme_rows = []
    for s_key, s_cfg in sectors.items():
        for c_key, c_cfg in s_cfg["categories"].items():
            col_name = f"att_weight_{s_key}_{c_key}"
            for theme in c_cfg["themes"]:
                theme_rows.append({"col_name": col_name, "theme": theme.upper()})
    con.register("all_themes_tbl", pd.DataFrame(theme_rows))
    
    # 4. REQUÊTE SQL (Échantillonnage optimisé)
    glob_pattern = str(parquet_dir / "gdelt_*.parquet")
    
    query = f"""
    -- A. Prélèvement d'un très large échantillon brut pour encaisser les filtres
    CREATE TEMPORARY TABLE raw_sample AS
    SELECT GKGRECORDID, EnhancedThemes, EnhancedLocations, SourceCommonName_ID, DocumentIdentifier
    FROM read_parquet('{glob_pattern}')
    USING SAMPLE {sample_size * 20} ROWS;

    -- B. Filtrage (Whitelist + Région) et limitation stricte à 'sample_size'
    CREATE TEMPORARY TABLE valid_articles AS
    WITH mapped_sources AS (
        SELECT 
            r.*, COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
        FROM raw_sample r
        LEFT JOIN src_map m ON RTRIM(regexp_extract(r.DocumentIdentifier, 'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
    ),
    filtered_whitelist AS (
        SELECT m.GKGRECORDID, m.EnhancedThemes, m.EnhancedLocations
        FROM mapped_sources m
        INNER JOIN retained_ids rid ON m.Src_ID = rid.id
        WHERE m.EnhancedThemes IS NOT NULL AND m.EnhancedThemes != ''
    )
    SELECT DISTINCT f.GKGRECORDID, f.EnhancedThemes,
           ARRAY_LENGTH(string_split(f.EnhancedThemes, ';')) AS total_themes_count
    FROM filtered_whitelist f, unnest(list_transform(string_split(f.EnhancedLocations, ';'), x -> split_part(x, '#', 3))) AS c(code)
    INNER JOIN regions_map rm ON c.code = rm.country_code
    LIMIT {sample_size};

    -- C. Calcul des poids thématiques (att_weight)
    WITH unnested_themes AS (
        SELECT GKGRECORDID, total_themes_count, unnest(list_transform(string_split(EnhancedThemes, ';'), x -> upper(trim(split_part(trim(x), ',', 1))))) AS theme
        FROM valid_articles
    ),
    matched_themes AS (
        SELECT u.GKGRECORDID, u.total_themes_count, st.col_name, COUNT(*) AS theme_hits
        FROM unnested_themes u
        INNER JOIN all_themes_tbl st ON u.theme = st.theme
        GROUP BY 1, 2, 3
    ),
    article_weights AS (
        SELECT GKGRECORDID, col_name, (theme_hits::DOUBLE / total_themes_count) AS w
        FROM matched_themes
    )
    -- D. Pivot final pour Pandas
    PIVOT article_weights ON col_name USING SUM(w) GROUP BY GKGRECORDID;
    """
    
    df_articles = con.execute(query).df()
    df_articles = df_articles.fillna(0) # Les articles ne parlent pas de tous les sujets
    return df_articles

def plot_bybee_tsne(df_articles, topic_cols, filename="Fig_Bybee_tSNE.pdf"):
    """
    Trace la figure t-SNE selon les spécifications de Bybee et al. (2024).
    """
    print("2. Identification des dominantes et pureté des articles...")
    df = df_articles.copy()
    
    # Identification du sujet dominant et de sa proportion
    df['max_proportion'] = df[topic_cols].max(axis=1)
    df['dominant_topic'] = df[topic_cols].idxmax(axis=1).str.replace('att_weight_', '', regex=False)

    # Masques de pureté (seuils du papier de Bybee)
    mask_pure = df['max_proportion'] > 0.33
    mask_mixed = df['max_proportion'] < 0.25

    print(f"   > Articles Purs (>33%) : {mask_pure.sum():,}")
    print(f"   > Articles Mixtes (<25%) : {mask_mixed.sum():,}")

    print("3. Calcul de la projection t-SNE (peut prendre 1-3 minutes)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1, init='pca')
    embedding = tsne.fit_transform(df[topic_cols])
    
    df['tsne_1'] = embedding[:, 0]
    df['tsne_2'] = embedding[:, 1]
    
    print("4. Génération de la planche graphique...")
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})
    sns.set_theme(style="white")
    
    unique_topics = df['dominant_topic'].unique()
    palette = sns.color_palette("husl", len(unique_topics))
    color_map = dict(zip(unique_topics, palette))

    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    scatter_kws = {'s': 1.0, 'alpha': 0.7, 'edgecolor': 'none'}
    
    # --- GRAPHIQUE HAUT : TOUS LES ARTICLES ---
    ax_top = fig.add_subplot(gs[0, :])
    sns.scatterplot(
        data=df, x='tsne_1', y='tsne_2', 
        hue='dominant_topic', palette=color_map, legend=False, 
        ax=ax_top, **scatter_kws
    )
    ax_top.set_title("Article-level Nearest Neighbor Embedding (All Articles)", fontsize=14, pad=20)
    ax_top.axis('off')

    # --- GRAPHIQUE BAS GAUCHE : ARTICLES PURS (>33%) ---
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.scatter(df['tsne_1'], df['tsne_2'], color='lightgray', s=0.5, alpha=0.2)
    sns.scatterplot(
        data=df[mask_pure], x='tsne_1', y='tsne_2', 
        hue='dominant_topic', palette=color_map, legend=False, 
        ax=ax_left, **scatter_kws
    )
    ax_left.set_title("“Pure” Articles (Max Proportion > 33%)", fontsize=12, pad=10)
    ax_left.axis('off')

    # --- GRAPHIQUE BAS DROITE : ARTICLES MIXTES (<25%) ---
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.scatter(df['tsne_1'], df['tsne_2'], color='lightgray', s=0.5, alpha=0.2)
    sns.scatterplot(
        data=df[mask_mixed], x='tsne_1', y='tsne_2', 
        hue='dominant_topic', palette=color_map, legend=False, 
        ax=ax_right, **scatter_kws
    )
    ax_right.set_title("“Mixed” Articles (Max Proportion < 25%)", fontsize=12, pad=10)
    ax_right.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"✔ Figure sauvegardée : {filename}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir",  type=Path, default=Path("/data/gdelt/gdelt_parquet_db"))
    p.add_argument("--source_map",   type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--config",       type=Path, default=Path("./sectors_config.json"))
    p.add_argument("--retained_ids", type=Path, default=Path("liste_ids_retenus.txt"))
    p.add_argument("--region",       type=str, default="France")
    p.add_argument("--sample_size",  type=int, default=100000)
    args = p.parse_args()

    con = make_connection(memory_gb=50)
    try:
        df_articles = extract_article_sample(
            con, args.parquet_dir, args.source_map, args.retained_ids, 
            args.config, args.region, args.sample_size
        )
        
        if df_articles.empty:
            print("⚠ Aucun article trouvé avec ces filtres.")
            return

        topic_cols = [c for c in df_articles.columns if c.startswith('att_weight_')]
        plot_bybee_tsne(df_articles, topic_cols)
        
    finally:
        con.close()

if __name__ == "__main__":
    main()