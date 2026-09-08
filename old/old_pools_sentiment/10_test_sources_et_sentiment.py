import pandas as pd
import numpy as np
from pathlib import Path
import pandas_datareader.data as web
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RÉCUPÉRATION DE L'INFLATION (Ground Truth Macro)
# =============================================================================
def get_france_inflation_2022():
    """
    Récupère l'inflation française (Headline CPI) pour 2022 via l'API FRED,
    en appliquant la transformation pct_change comme dans votre modèle académique.
    """
    start_date = '2021-12-01'
    end_date = '2022-12-31'
    # Ticker CPI France issu de votre notebook
    df = web.DataReader(['CP0000FRM086NEST'], 'fred', start_date, end_date)
    df = df.rename(columns={'CP0000FRM086NEST': 'inflation'})
    
    # Transformation en pourcentage de variation mensuelle
    df['inflation'] = df['inflation'].pct_change() * 100
    df = df.dropna()
    df.index = df.index.tz_localize(None)
    
    # On ne garde que l'année 2022
    return df[df.index.year == 2022]

# =============================================================================
# 2. FONCTION D'ÉVALUATION DES MÉDIAS
# =============================================================================
def evaluate_media_reliability(parquet_dir: str, year: int = 2022):
    print(f"\n{'='*80}")
    print(f" ANALYSE DE FIABILITÉ DES MÉDIAS - FRANCE {year}")
    print(f"{'='*80}")
    
    # 1. Charger l'inflation de la Ground Truth
    macro_df = get_france_inflation_2022()
    macro_df['period'] = macro_df.index
    print(f"✓ Données d'inflation {year} chargées avec succès.")
    
    # 2. Charger les données GDELT par source
    path = Path(parquet_dir)
    sectors_to_load = ['energy', 'agriculture', 'commodities']
    dfs = []
    
    for sector in sectors_to_load:
        file_path = path / f"{sector}_{year}_source_monthly.parquet"
        if file_path.exists():
            dfs.append(pd.read_parquet(file_path))
        else:
            print(f"⚠ Attention : Fichier manquant {file_path}")
            
    if not dfs:
        print("❌ Aucune donnée GDELT trouvée. Vérifiez le chemin 'parquet_dir'.")
        return
        
    # Fusionner tous les secteurs
    df_gdelt = dfs[0]
    for i in range(1, len(dfs)):
        df_gdelt = df_gdelt.merge(dfs[i], on=['period', 'region_key', 'Src_ID', 'SourceCommonName'], how='outer')

    # Filtrer uniquement la France
    df_fr = df_gdelt[df_gdelt['region_key'] == 'France'].copy()
    df_fr['period'] = pd.to_datetime(df_fr['period'])
    
    # Fusionner les sentiments avec l'inflation
    df_merged = df_fr.merge(macro_df[['period', 'inflation']], on='period', how='inner')
    
    # 3. Définir les indicateurs de la Ground Truth 2022
    target_indicators = [
        'sent_bin_weight_energy_fossil', 
        'sent_bin_weight_agriculture_food_prices',
        'sent_bin_weight_commodities_market_prices'
    ]
    
    results = []
    sources = df_merged['SourceCommonName'].unique()
    print(f"✓ Analyse en cours sur {len(sources)} médias distincts...\n")
    
    # 4. Calculs statistiques média par média
    for source in sources:
        df_source = df_merged[df_merged['SourceCommonName'] == source]
        
        # Filtre mathématique minimal : au moins 6 mois de données pour calculer Pearson
        if len(df_source) < 6:
            continue
            
        source_metrics = {'Source': source, 'Mois_Actifs': len(df_source)}
        
        for ind in target_indicators:
            if ind in df_source.columns:
                valid_data = df_source[['period', ind, 'inflation']].dropna()
                
                if len(valid_data) >= 6:
                    avg_sent = valid_data[ind].mean()
                    corr, pval = pearsonr(valid_data[ind], valid_data['inflation'])
                    
                    source_metrics[f'Avg_{ind}'] = avg_sent
                    source_metrics[f'Corr_{ind}'] = corr
                else:
                    source_metrics[f'Avg_{ind}'] = np.nan
                    source_metrics[f'Corr_{ind}'] = np.nan

        results.append(source_metrics)
        
    df_results = pd.DataFrame(results)
    
    # =========================================================================
    # 5. RÉSULTATS : FOCUS SUR L'ÉNERGIE ET L'ALIMENTAIRE
    # =========================================================================
    
    def print_leaderboard(df, indicator_name, label):
        avg_col = f'Avg_{indicator_name}'
        corr_col = f'Corr_{indicator_name}'
        
        if avg_col not in df.columns:
            return
            
        df_valid = df.dropna(subset=[avg_col, corr_col])
        if df_valid.empty:
            return
            
        total_media = len(df_valid)
        
        # Catégorisation
        reliable = df_valid[(df_valid[avg_col] < 0) & (df_valid[corr_col] < -0.1)].sort_values(by=corr_col)
        unreliable = df_valid[(df_valid[avg_col] > 0) | (df_valid[corr_col] > 0.1)].sort_values(by=avg_col, ascending=False)
        neutral = df_valid[(df_valid[avg_col] <= 0) & (df_valid[corr_col] >= -0.1) & (df_valid[corr_col] <= 0.1)]
        
        print(f"\n{'='*70}")
        print(f" THÈME : {label.upper()}")
        print(f" Ground Truth 2022 : Sentiment NÉGATIF, Corrélation NÉGATIVE (<-0.1)")
        print(f"{'='*70}")
        
        # --- QUANTIFICATION GLOBALE ---
        print(f"\n📊 RÉPARTITION GLOBALE (Sur {total_media} médias qualifiés) :")
        print(f" ✅ Fiables (En phase avec la crise) : {len(reliable)} ({(len(reliable)/total_media)*100:.1f}%)")
        print(f" ❌ Déconnectés (Biais positif ou inverse) : {len(unreliable)} ({(len(unreliable)/total_media)*100:.1f}%)")
        print(f" ➖ Zone Grise (Ton négatif mais décorrélé) : {len(neutral)} ({(len(neutral)/total_media)*100:.1f}%)")
        
        # --- LES BONS ÉLÈVES ---
        print("\n🏆 TOP 10 - MÉDIAS LES PLUS FIABLES")
        if not reliable.empty:
            display_rel = reliable[['Source', 'Mois_Actifs', avg_col, corr_col]].head(10).copy()
            display_rel.columns = ['Média', 'Mois', 'Sentiment_Moyen', 'Corrélation_Inflation']
            print(display_rel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        # --- LES MAUVAIS ÉLÈVES ---
        print("\n🤡 TOP 10 - MÉDIAS LES MOINS FIABLES")
        if not unreliable.empty:
            display_unrel = unreliable[['Source', 'Mois_Actifs', avg_col, corr_col]].head(10).copy()
            display_unrel.columns = ['Média', 'Mois', 'Sentiment_Moyen', 'Corrélation_Inflation']
            print(display_unrel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Affichage pour les deux thèmes majeurs de 2022
    print_leaderboard(df_results, 'sent_bin_weight_energy_fossil', "Énergies Fossiles (Crise du Gaz/Pétrole)")
    print_leaderboard(df_results, 'sent_bin_weight_agriculture_food_prices', "Prix de l'Alimentaire (Crise du Blé/Agricole)")

if __name__ == "__main__":
    # Assurez-vous que le dossier pointe vers vos exports par SOURCE
    evaluate_media_reliability(parquet_dir="./data/indicators_geo_source_monthly", year=2022)