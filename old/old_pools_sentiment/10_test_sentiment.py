import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MACRO : INFLATION FRANCE 2022
# =============================================================================
def get_france_inflation(year=2022):
    start_date = f'{year-1}-12-01'
    end_date = f'{year}-12-31'
    df = web.DataReader(['CP0000FRM086NEST'], 'fred', start_date, end_date)
    df = df.rename(columns={'CP0000FRM086NEST': 'inflation'})
    df['inflation'] = df['inflation'].pct_change() * 100
    df = df.dropna()
    df.index = df.index.tz_localize(None)
    return df[df.index.year == year]

# =============================================================================
# 2. PIPELINE DE CLASSIFICATION ET EXPORT
# =============================================================================
def build_and_export_trust_index(parquet_dir: str, year: int = 2022):
    print(f"Extraction de l'inflation pour l'année {year}...")
    macro_df = get_france_inflation(year)
    macro_df['period'] = macro_df.index
    
    path = Path(parquet_dir)
    sectors_to_load = ['energy', 'agriculture']
    dfs = []
    
    for sector in sectors_to_load:
        file_path = path / f"{sector}_{year}_source_monthly.parquet"
        if file_path.exists():
            dfs.append(pd.read_parquet(file_path))
            
    if not dfs:
        print("Erreur : Aucune donnée trouvée dans le répertoire spécifié.")
        return
        
    print("Fusion des données sectorielles...")
    df_gdelt = dfs[0]
    for i in range(1, len(dfs)):
        df_gdelt = df_gdelt.merge(dfs[i], on=['period', 'region_key', 'Src_ID', 'SourceCommonName'], how='outer')

    df_fr = df_gdelt[df_gdelt['region_key'] == 'France'].copy()
    df_fr['period'] = pd.to_datetime(df_fr['period'])
    df_merged = df_fr.merge(macro_df[['period', 'inflation']], on='period', how='inner')
    
    crises = {
        'sent_bin_weight_energy_fossil': 'Energie',
        'sent_bin_weight_agriculture_food_prices': 'Alimentation'
    }
    
    media_tracker = []
    sources = df_merged['SourceCommonName'].unique()
    print(f"Analyse de {len(sources)} sources médiatiques en cours...")
    
    for source in sources:
        df_source = df_merged[df_merged['SourceCommonName'] == source]
        
        covered_topics = 0
        failed_topics = 0
        passed_topics = 0
        
        for ind, crisis_name in crises.items():
            if ind in df_source.columns:
                valid_data = df_source[['period', ind, 'inflation']].dropna()
                if len(valid_data) >= 6:
                    covered_topics += 1
                    corr, _ = pearsonr(valid_data[ind], valid_data['inflation'])
                    
                    if corr > 0:
                        failed_topics += 1
                    elif corr < 0:
                        passed_topics += 1

        if covered_topics > 0:
            if passed_topics > 0 and failed_topics == 0:
                classement = "Consommateur"
            elif failed_topics > 0 and passed_topics == 0:
                classement = "Corporate"
            elif passed_topics > 0 and failed_topics > 0:
                classement = "Mixte"
            else:
                classement = "Non-significatif"

            media_tracker.append({
                'Src_ID': int(df_source['Src_ID'].iloc[0]),
                'Classification': classement
            })
            
    df_results = pd.DataFrame(media_tracker)
    
    # 3. Séparation des pools et export au format texte (un ID par ligne)
    df_consumer = df_results[df_results['Classification'] == 'Consommateur']
    df_corporate = df_results[df_results['Classification'] == 'Corporate']
    
    consumer_file = "media_pool_consommateur.txt"
    corporate_file = "media_pool_corporate.txt"
    
    df_consumer['Src_ID'].to_csv(consumer_file, index=False, header=False)
    df_corporate['Src_ID'].to_csv(corporate_file, index=False, header=False)
    
    print("\n--- Bilan de l'export ---")
    print(f"IDs 'Consommateur' (Main Street) sauvegardés : {len(df_consumer)} ({consumer_file})")
    print(f"IDs 'Corporate' (Wall Street) sauvegardés : {len(df_corporate)} ({corporate_file})")

if __name__ == "__main__":
    build_and_export_trust_index(parquet_dir="./data/indicators_geo_source_monthly", year=2022)