import pandas as pd
import numpy as np
from pathlib import Path
import pandas_datareader.data as web
import statsmodels.api as sm
import warnings

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
# 2. CALCUL DE LA SENSIBILITÉ (BETA) ET EXTRACTION DES POOLS
# =============================================================================
def extract_media_pools_by_beta(parquet_dir: str, year: int = 2022):
    print(f"Extraction de l'inflation FR pour {year}...")
    macro_df = get_france_inflation(year)
    macro_df['period'] = macro_df.index
    
    path = Path(parquet_dir)
    # On se concentre sur les deux vecteurs d'inflation les plus purs : Énergie et Alimentation
    sectors_to_load = ['energy', 'agriculture']
    dfs = []
    
    for sector in sectors_to_load:
        file_path = path / f"{sector}_{year}_source_monthly.parquet"
        if file_path.exists():
            dfs.append(pd.read_parquet(file_path))
            
    if not dfs:
        print("❌ Aucune donnée trouvée.")
        return
        
    df_gdelt = dfs[0]
    for i in range(1, len(dfs)):
        df_gdelt = df_gdelt.merge(dfs[i], on=['period', 'region_key', 'Src_ID', 'SourceCommonName'], how='outer')

    df_fr = df_gdelt[df_gdelt['region_key'] == 'France'].copy()
    df_fr['period'] = pd.to_datetime(df_fr['period'])
    df_merged = df_fr.merge(macro_df[['period', 'inflation']], on='period', how='inner')
    
    indicators = [
        'sent_bin_weight_energy_fossil',
        'sent_bin_weight_agriculture_food_prices'
    ]
    
    media_pool = []
    sources = df_merged['SourceCommonName'].unique()
    
    for source in sources:
        df_source = df_merged[df_merged['SourceCommonName'] == source]
        
        source_betas = []
        source_means = []
        
        for ind in indicators:
            if ind in df_source.columns:
                valid_data = df_source[['period', ind, 'inflation']].dropna()
                # On exige au moins 8 mois de données pour une régression robuste
                if len(valid_data) >= 8:
                    X = sm.add_constant(valid_data['inflation'])
                    y = valid_data[ind]
                    model = sm.OLS(y, X).fit()
                    
                    source_betas.append(model.params['inflation'])
                    source_means.append(y.mean())
                    
        if source_betas:
            avg_beta = np.mean(source_betas)
            avg_level = np.mean(source_means)
            
            # CLASSIFICATION ÉCONOMÉTRIQUE
            # 1. Pool Consommateur : Sensibilité négative (le ton s'assombrit avec l'inflation) ET Ton global négatif
            if avg_beta < 0 and avg_level < 0:
                pool = "Consommateur"
            # 2. Pool Corporate : Sensibilité positive (le ton s'améliore ou résiste avec l'inflation)
            elif avg_beta > 0:
                pool = "Corporate"
            else:
                pool = "Neutre/Inclassable"
                
            media_pool.append({
                'Src_ID': int(df_source['Src_ID'].iloc[0]),
                'Source': source,
                'Beta_Moyen': avg_beta,
                'Niveau_Moyen': avg_level,
                'Pool': pool
            })
            
    df_results = pd.DataFrame(media_pool)
    
    # Export des identifiants purs (.txt) pour votre pipeline Group Lasso
    df_consumer = df_results[df_results['Pool'] == 'Consommateur']
    df_corporate = df_results[df_results['Pool'] == 'Corporate']
    
    df_consumer['Src_ID'].to_csv("pool_consommateur_ids.txt", index=False, header=False)
    df_corporate['Src_ID'].to_csv("pool_corporate_ids.txt", index=False, header=False)
    
    # Export du tableau détaillé (.csv) pour justification dans le mémoire
    df_results.to_csv("media_classification_details_2022.csv", index=False)
    
    print(f"\n{'='*60}")
    print(" RÉSULTATS DE LA SÉLECTION PAR ÉLASTICITÉ (BETA)")
    print(f"{'='*60}")
    print(f" Total médias évalués (>= 8 mois de data) : {len(df_results)}")
    print(f" 🛒 Pool 'Consommateur' (Beta < 0 & Mean < 0) : {len(df_consumer)} médias sauvegardés.")
    print(f" 🏢 Pool 'Corporate/PR' (Beta > 0)          : {len(df_corporate)} médias sauvegardés.")
    print("\n -> Fichiers 'pool_consommateur_ids.txt' et 'pool_corporate_ids.txt' générés avec succès.")

if __name__ == "__main__":
    extract_media_pools_by_beta(parquet_dir="./data/indicators_geo_source_monthly", year=2022)