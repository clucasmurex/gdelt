import pandas as pd
from pathlib import Path
import numpy as np

def load_gdelt_base(dir_main_str, dir_uk_str):
    """Fonction utilitaire pour charger une base GDELT selon les chemins donnés."""
    dir_main = Path(dir_main_str)
    dir_uk = Path(dir_uk_str)
    
    # 1. Main (France/US)
    if dir_main.exists():
        files = list(dir_main.glob('*.parquet'))
        df_main = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df_main = pd.DataFrame()
        
    # 2. UK
    if dir_uk.exists():
        files_uk = list(dir_uk.glob('*.parquet'))
        if len(files_uk) > 0:
            df_uk = pd.concat([pd.read_parquet(f) for f in files_uk], ignore_index=True)
            df_uk['region_key'] = 'UK'
        else:
            df_uk = pd.DataFrame()
    else:
        df_uk = pd.DataFrame()
        
    # 3. Fusion et nettoyage
    df = pd.concat([df_main, df_uk], ignore_index=True)
    if not df.empty and 'region_key' in df.columns:
        df = df.groupby(['period', 'region_key']).max().reset_index().copy()
        df['period'] = pd.to_datetime(df['period'])
    return df

# ==============================================================================
# 1. CHARGEMENT DES DEUX BASES
# ==============================================================================
print("⏳ Chargement des bases de données...")
df_old = load_gdelt_base('./data/indicators_old/indicators_geo_monthly', 
                         './data/indicators_old/indicators_geo_monthly_UK')
df_new = load_gdelt_base('./data/indicators_geo_monthly', 
                         './data/indicators_geo_monthly_UK')

# ==============================================================================
# 2. ANALYSE COMPARATIVE PAR RÉGION
# ==============================================================================
regions = ['France', 'US', 'UK']

print("\n" + "="*70)
print(" 🕵️ RAPPORT DE DIAGNOSTIC : OLD vs NEW GDELT INDICATORS")
print("="*70)

for region in regions:
    print(f"\n🌍 RÉGION : {region.upper()}")
    print("-" * 40)
    
    # Filtrage par région et indexation sur la date
    df_old_reg = df_old[df_old['region_key'] == region].set_index('period')
    df_new_reg = df_new[df_new['region_key'] == region].set_index('period')
    
    # Isoler uniquement les colonnes 'att_weight_'
    cols_old = set([c for c in df_old_reg.columns if c.startswith('att_weight_')])
    cols_new = set([c for c in df_new_reg.columns if c.startswith('att_weight_')])
    
    # 1. Analyse des colonnes (Disparitions / Apparitions)
    missing_in_new = cols_old - cols_new
    added_in_new = cols_new - cols_old
    
    print(f"📊 Nombre de variables 'att_weight_' -> OLD: {len(cols_old)} | NEW: {len(cols_new)}")
    if missing_in_new:
        print(f"  ❌ Variables disparues dans la NEW : {len(missing_in_new)} (ex: {list(missing_in_new)[:3]})")
    if added_in_new:
        print(f"  🆕 Variables ajoutées dans la NEW : {len(added_in_new)} (ex: {list(added_in_new)[:3]})")
        
    # 2. Analyse Temporelle
    print(f"📅 Dates OLD : {df_old_reg.index.min().date()} à {df_old_reg.index.max().date()} ({len(df_old_reg)} mois)")
    print(f"📅 Dates NEW : {df_new_reg.index.min().date()} à {df_new_reg.index.max().date()} ({len(df_new_reg)} mois)")
    
    # 3. Analyse Mathématique sur les variables communes
    common_cols = list(cols_old.intersection(cols_new))
    if common_cols:
        # Alignement strict sur les dates communes pour pouvoir comparer mathématiquement
        df_old_align, df_new_align = df_old_reg[common_cols].align(df_new_reg[common_cols], join='inner')
        
        correlations = []
        exact_matches = 0
        
        for col in common_cols:
            s_old = df_old_align[col]
            s_new = df_new_align[col]
            
            # Vérifier si les séries sont strictement identiques
            if s_old.equals(s_new):
                exact_matches += 1
            else:
                # Calcul de la corrélation si elles diffèrent
                if s_old.std() > 0 and s_new.std() > 0:
                    corr = s_old.corr(s_new)
                    correlations.append(corr)
                    
        print("\n🔍 Analyse des valeurs sur la période commune :")
        print(f"  -> Variables strictement identiques : {exact_matches} sur {len(common_cols)}")
        if correlations:
            print(f"  -> Corrélation moyenne sur les variables divergentes : {np.mean(correlations):.4f}")
            print(f"  -> Corrélation minimale observée : {np.min(correlations):.4f}")
            if np.mean(correlations) < 0.95:
                print("  ⚠️ ATTENTION : Les séries temporelles ont subi des modifications mathématiques significatives (nouvelle méthodologie d'agrégation GDELT, lissage, ou correction de requêtes).")