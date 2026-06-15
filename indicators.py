import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def generate_gdelt_indices(parquet_dir, start_date=None, end_date=None, alpha=1.0, beta=1.5):
    """
    Calcule la matrice complète des indices d'attention et de sentiment GDELT GKG
    pour les 8 catégories de la taxonomie spécifiée, selon les spécifications
    méthodologiques de l'utilisateur (Article-level approach).
    
    Parameters:
    -----------
    parquet_dir : str ou Path
        Dossier contenant les fichiers 'gdelt_YYYY-MM.parquet'.
    start_date : str, optionnel
        Date de début au format 'YYYY-MM-DD'.
    end_date : str, optionnel
        Date de fin au format 'YYYY-MM-DD'.
    alpha : float
        Poids appliqué aux articles positifs dans l'AsymIndex (s_j = +1).
    beta : float
        Poids appliqué aux articles négatifs dans l'AsymIndex (s_j = -1). Décoché si beta > alpha.
    """
    
    # ---------------------------------------------------------
    # DÉFINITION DE LA TAXONOMIE DES 8 CATÉGORIES
    # ---------------------------------------------------------
    categories = {
        'C1_Capital_Financing': [
            'WB_380_FUNDING_INNOVATION', 'WB_376_INNOVATION_TECHNOLOGY_AND_ENTREPRENEURSHIP',
            'WB_2649_SUPPORT_TO_TECHNOLOGY_ENTREPRENEURS', 'WB_2424_ICT_AND_FINANCIAL_SECTOR'
        ],
        'C2_Firm_Performance': [
            'WB_1944_INNOVATION_AND_PRODUCTIVITY_GROWTH', 'WB_377_FIRM_INNOVATION_PRODUCTIVITY_AND_GROWTH',
            'WB_1275_INNOVATION_COLLABORATION'
        ],
        'C3_Tech_Infrastructure': [
            'WB_667_ICT_INFRASTRUCTURE', 'WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES',
            'WB_669_SOFTWARE_INFRASTRUCTURE', 'WB_2381_SOFTWARE_DEVELOPMENT',
            'WB_1228_ICT_VALUE_CHAIN', 'WB_692_INFORMATION_TECHNOLOGY_PARKS_AND_ZONES',
            'WB_691_TECHNOLOGY_AND_SCIENCE_PARKS'
        ],
        'C4_Innovation_Systems': [
            'WB_689_SCIENCE_TECHNOLOGY_AND_INNOVATION', 'WB_2399_ICT_INNOVATION_AND_TRANSFORMATION',
            'WB_2401_ICT_INNOVATION_METHODOLOGIES', 'WB_652_ICT_APPLICATIONS',
            'WB_281_ICT_INDUSTRY_AND_SERVICES'
        ],
        'C5_Policy_Regulation': [
            'WB_378_INNOVATION_AND_TECHNOLOGY_POLICY', 'WB_2350_ICT_INNOVATION_POLICY',
            'WB_279_ICT_STRATEGY_POLICY_AND_REGULATION', 'WB_282_ICT_POLICY_REGULATORY_FRAMEWORK_AND_INSTITUTIONS',
            'WB_283_ICT_LAW', 'WB_2344_ICT_STRATEGY'
        ],
        'C6_Tech_Diffusion': [
            'WB_1084_TECHNOLOGY_TRANSFER_AND_DIFFUSION', 'WB_1274_TECHNOLOGY_TRANSFER_OFFICES',
            'WB_394_TECHNOLOGY_EXTENSION_SERVICES'
        ],
        'C7_Human_Capital': [
            'TAX_FNCACT_TECHNOLOGIST', 'SOC_TECHNOLOGYSECTOR'
        ],
        'C8_Tech_Design': [
            'WB_2377_TECHNOLOGY_ARCHITECTURE'
        ]
    }

    parquet_path = Path(parquet_dir)
    all_parquet_files = sorted(glob.glob(str(parquet_path / "gdelt_*.parquet")))
    
    if not all_parquet_files:
        print(f"❌ Aucun fichier Parquet trouvé dans {parquet_dir}")
        return None

    # --- ÉTAPE 1 : FILTRAGE SMART DES FICHIERS PARQUET ---
    t_start = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
    t_end = pd.to_datetime(end_date) if end_date else pd.Timestamp.max
    
    target_files = []
    for f in all_parquet_files:
        try:
            file_name = os.path.basename(f)
            year_month = file_name.split('_')[1].split('.')[0]
            file_start = pd.to_datetime(f"{year_month}-01")
            file_end = file_start + pd.offsets.MonthEnd(1)
            
            if (file_start <= t_end) and (file_end >= t_start):
                target_files.append(f)
        except Exception:
            target_files.append(f)

    if not target_files:
        print("⚠ Aucun fichier ne correspond à la période demandée.")
        return None

    print(f"📅 Fenêtre d'analyse : {start_date if start_date else 'Full'} à {end_date if end_date else 'Full'}")
    print(f"📦 Traitement sélectif de {len(target_files)} fichiers Parquet...")

    daily_containers = []

    # --- ÉTAPE 2 : PIPELINE DE CALCUL VECTORISÉ MONTH-BY-MONTH ---
    for file_idx, file_path in enumerate(target_files, 1):
        print(f"   Reading parquet [{file_idx}/{len(target_files)}]: {os.path.basename(file_path)}")
        df = pd.read_parquet(file_path, columns=['DATE', 'Enhanced_Themes', 'Tone'])
        
        # Formating de l'index de date
        df['Date_Day'] = pd.to_datetime(df['DATE'].str[:8], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date_Day'])
        
        # Filtrage fin au jour le jour
        if start_date or end_date:
            df = df[(df['Date_Day'] >= t_start) & (df['Date_Day'] <= t_end)]
            if df.empty:
                continue

        df['Enhanced_Themes'] = df['Enhanced_Themes'].fillna('')
        
        # Dénominateur quotidien : TotalNews_t
        total_news_t = df.groupby('Date_Day').size().rename('TotalNews')
        
        # Pré-calcul des variables de sentiment au niveau de l'article j (Section 2 & 5)
        # s_j = +1 si Tone > 0, -1 si Tone < 0, et 0 si Tone == 0 (Correction Neutre)
        df['s_j'] = np.select(
            [df['Tone'] > 0, df['Tone'] < 0], 
            [1, -1], 
            default=0
        ).astype(np.int8)
        
        # Variable asymétrique par article : alpha * 1(s_j = +1) - beta * 1(s_j = -1)
        df['asym_j'] = np.select(
            [df['s_j'] == 1, df['s_j'] == -1],
            [alpha, -beta],
            default=0.0
        ).astype(np.float32)

        # Conteneur de métriques du mois en cours
        month_df = pd.DataFrame(index=total_news_t.index)
        month_df['TotalNews'] = total_news_t

        # Calcul vectorisé par bloc pour chaque catégorie c
        for cat_name, themes_list in categories.items():
            regex_pattern = '|'.join(themes_list)
            # w_cj variable binaire de présence (Article-Level)
            df['w_cj'] = df['Enhanced_Themes'].str.contains(regex_pattern, regex=True, na=False).astype(np.int8)
            
            # Isolement des articles contenant la catégorie
            df_cat = df[df['w_cj'] == 1]
            
            if df_cat.empty:
                month_df[f'{cat_name}_N'] = 0
                month_df[f'{cat_name}_Sum_sj'] = 0.0
                month_df[f'{cat_name}_Sum_Tone'] = 0.0
                month_df[f'{cat_name}_Sum_Asym'] = 0.0
                continue
                
            # Calculs agrégés quotidiens (N_ct, Sum(s_j), Sum(Tone), Sum(Asym))
            agg_results = df_cat.groupby('Date_Day').agg(
                N_ct=('w_cj', 'count'),
                Sum_sj=('s_j', 'sum'),
                Sum_Tone=('Tone', 'sum'),
                Sum_Asym=('asym_j', 'sum')
            )
            
            # Merge dans la structure mensuelle
            month_df = month_df.join(agg_results)
            month_df[f'{cat_name}_N'] = month_df['N_ct'].fillna(0).astype(np.int32)
            month_df[f'{cat_name}_Sum_sj'] = month_df['Sum_sj'].fillna(0.0)
            month_df[f'{cat_name}_Sum_Tone'] = month_df['Sum_Tone'].fillna(0.0)
            month_df[f'{cat_name}_Sum_Asym'] = month_df['Sum_Asym'].fillna(0.0)
            month_df.drop(columns=['N_ct', 'Sum_sj', 'Sum_Tone', 'Sum_Asym'], inplace=True, errors='ignore')

        daily_containers.append(month_df)
        del df, month_df

    # --- ÉTAPE 3 : CONSOLIDATION TEMPORELLE GLOBALE ET APPLICATION DES FORMULES MATHÉMATIQUES ---
    print("\n🏁 Compilation finale de la série temporelle...")
    master_raw = pd.concat(daily_containers).sort_index()
    master_raw = master_raw.groupby(master_raw.index).sum() # Somme en cas de recouvrement
    
    output_indices = pd.DataFrame(index=master_raw.index)
    
    print("📈 Application des formules de ton modèle de recherche...")
    for cat_name in categories.keys():
        N = master_raw[f'{cat_name}_N']
        TotalNews = master_raw['TotalNews']
        
        # 1. Volume-Based Index (Baseline Section 1.a)
        output_indices[f'{cat_name}_VolumeIdx'] = (N / TotalNews).fillna(0.0)
        
        # 2. Binary Sentiment Index (Section 2)
        output_indices[f'{cat_name}_BinarySentIdx'] = (master_raw[f'{cat_name}_Sum_sj'] / N).fillna(0.0)
        
        # 3. Continuous Sentiment Index (Section 3)
        output_indices[f'{cat_name}_ContinuousSentIdx'] = (master_raw[f'{cat_name}_Sum_Tone'] / N).fillna(0.0)
        
        # 4. Sentiment-Adjusted Volume Index (Section 4)
        output_indices[f'{cat_name}_AdjVolumeIdx'] = (master_raw[f'{cat_name}_Sum_sj'] / TotalNews).fillna(0.0)
        
        # 5. Asymmetric Sentiment Index (Section 5)
        output_indices[f'{cat_name}_AsymIndex'] = (master_raw[f'{cat_name}_Sum_Asym'] / TotalNews).fillna(0.0)

    return output_indices

if __name__ == "__main__":
    # Paramètres de calcul
    source_db = './gdelt_parquet_db'
    
    # Exécution sur une plage temporelle au choix (Ex: Rotation Fin 2022 / Début 2023)
    final_alpha_matrix = generate_gdelt_indices(
        parquet_dir=source_db,
        start_date='2015-02-01',
        end_date='2026-06-01',
        alpha=1.0,   # Poids positif
        beta=2.0     # Poids négatif accentué (Aversion aux pertes)
    )
    
    if final_alpha_matrix is not None:
        final_alpha_matrix.to_csv("gdelt_8_categories_alpha_matrix.csv")
        print(f"🏆 Matrice d'indicateurs générée ! Forme : {final_alpha_matrix.shape}")
        # Visualisation rapide des colonnes générées pour C1 (Capital & Financing)
        print(final_alpha_matrix[[c for c in final_alpha_matrix.columns if 'C1' in c]].tail())