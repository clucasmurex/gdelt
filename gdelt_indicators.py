# fileName: produce_indicators.py
import os
import time
import pandas as pd
import duckdb

def compute_macro_indicators(parquet_dir, output_csv='macro_financial_signals.csv'):
    """
    Exécute des requêtes colonnaires de haute performance via DuckDB sur l'ensemble
    des fichiers Parquet pour extraire les indices Tech, Credit, Cyber et Supply Chain.
    """
    start_time = time.time()
    
    # Validation du dossier source
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {parquet_dir}")
        
    print("✨ --- INITIALISATION DE LA MACHINE DE CALCUL DUCKDB ---")
    print(f"📂 Dossier source des Parquets : {os.path.abspath(parquet_dir)}")
    print(f"💾 Fichier de sortie : {output_csv}\n")
    
    # Connexion à une instance DuckDB éphémère en mémoire
    con = duckdb.connect()
    
    # Chemin générique pour scanner tous les fichiers Parquet du dossier
    parquet_pattern = os.path.join(parquet_dir, "*.parquet")
    
    # --- STRATÉGIE DE REQUÊTE SQL COLONNAIRE GÉANTE ---
    # Nous utilisons REGEXP_MATCHES ou LIKE pour scanner à haute vitesse les thèmes imbriqués.
    # Pour le sentiment (Net Sentiment), nous calculons une MOYENNE PONDÉRÉE par le WordCount.
    
    sql_query = f"""
    WITH raw_data AS (
        SELECT 
            -- Extraction propre de la date au niveau du jour (YYYY-MM-DD)
            strptime(substring(cast(DATE as string), 1, 8), '%Y%m%d') AS date_day,
            WordCount,
            Tone,
            Enhanced_Themes,
            Organizations,
            Locations
        FROM '{parquet_pattern}'
        WHERE DATE IS NOT NULL
    ),
    daily_base AS (
        -- Calcul du dénominateur global par jour pour l'Attention Intensity
        SELECT 
            date_day,
            COUNT(*) AS total_docs_day,
            SUM(WordCount) AS total_words_day
        FROM raw_data
        GROUP BY date_day
    ),
    thematic_flags AS (
        SELECT 
            date_day,
            WordCount,
            Tone,
            
            -- 1. TECH SUB-INDICES (Utilisation de Enhanced_Themes)
            CASE WHEN Enhanced_Themes LIKE '%WB_667_ICT_INFRASTRUCTURE%' 
                   OR Enhanced_Themes LIKE '%WB_669_SOFTWARE_INFRASTRUCTURE%' 
                   OR Enhanced_Themes LIKE '%WB_1228_ICT_VALUE_CHAIN%' 
                   OR Enhanced_Themes LIKE '%WB_692_INFORMATION_TECHNOLOGY_PARKS_AND_ZONES%' THEN 1 ELSE 0 END AS is_tech_capex,
                   
            CASE WHEN Enhanced_Themes LIKE '%WB_380_FUNDING_INNOVATION%' 
                   OR Enhanced_Themes LIKE '%WB_2649_SUPPORT_TO_TECHNOLOGY_ENTREPRENEURS%' 
                   OR Enhanced_Themes LIKE '%WB_376_INNOVATION_TECHNOLOGY_AND_ENTREPRENEURSHIP%' 
                   OR Enhanced_Themes LIKE '%WB_1275_INNOVATION_COLLABORATION%' THEN 1 ELSE 0 END AS is_tech_funding,
                   
            CASE WHEN Enhanced_Themes LIKE '%WB_2381_SOFTWARE_DEVELOPMENT%' 
                   OR Enhanced_Themes LIKE '%WB_1084_TECHNOLOGY_TRANSFER_AND_DIFFUSION%' 
                   OR Enhanced_Themes LIKE '%WB_1944_INNOVATION_AND_PRODUCTIVITY_GROWTH%' 
                   OR Enhanced_Themes LIKE '%SOC_TECHNOLOGYSECTOR%' THEN 1 ELSE 0 END AS is_tech_adoption,
                   
            CASE WHEN Enhanced_Themes LIKE '%WB_283_ICT_LAW%' 
                   OR Enhanced_Themes LIKE '%WB_279_ICT_STRATEGY_POLICY_AND_REGULATION%' 
                   OR Enhanced_Themes LIKE '%WB_282_ICT_POLICY_REGULATORY_FRAMEWORK_AND_INSTITUTIONS%' THEN 1 ELSE 0 END AS is_tech_reg,

            -- 2. CREDIT INDEX
            CASE WHEN Enhanced_Themes LIKE '%CLAIM_CREDIT%' 
                   OR Enhanced_Themes LIKE '%EPU_POLICY_CREDIT_CRUNCH%' 
                   OR Enhanced_Themes LIKE '%WB_357_CREDIT_REPORTING%' 
                   OR Enhanced_Themes LIKE '%WB_846_INSOLVENCY_AND_DEBTOR_CREDITOR_LAW%' 
                   OR Enhanced_Themes LIKE '%WB_1894_EFFECTIVE_INSOLVENCY_AND_CREDITOR_RIGHTS_SYSTEMS%' 
                   OR Enhanced_Themes LIKE '%ECON_HOUSING_PRICES%' 
                   OR Enhanced_Themes LIKE '%WB_612_HOUSING_FINANCE%' 
                   OR Enhanced_Themes LIKE '%WB_904_HOUSING_MARKETS%' 
                   OR Enhanced_Themes LIKE '%WB_3001_IMPROVING_ACCESS_TO_FINANCE%' 
                   OR Enhanced_Themes LIKE '%WB_1250_ACCESS_TO_FINANCE_FOR_HOUSEHOLDS_AND_INDIVIDUALS%' 
                   OR Enhanced_Themes LIKE '%WB_1188_TRADE_LIQUIDITY%' THEN 1 ELSE 0 END AS is_credit,

            -- 3. CYBERSECURITY INDEX
            CASE WHEN Enhanced_Themes LIKE '%CYBER_ATTACK%' 
                   OR Enhanced_Themes LIKE '%WB_2457_CYBER_CRIME%' 
                   OR Enhanced_Themes LIKE '%TAX_TERROR_GROUP_ISLAMIC_CYBER_RESISTANCE%' 
                   OR Enhanced_Themes LIKE '%TAX_TERROR_GROUP_TUNISIAN_CYBER_ARMY%' 
                   OR Enhanced_Themes LIKE '%WB_670_ICT_SECURITY%' 
                   OR Enhanced_Themes LIKE '%INTERNET_BLACKOUT%' THEN 1 ELSE 0 END AS is_cyber,

            -- 4. SUPPLY CHAINS INDEX
            CASE WHEN Enhanced_Themes LIKE '%WB_1182_CROSS_BORDER_SUPPLY%' 
                   OR Enhanced_Themes LIKE '%WB_2605_SUPPLY_CHAIN_ANALYSIS%' 
                   OR Enhanced_Themes LIKE '%MARITIME%' 
                   OR Enhanced_Themes LIKE '%MARITIME_INCIDENT%' 
                   OR Enhanced_Themes LIKE '%MARITIME_PIRACY%' 
                   OR Enhanced_Themes LIKE '%WB_2460_MARITIME_PIRACY%' 
                   OR Enhanced_Themes LIKE '%TAX_TERROR_GROUP_RED_SEA_AFAR_DEMOCRATIC_ORGANIZATION%' 
                   OR Enhanced_Themes LIKE '%WB_865_TRADE_CORRIDORS%' 
                   OR Enhanced_Themes LIKE '%WB_2584_CORRIDOR_MANAGEMENT%' 
                   OR Enhanced_Themes LIKE '%WB_2973_MAIN_CANALS%' THEN 1 ELSE 0 END AS is_supply
        FROM raw_data
    )
    SELECT 
        b.date_day AS date,
        b.total_docs_day,
        
        -- --- CALCULS DES INTENSITÉS (Normalisées en % du volume d'articles du jour) ---
        ROUND((COUNT(CASE WHEN t.is_tech_capex = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS tech_capex_intensity,
        ROUND((COUNT(CASE WHEN t.is_tech_funding = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS tech_funding_intensity,
        ROUND((COUNT(CASE WHEN t.is_tech_adoption = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS tech_adoption_intensity,
        ROUND((COUNT(CASE WHEN t.is_tech_reg = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS tech_reg_intensity,
        ROUND((COUNT(CASE WHEN t.is_credit = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS credit_intensity,
        ROUND((COUNT(CASE WHEN t.is_cyber = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS cyber_intensity,
        ROUND((COUNT(CASE WHEN t.is_supply = 1 THEN 1 END) * 100.0) / b.total_docs_day, 5) AS supply_intensity,

        -- --- CALCULS DES SENTIMENTS NETS (Moyenne du Tone pondérée par le WordCount de chaque article) ---
        ROUND(SUM(CASE WHEN t.is_tech_capex = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_tech_capex = 1 THEN t.WordCount END), 1), 4) AS tech_capex_sentiment,
        ROUND(SUM(CASE WHEN t.is_tech_funding = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_tech_funding = 1 THEN t.WordCount END), 1), 4) AS tech_funding_sentiment,
        ROUND(SUM(CASE WHEN t.is_tech_adoption = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_tech_adoption = 1 THEN t.WordCount END), 1), 4) AS tech_adoption_sentiment,
        ROUND(SUM(CASE WHEN t.is_tech_reg = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_tech_reg = 1 THEN t.WordCount END), 1), 4) AS tech_reg_sentiment,
        ROUND(SUM(CASE WHEN t.is_credit = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_credit = 1 THEN t.WordCount END), 1), 4) AS credit_sentiment,
        ROUND(SUM(CASE WHEN t.is_cyber = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_cyber = 1 THEN t.WordCount END), 1), 4) AS cyber_sentiment,
        ROUND(SUM(CASE WHEN t.is_supply = 1 THEN t.Tone * t.WordCount END) / COALESCE(SUM(CASE WHEN t.is_supply = 1 THEN t.WordCount END), 1), 4) AS supply_sentiment

    FROM daily_base b
    JOIN thematic_flags t ON b.date_day = t.date_day
    GROUP BY b.date_day, b.total_docs_day
    ORDER BY b.date_day ASC;
    """
    
    print("🏃 Analyse en cours sur les 2 mois de données...")
    # Exécution de la requête et conversion directe en DataFrame Pandas
    final_df = con.execute(sql_query).df()
    
    # Sauvegarde sur le disque
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    elapsed_time = time.time() - start_time
    print(f"\n🟢 --- CALCUL TERMINÉ EN {elapsed_time:.2f} SECONDES ---")
    print(f"📊 Nombre de jours analysés : {len(final_df)}")
    print(f"💾 Fichier d'indicateurs généré avec succès : {output_csv}")
    print(final_df.head(10))

if __name__ == "__main__":
    # Test local sur tes 2 mois de Parquet présents dans ton dossier projet
    compute_macro_indicators(
        parquet_dir='./gdelt_parquet_db', 
        output_csv='macro_financial_signals.csv'
    )