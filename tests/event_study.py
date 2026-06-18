import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def run_advanced_event_study(gdelt_csv, ticker="^VIX", gdelt_col="C5_Policy_Regulation_AsymIndex", 
                             sent_col="C5_Policy_Regulation_ContinuousSentIdx",
                             z_threshold=1.5, mode="panic", window_before=5, window_after=20):
    """
    Réalise une étude d'événement filtrée selon le régime de sentiment ou l'amplitude.
    Modes disponibles :
      - 'panic'      : Choc de volume ET sentiment négatif (Directionnel Baissier/Haussier VIX)
      - 'relief'     : Choc de volume ET sentiment positif
      - 'volatility' : Analyse de l'amplitude absolue des mouvements (Non-directionnel)
    """
    if not os.path.exists(gdelt_csv):
        print(f"❌ Fichier {gdelt_csv} introuvable.")
        return

    # 1. Chargement et calcul du Z-Score de volume
    gdelt_df = pd.read_csv(gdelt_csv, index_col=0, parse_dates=True).sort_index()
    series_raw = gdelt_df[gdelt_col]
    rolling_mean = series_raw.rolling(window=30, min_periods=10).mean()
    rolling_std = series_raw.rolling(window=30, min_periods=10).std()
    gdelt_df['Z_Score'] = (series_raw - rolling_mean) / (rolling_std + 1e-8)

    # 2. Application du filtre conditionnel (La clé de l'Alpha)
    if mode == "panic":
        # Gros volume d'actualité ET moral en baisse
        event_mask = (gdelt_df['Z_Score'] > z_threshold) & (gdelt_df[sent_col] < 0)
        title_suffix = "Régime de PANIC (Volume Haut & Sentiment Bas)"
    elif mode == "relief":
        # Gros volume d'actualité ET moral en hausse
        event_mask = (gdelt_df['Z_Score'] > z_threshold) & (gdelt_df[sent_col] > 0)
        title_suffix = "Régime de SOULAGEMENT (Volume Haut & Sentiment Haut)"
    else:
        # Mode Volatilité : on prend tous les chocs de volume sans distinction
        event_mask = gdelt_df['Z_Score'] > z_threshold
        title_suffix = "Analyse de l'AMPLITUDE ABSOLUE (Effet Straddle)"

    event_dates = gdelt_df.index[event_mask]
    
    # Élimination des événements trop proches (De-clumping)
    cleaned_event_dates = []
    last_date = pd.Timestamp("1900-01-01")
    for d in event_dates:
        if (d.to_pydatetime() - last_date.to_pydatetime()).days > window_after:
            cleaned_event_dates.append(d)
            last_date = d
            
    print(fr"🚨 Mode [{mode.upper()}] : {len(cleaned_event_dates)} événements retenus (Z > {z_threshold}\sigma).")
    if len(cleaned_event_dates) == 0:
        return

    # 3. Données de marché et calcul des rendements
    start_date = (gdelt_df.index.min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (gdelt_df.index.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    market_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    close_col = 'Adj Close' if 'Adj Close' in market_raw.columns.get_level_values(0) else 'Close'
    market_close = market_raw[close_col]
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]

    if ticker == '^VIX':
        market_diffs = market_close.diff()
    else:
        market_diffs = market_close.pct_change() * 100

    # 4. Extraction et alignement des trajectoires
    event_curves = []
    timeline = list(range(-window_before, window_after + 1))
    trading_days = market_diffs.index.tolist()
    
    for event_date in cleaned_event_dates:
        if event_date in trading_days:
            t0_idx = trading_days.index(event_date)
            if t0_idx - window_before >= 0 and t0_idx + window_after < len(trading_days):
                window_data = market_diffs.iloc[t0_idx - window_before : t0_idx + window_after + 1].values
                
                # C'est ici qu'on applique la transformation selon le mode choisi :
                if mode == "volatility":
                    # Si on cherche la volatilité, on cumule la VALEUR ABSOLUE du mouvement
                    cum_perf = np.cumsum(np.abs(window_data)) - np.cumsum(np.abs(window_data))[window_before - 1]
                else:
                    # Mode directionnel classique (Panique ou Soulagement)
                    cum_perf = np.cumsum(window_data) - np.cumsum(window_data)[window_before - 1]
                    
                event_curves.append(cum_perf)

    # 5. Calcul des statistiques et génération du graphique
    matrice_evenements = np.array(event_curves)
    trajectoire_moyenne = np.mean(matrice_evenements, axis=0)
    erreur_type = np.std(matrice_evenements, axis=0) / np.sqrt(len(event_curves))

    plt.figure(figsize=(13, 6.5))
    for curve in event_curves:
        plt.plot(timeline, curve, color='grey', alpha=0.15, linewidth=1)
        
    plt.plot(timeline, trajectoire_moyenne, color='#d62728' if mode=='panic' else '#2ca02c', linewidth=3, label="Trajectoire Moyenne")
    plt.fill_between(timeline, trajectoire_moyenne - 1.96 * erreur_type, trajectoire_moyenne + 1.96 * erreur_type, 
                     color='red' if mode=='panic' else 'green', alpha=0.12, label="Intervalle de confiance à 95%")
    
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label="Choc GDELT (T=0)")
    plt.axhline(0, color='grey', linestyle='-', linewidth=0.8)
    
    plt.title(f"Étude d'Événement Pro ({ticker})\nDéclencheur : {gdelt_col} | {title_suffix}", fontsize=13, fontweight='bold')
    plt.xlabel("Jours de Trading autour de l'événement", fontsize=10)
    plt.ylabel("Variation Cumulée" + (" Absolue" if mode=="volatility" else ""), fontsize=10)
    plt.legend(loc='upper left', frameon=True, facecolor='#ffffff')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_filename = f"event_study_pro_{ticker}_{mode}.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"🟢 Graphique enregistré : {output_filename}")

if __name__ == "__main__":
    csv_path = "gdelt_8_categories_alpha_matrix.csv"
    
    # --- TEST COMPORTEMENTAL 1 : Isoler uniquement la VRAIE Panique Réglementaire ---
    run_advanced_event_study(csv_path, ticker="^VIX", mode="panic", z_threshold=1.5)
    
    # --- TEST COMPORTEMENTAL 2 : Analyser l'effet d'amplitude absolue sur la Tech (ex: QQQ) ---
    run_advanced_event_study(csv_path, ticker="QQQ", mode="volatility", z_threshold=1.5)