import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_lead_lag(csv_path, ticker="SMH", gdelt_col="C3_Tech_Infrastructure_VolumeIdx", max_lag=15):
    """
    Analyse la corrélation cross-temporelle entre un indicateur GDELT lissé
    et les rendements futurs d'un actif (Recherche de lead/lag).
    """
    # 1. Chargement et lissage des données GDELT (Moyenne mobile 7 jours pour retirer le bruit)
    gdelt = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    gdelt_smoothed = gdelt[gdelt_col].rolling(window=7, min_periods=3).mean()
    
    # 2. Téléchargement et calcul des rendements financiers
    start, end = gdelt.index.min().strftime('%Y-%m-%d'), (gdelt.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    market = yf.download(ticker, start=start, end=end, progress=False)
    
    # Gestion des prix selon la version de yfinance
    # NOUVEAU CODE (Aplatit les dimensions yfinance automatiquement)
    close_col = 'Adj Close' if 'Adj Close' in market.columns.get_level_values(0) else 'Close'

    # On extrait la colonne de prix de clôture
    market_close = market[close_col]

    # Si yfinance a renvoyé un DataFrame (MultiIndex), on extrait la colonne du ticker
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]

    # Calcul du rendement quotidien à 1 dimension
    market_returns = market_close.pct_change() * 100
    market_returns.name = 'Market'

    # Alignement initial via une jointure explicite pour éviter tout conflit d'index
    df_clean = pd.concat([gdelt_smoothed.rename('GDELT'), market_returns], axis=1).dropna()
    
    # 3. Calcul des corrélations pour différents lags (Décalages temporels)
    lags = range(0, max_lag + 1)
    correlations = []
    
    for lag in lags:
        # On décale le marché dans le FUTUR par rapport à GDELT (Market_t+lag vs GDELT_t)
        shifted_market = df_clean['Market'].shift(-lag)
        corr = df_clean['GDELT'].corr(shifted_market)
        correlations.append(corr)
        
    # 4. Visualisation de la courbe de cross-corrélation
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlations, marker='o', linewidth=2, color='#d62728')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title(f"Cross-Corrélation Temporelle : {gdelt_col} vs {ticker}", fontsize=14, fontweight='bold')
    plt.xlabel("Lag (Nombre de jours de décalage vers le futur t + n)", fontsize=11)
    plt.ylabel("Coefficient de Corrélation (r)", fontsize=11)
   # --- NOUVEAU CODE (Sauvegarde automatique en fichier) ---
    plt.xticks(lags)
    plt.grid(True, linestyle=':')
    
    output_img = f"cross_corr_{ticker}_vs_{gdelt_col[:10]}.png"
    plt.savefig(output_img, dpi=200, bbox_inches='tight')
    plt.close()
    
    best_lag = np.argmax(np.abs(correlations))
    print(f"🟢 Graphique sauvegardé avec succès sous : {output_img}")
    print(f"💡 Corrélation maximale trouvée à un lag de {best_lag} jours (r = {correlations[best_lag]:.2f})")
    

if __name__ == "__main__":
    # Testons sur les Semi-conducteurs (C3) vs SMH
    analyze_lead_lag("gdelt_8_categories_alpha_matrix.csv", ticker="SMH", gdelt_col="C3_Tech_Infrastructure_VolumeIdx", max_lag=10)