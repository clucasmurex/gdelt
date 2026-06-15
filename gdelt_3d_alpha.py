import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def build_macro_horizon_cube(gdelt_csv, start_date="2015-01-01", end_date="2026-06-01"):
    """
    Calcule les corrélations entre les indicateurs GDELT à l'instant t
    et les rendements CUMULÉS des actifs sur des horizons macro (5j, 10j, 20j).
    """
    if not os.path.exists(gdelt_csv):
        print(f"❌ Matrice globale {gdelt_csv} introuvable.")
        return

    # 1. Chargement GDELT
    gdelt_df = pd.read_csv(gdelt_csv, index_col=0, parse_dates=True).sort_index().loc[start_date:end_date]

    # 2. Données de marché
    tickers = ['SMH', 'IGV', 'QQQ', 'BOTZ', '^VIX']
    market_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    close_col = 'Adj Close' if 'Adj Close' in market_raw.columns.get_level_values(0) else 'Close'
    market_close = market_raw[close_col]
    
    # 3. Calcul des RENDEMENTS CUMULÉS FUTURS (Horizons de détention / Holding Periods)
    # On cherche à savoir si le newsflow d'aujourd'hui prédit la performance sur les X prochains jours
    market_horizons = pd.DataFrame(index=market_close.index)
    for ticker in tickers:
        if ticker == '^VIX':
            market_horizons[f'{ticker}_Fwd_5j'] = market_close[ticker].shift(-5) # Niveau brut futur
        else:
            # Rendement cumulé sur les 5, 10 et 20 prochains jours de trading
            market_horizons[f'{ticker}_Fwd_5j'] = market_close[ticker].pct_change(5).shift(-5) * 100
            market_horizons[f'{ticker}_Fwd_20j'] = market_close[ticker].pct_change(20).shift(-20) * 100

    # Alignement
    combined_df = gdelt_df.join(market_horizons, how='inner').dropna()
    
    categories = ['C1_Capital_Financing', 'C2_Firm_Performance', 'C3_Tech_Infrastructure', 'C5_Policy_Regulation']
    methods = ['_ContinuousSentIdx', '_AsymIndex']
    horizons = ['_Fwd_5j', '_Fwd_20j']

    # Génération d'une Heatmap ciblée pour voir si les actions s'allument sur des horizons longs
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()
    
    plot_idx = 0
    for method in methods:
        for horizon in horizons:
            corr_matrix = np.zeros((len(tickers), len(categories)))

            for t_idx, ticker in enumerate(tickers):
                for c_idx, cat in enumerate(categories):
                    gdelt_col = f"{cat}{method}"
                    market_col = f"{ticker}{horizon}"
                    
                    # Vérification de sécurité : si la colonne n'existe pas, on met la corrélation à 0 (ou NaN)
                    if gdelt_col in combined_df.columns and market_col in combined_df.columns:
                        x = combined_df[gdelt_col].rolling(14, min_periods=5).mean()
                        y = combined_df[market_col]
                        corr_matrix[t_idx, c_idx] = x.corr(y)
                    else:
                        corr_matrix[t_idx, c_idx] = np.nan # Remplissage neutre si la donnée manque
                        
            ax = axes[plot_idx]
            sns.heatmap(pd.DataFrame(corr_matrix, index=tickers, columns=categories), 
                        ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-0.15, vmax=0.15, center=0.0)
            
            ax.set_title(f"GDELT {method.replace('_','')} vs Rendement Futur {horizon.replace('_Fwd_','')}", fontsize=11, fontweight='bold')
            plot_idx += 1

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig("gdelt_macro_horizons.png", dpi=200, bbox_inches='tight')
    print("🟢 Analyse des horizons macro-temporels sauvegardée sous 'gdelt_macro_horizons.png'")

if __name__ == "__main__":
    build_macro_horizon_cube("gdelt_8_categories_alpha_matrix.csv")