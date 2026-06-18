import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Configuration visuelle style Quantitative Research
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['figure.facecolor'] = '#f4f5f7'
plt.rcParams['axes.facecolor'] = '#ffffff'

def build_market_comparison(csv_path, output_img="gdelt_vs_market_scatters.png"):
    """
    Télécharge les données Yahoo Finance pertinentes et génère des nuages de points
    pour analyser la corrélation avec les indicateurs GDELT.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Fichier de signaux {csv_path} introuvable. Calcule d'abord tes indicateurs.")
        return

    # 1. Chargement de la matrice d'indicateurs GDELT
    gdelt_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    gdelt_df = gdelt_df.sort_index()
    
    start_date = gdelt_df.index.min().strftime('%Y-%m-%d')
    end_date = (gdelt_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📅 Période détectée dans le CSV : {start_date} au {gdelt_df.index.max().strftime('%Y-%m-%d')}")

    # 2. Définition du Mapping de Marché : Quelle catégorie GDELT impacte quel actif ?
    # On choisit la méthode de calcul la plus pertinente de ton papier pour chaque cas
    mapping = {
        'C1_Capital_Financing_ContinuousSentIdx': {
            'ticker': 'IGV',          # ETF Tech Software (sensible aux conditions de financement)
            'label': 'Software ETF (IGV) Return %',
            'metric': 'return',
            'title': 'Financement (Continuous Sent.) vs Rendement IGV'
        },
        'C3_Tech_Infrastructure_VolumeIdx': {
            'ticker': 'SMH',          # ETF Semi-conducteurs (Demande physique amont)
            'label': 'Semiconductor ETF (SMH) Return %',
            'metric': 'return',
            'title': 'Attention Infrastructure (Volume) vs Rendement SMH'
        },
        'C5_Policy_Regulation_AsymIndex': {
            'ticker': '^VIX',         # Indice de la Volatilité (Peur/Incertitude réglementaire)
            'label': 'Volatilité Marché (VIX) Niveau',
            'metric': 'level',
            'title': 'Risque Réglementaire (AsymIndex) vs Niveau du VIX'
        },
        'C7_Human_Capital_AdjVolumeIdx': {
            'ticker': 'NVDA',         # NVIDIA (Guerre des talents / Moteur de l'IA)
            'label': 'NVIDIA (NVDA) Return %',
            'metric': 'return',
            'title': 'Capital Humain (Adj. Volume) vs Rendement NVDA'
        }
    }

    # 3. Téléchargement des données Yahoo Finance
    tickers_to_download = list(set([m['ticker'] for m in mapping.values()]))
    print(f"📥 Téléchargement des cours depuis Yahoo Finance ({tickers_to_download})...")
    market_data = yf.download(tickers_to_download, start=start_date, end=end_date, progress=False)
    
    if market_data.empty:
        print("❌ Échec du téléchargement des données Yahoo Finance.")
        return
        
    # Extraction des cours de clôture ajustés
    # NOUVEAU CODE (Flexible selon la version de yfinance)
    if 'Adj Close' in market_data.columns.levels[0] if isinstance(market_data.columns, pd.MultiIndex) else 'Adj Close' in market_data.columns:
        adj_close = market_data['Adj Close']
    elif 'Close' in market_data.columns.levels[0] if isinstance(market_data.columns, pd.MultiIndex) else 'Close' in market_data.columns:
        print("ℹ 'Adj Close' absent, utilisation de 'Close' (déjà ajusté dans les versions récentes).")
        adj_close = market_data['Close']
    else:
        raise KeyError(f"Impossible de trouver les colonnes de prix de clôture. Colonnes disponibles : {market_data.columns}")

    # 4. Calcul des métriques financières (Rendements log ou Niveaux bruts)
    market_processed = pd.DataFrame(index=adj_close.index)
    for col in adj_close.columns:
        # Pour le VIX, on garde le niveau brut. Pour les actions/ETFs, on calcule le rendement quotidien en %
        if col == '^VIX':
            market_processed[f'{col}_metric'] = adj_close[col]
        else:
            market_processed[f'{col}_metric'] = adj_close[col].pct_change() * 100

    # 5. Alignement temporel des deux bases (Inner Join pour éliminer les week-ends boursiers)
    combined_df = gdelt_df.join(market_processed, how='inner').dropna()
    print(f"🔗 Alignement terminé. Nombre de jours de trading analysés : {len(combined_df)}")

    # 6. Construction de la figure (Grille 2x2 de nuages de points)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle("Analyse de Corrélation : Signaux GDELT vs Réalité des Marchés", fontsize=18, fontweight='bold', color='#14233c', y=0.96)

    for idx, (gdelt_col, config) in enumerate(mapping.items()):
        ax = axes[idx]
        ticker = config['ticker']
        market_col = f'{ticker}_metric'
        
        x_data = combined_df[gdelt_col]
        y_data = combined_df[market_col]
        
        # Calcul du coefficient de corrélation de Pearson pour l'affichage
        correlation = x_data.corr(y_data)

        # Tracé du Scatter Plot avec droite de régression linéaire via Seaborn
        sns.regplot(
            x=x_data, y=y_data, ax=ax,
            scatter_kws={'alpha': 0.5, 'color': '#1f77b4', 's': 40},
            line_kws={'color': '#d62728', 'linewidth': 2, 'label': f'Régression (r = {correlation:.2f})'}
        )
        
        # Formatage esthétique
        ax.set_title(config['title'], fontsize=13, fontweight='bold', pad=10, color='#333333')
        ax.set_xlabel(f"Indicateur GDELT : {gdelt_col.split('_')[-1]}", fontsize=10, fontweight='semibold')
        ax.set_ylabel(config['label'], fontsize=10, fontweight='semibold')
        ax.legend(loc='best', frameon=True, facecolor='#ffffff')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.subplots_adjust(wspace=0.25, hspace=0.28, top=0.88, bottom=0.08)
    
    # Sauvegarde
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🟢 Analyse graphique complétée ! Nuages de points sauvegardés sous : {output_img}")

if __name__ == "__main__":
    # Pointage direct vers le fichier d'indicateurs à 8 catégories
    build_market_comparison(csv_path="gdelt_8_categories_alpha_matrix.csv")