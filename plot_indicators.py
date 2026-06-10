# fileName: plot_indicators.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_macro_signals(csv_path='macro_financial_signals.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}. Lance d'abord le calcul des indicateurs.")

    # 1. Chargement et préparation des données
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Configuration du style graphique
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("🤖 GDELT Macro-Financial Signals — AI & Tech Sector (Test 2 Mois)", fontsize=16, fontweight='bold', y=0.95)

    # Configuration des 4 sous-indices à tracer
    sub_indices = [
        {
            'name': '1. CAPEX & Infrastructure (Upstream)',
            'intensity_col': 'tech_capex_intensity',
            'sentiment_col': 'tech_capex_sentiment',
            'color': '#1f77b4' # Bleu
        },
        {
            'name': '2. Funding & Ecosystem (Liquidity/Duration)',
            'intensity_col': 'tech_funding_intensity',
            'sentiment_col': 'tech_funding_sentiment',
            'color': '#2ca02c' # Vert
        },
        {
            'name': '3. Adoption & R&D (Coincident Momentum)',
            'intensity_col': 'tech_adoption_intensity',
            'sentiment_col': 'tech_adoption_sentiment',
            'color': '#ff7f0e' # Orange
        },
        {
            'name': '4. Regulation & Policy Risk (Downside Risk)',
            'intensity_col': 'tech_reg_intensity',
            'sentiment_col': 'tech_reg_sentiment',
            'color': '#d62728' # Rouge
        }
    ]

    # 2. Construction de la boucle graphique multi-panneaux
    for idx, index_data in enumerate(sub_indices):
        ax1 = axes[idx]
        
        # Axe 1 : Attention Intensity (Histogramme / Barres)
        ax1.bar(df['date'], df[index_data['intensity_col']], color=index_data['color'], alpha=0.6, width=0.6, label='Attention Intensity (%)')
        ax1.set_ylabel("Attention Intensity (%)", color=index_data['color'], fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=index_data['color'])
        ax1.set_title(index_data['name'], fontsize=12, fontweight='bold', loc='left')
        
        # Axe 2 : Dual Axis pour le Net Sentiment (Courbe)
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df[index_data['sentiment_col']], color='#222222', linewidth=1.8, linestyle='-', marker='o', markersize=4, label='Net Sentiment (Weighted)')
        ax2.set_ylabel("Net Sentiment (Tone)", color='#222222', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#222222')
        
        # Ligne de neutralité du sentiment (y=0)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Ajustement des légendes combinées
        if idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    # 3. Optimisation des axes temporels
    plt.gcf().autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.95, 0.92])
    
    # Sauvegarde de la figure
    output_png = 'tech_macro_indicators_plot.png'
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"🟢 Graphique de diagnostic généré avec succès : {output_png}")

if __name__ == "__main__":
    plot_macro_signals()