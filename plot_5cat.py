import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator

# Configuration esthétique haut de gamme (Quantitative Research Style)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['figure.facecolor'] = '#f4f5f7'
plt.rcParams['axes.facecolor'] = '#ffffff'
sns.set_context("notebook", font_scale=0.9)

def plot_every_category_dashboard(csv_path, output_dir="./gdelt_dashboards"):
    """
    Génère un tableau de bord complet à 5 subplots pour chacune des 8 catégories
    afin de comparer visuellement l'impact mathématique de chaque méthode.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Fichier source {csv_path} introuvable.")
        return

    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chargement et tri des données par l'index temporel
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Liste propre de tes 8 sous-indicateurs macro
    categories = [
        'C1_Capital_Financing', 'C2_Firm_Performance', 'C3_Tech_Infrastructure',
        'C4_Innovation_Systems', 'C5_Policy_Regulation', 'C6_Tech_Diffusion',
        'C7_Human_Capital', 'C8_Tech_Design'
    ]

    # Définition des 5 méthodes de ton papier avec leur configuration graphique
    methods_config = [
        {'suffix': '_VolumeIdx', 'title': '1. Volume-Based Index (Baseline Attention)', 'color': '#1f77b4', 'type': 'line'},
        {'suffix': '_BinarySentIdx', 'title': '2. Binary Sentiment Index (Polarity [-1, 1])', 'color': '#2ca02c', 'type': 'fill'},
        {'suffix': '_ContinuousSentIdx', 'title': '3. Continuous Sentiment Index (Raw Tone)', 'color': '#9467bd', 'type': 'line'},
        {'suffix': '_AdjVolumeIdx', 'title': '4. Sentiment-Adjusted Volume Index (Hybrid)', 'color': '#ff7f0e', 'type': 'fill'},
        {'suffix': '_AsymIndex', 'title': '5. Asymmetric Sentiment Index (Loss Aversion Mode)', 'color': '#d62728', 'type': 'line'}
    ]

    print(f"🎨 Début de la génération des graphiques dans : {output_dir}")

    # 2. Boucle principale sur les 8 catégories
    for cat in categories:
        print(f"📊 Génération du dashboard complet pour : {cat}...")
        
        # Vérification de sécurité : on s'assure qu'au moins une colonne de la catégorie existe
        if not any(f"{cat}{m['suffix']}" in df.columns for m in methods_config):
            print(f"   ⏭ Skippé : Aucune colonne trouvée pour {cat}")
            continue

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 15), sharex=True)
        fig.suptitle(f"Analyse Comparative des Méthodes Mathématiques\nThématique : {cat.replace('_', ' ')}", 
                     fontsize=18, fontweight='bold', color='#14233c', y=0.97)

        # 3. Sous-boucle sur les 5 méthodes pour construire les subplots verticaux
        for i, idx_type in enumerate(methods_config):
            ax = axes[i]
            col_name = f"{cat}{idx_type['suffix']}"
            
            if col_name not in df.columns:
                ax.text(0.5, 0.5, f"Colonne {col_name} manquante", transform=ax.transAxes, ha='center')
                continue
                
            series_data = df[col_name]
            color = idx_type['color']

            # Type de tracé ajusté à la nature de la formule
            if idx_type['type'] == 'fill':
                ax.fill_between(series_data.index, series_data, 0, where=(series_data >= 0), color=color, alpha=0.3)
                ax.fill_between(series_data.index, series_data, 0, where=(series_data < 0), color='#d62728', alpha=0.3)
                ax.plot(series_data.index, series_data, color=color, linewidth=1.5)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            else:
                ax.plot(series_data.index, series_data, color=color, linewidth=2)
                if idx_type['suffix'] in ['_ContinuousSentIdx', '_AsymIndex']:
                    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

            # Titre et axes de chaque subplot
            ax.set_title(idx_type['title'], fontsize=12, loc='left', fontweight='semibold', color='#333333')
            ax.set_ylabel("Score", fontsize=10, fontweight='bold')
            
            # Formatage de l'arrière-plan des subplots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            ax.grid(True, linestyle=':', alpha=0.5, color='#dddddd')

        # 4. Formatage global de l'axe temporel partagé (X)
        plt.gca().xaxis.set_major_locator(MonthLocator(interval=1))  # Un repère par mois
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
        fig.autofmt_xdate(rotation=30, ha='right')

        # Ajustement des espaces entre les panneaux pour éviter les chevauchements de textes
        plt.subplots_adjust(hspace=0.35, top=0.91, bottom=0.06)

        # Sauvegarde du fichier individuel par catégorie
        file_out = os.path.join(output_dir, f"dashboard_{cat}.png")
        plt.savefig(file_out, dpi=200, bbox_inches='tight')
        plt.close(fig) # Libère la mémoire graphique immédiatement

    print(f"🏆 Opération terminée ! Les 8 dashboards de tes indicateurs ont été sauvegardés.")

if __name__ == "__main__":
    # Pointage direct vers la matrice globale générée par l'étape précédente
    plot_every_category_dashboard(csv_path="gdelt_8_categories_alpha_matrix.csv")