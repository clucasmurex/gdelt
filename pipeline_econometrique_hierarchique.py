import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 🛠️ FONCTIONS UTILITAIRES (Moteur Mathématique et Esthétique)
# =============================================================================

def apply_strict_color_code(val):
    """
    Code couleur professionnel : 
    - Bleu Foncé (+)/Rouge Foncé (-) : Signal ultra-robuste (p < 0.05)
    - Bleu Pâle (+)/Rouge Pâle (-) : Signal faible à surveiller (p < 0.10)
    - Gris : Bruit éliminé ou non significatif
    """
    val_str = str(val)
    if pd.isna(val) or val == "0.00" or val == "N/A":
        return 'color: #cfd8dc; background-color: #fafafa;'
    
    is_strong = "***" in val_str or "**" in val_str
    is_weak = "*" in val_str and not is_strong
    is_positive = "+" in val_str
    is_negative = "-" in val_str
    
    if is_strong and is_positive:
        return 'background-color: #1565c0; color: white; font-weight: bold;'
    elif is_strong and is_negative:
        return 'background-color: #c62828; color: white; font-weight: bold;'
    elif is_weak and is_positive:
        return 'background-color: #bbdefb; color: black;'
    elif is_weak and is_negative:
        return 'background-color: #ffcdd2; color: black;'
    else:
        return 'color: #78909c; background-color: #eceff1;'

def format_coef(coef, p_val):
    if coef == 0: return "0.00"
    stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else " (ns)"
    return f"{coef:+.2f}{stars}"

def extract_abs_coef(val_str):
    """Extrait la valeur absolue d'un coefficient sous forme de string pour le classement"""
    try:
        clean_str = str(val_str).replace('*', '').replace(' (ns)', '').replace('+', '')
        return abs(float(clean_str))
    except:
        return 0.0

def run_lasso_horizon(df_model, features, target):
    """Mini-moteur LASSO + Post-OLS pour un horizon temporel précis"""
    X = df_model[features]
    y = df_model[target]
    
    # 1. Nettoyage et Sélection LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled, y)
    
    # 2. Test de significativité OLS sur les survivants
    selected = [features[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
    p_values = {f: 1.0 for f in features}
    
    if selected:
        X_ols = sm.add_constant(X[selected])
        ols_model = sm.OLS(y, X_ols).fit()
        for f in selected:
            p_values[f] = ols_model.pvalues[f]
            
    return lasso.coef_, p_values

def prep_gdelt_data(df_geo_region, cols_to_fetch, df_macro, macro_col):
    """Filtre, rééchantillonne, calcule les Z-Scores et fusionne avec la Macro"""
    valid_cols = [c for c in cols_to_fetch if c in df_geo_region.columns]
    if not valid_cols: return None, []
    
    df_g_monthly = df_geo_region[valid_cols].resample('MS').mean()
    z_cols = []
    
    for col in valid_cols:
        rmean = df_g_monthly[col].rolling(12).mean()
        rstd = df_g_monthly[col].rolling(12).std()
        z_col = f"{col}_zscore"
        df_g_monthly[z_col] = (df_g_monthly[col] - rmean) / rstd
        z_cols.append(z_col)
        
    df_merged = pd.merge(df_g_monthly[z_cols], df_macro[[macro_col]], left_index=True, right_index=True, how='inner').dropna()
    return df_merged, z_cols

# =============================================================================
# 🚀 LE PIPELINE HIÉRARCHIQUE COMPLET
# =============================================================================

def run_hierarchical_pipeline(df_geo, region, type_gdelt, parent_sectors, macro_tickers_dict, start_date="2014-01-01"):
    print("="*90)
    print(f"🌍 AUDIT ÉCONOMÉTRIQUE HIÉRARCHIQUE : {region.upper()} | IND. : {type_gdelt.upper()}")
    print("="*90)
    
    # ─── 0. TÉLÉCHARGEMENT MACRO ───
    if region not in macro_tickers_dict:
        print(f"❌ Erreur : Région '{region}' introuvable dans le dictionnaire des tickers.")
        return
        
    ticker = macro_tickers_dict[region]
    df_macro = web.DataReader(ticker, "fred", start_date)
    df_macro.index.name = 'period'
    macro_col = 'Inflation_YoY'
    df_macro[macro_col] = df_macro[ticker].pct_change(periods=12) * 100
    
    # Filtre sur la région pour GDELT
    df_geo_region = df_geo[df_geo['region_key'] == region].set_index('period')
    
    # Lags à étudier (De T+6 = GDELT anticipe l'économie de 6 mois, à T-6 = GDELT réagit avec 6 mois de retard)
    lag_range = list(range(6, -7, -1))
    col_mapper = {l: (f"T+{l}" if l > 0 else (f"T{l}" if l < 0 else "T(0)")) for l in lag_range}

    # =========================================================================
    # 📌 PHASE 1 : LE FILTRE GLOBAL (GRANDS SECTEURS)
    # =========================================================================
    print("\n" + "▼"*90)
    print("📌 PHASE 1 : COMPÉTITION DES GRANDS THÈMES (MACRO-NARRATIONS)")
    print("Objectif : Identifier les secteurs majeurs qui annoncent l'inflation (Lags T+1 à T+6).")
    print("Lecture  : Les cases foncées à droite (T > 0) sont vos Indicateurs Avancés robustes.")
    print("▼"*90)
    
    base_cols = [f"{type_gdelt}_{s}" for s in parent_sectors]
    df_phase1, z_cols_p1 = prep_gdelt_data(df_geo_region, base_cols, df_macro, macro_col)
    
    if df_phase1 is None or df_phase1.empty:
        print("❌ Données GDELT insuffisantes pour les secteurs demandés.")
        return

    row_labels_p1 = [c.replace('_zscore', '') for c in z_cols_p1] + ["_INERTIE_MACRO_"]
    df_table1 = pd.DataFrame("0.00", index=row_labels_p1, columns=lag_range)
    
    for lag in lag_range:
        X_dict = {col: df_phase1[col].shift(lag) for col in z_cols_p1}
        if lag > 0: # Contrôle Autorégressif (AR) uniquement pour la prévision du futur
            X_dict["_INERTIE_MACRO_"] = df_phase1[macro_col].shift(lag)
            
        df_lag = pd.DataFrame(X_dict, index=df_phase1.index)
        df_lag[macro_col] = df_phase1[macro_col]
        df_lag = df_lag.dropna()
        
        if len(df_lag) < 20: continue
            
        features = list(X_dict.keys())
        coefs, p_vals = run_lasso_horizon(df_lag, features, macro_col)
        
        for i, feat in enumerate(features):
            row_name = "_INERTIE_MACRO_" if feat == "_INERTIE_MACRO_" else feat.replace('_zscore', '')
            df_table1.loc[row_name, lag] = format_coef(coefs[i], p_vals[feat])
            
    df_table1.loc["_INERTIE_MACRO_", [l for l in lag_range if l <= 0]] = "N/A"
    df_table1 = df_table1.rename(columns=col_mapper)[list(col_mapper.values())]
    
    display(df_table1.style.map(apply_strict_color_code).set_properties(**{
        'font-size': '12px', 'padding': '5px', 'text-align': 'center', 'border': '1px solid #e0e0e0'
    }))

    # --> Sélection des Vainqueurs : Thèmes ayant un coef fort (*** ou **) dans le FUTUR (T+1 à T+6)
    winning_parents = []
    for col in z_cols_p1:
        clean_name = col.replace('_zscore', '')
        has_signal = any(("***" in str(df_table1.loc[clean_name, col_mapper[l]]) or 
                          "**" in str(df_table1.loc[clean_name, col_mapper[l]])) for l in range(1, 7))
        if has_signal: winning_parents.append(clean_name)

    if not winning_parents:
        print("\n❌ Aucun secteur majeur n'a de pouvoir prédictif statistiquement significatif. Fin de l'analyse.")
        return

    # =========================================================================
    # 📌 PHASE 2 : LE MICROSCOPE (SOUS-INDICATEURS DES VAINQUEURS)
    # =========================================================================
    print("\n\n" + "▼"*90)
    print(f"📌 PHASE 2 : DEEP DIVE SUR LES THÈMES GAGNANTS ({len(winning_parents)} trouvés)")
    print("Objectif : Pour chaque thème vainqueur, isoler la SOUS-CATÉGORIE exacte qui porte le signal.")
    print("▼"*90)

    top_candidates_for_irf = [] # Stockera les meilleurs sous-indicateurs absolus
    
    for parent in winning_parents:
        print(f"\n🔍 FORAGE DANS LE SECTEUR : {parent.upper()}")
        
        # On aspire toutes les sous-catégories de ce parent précis
        sub_cols = [c for c in df_geo_region.columns if c.startswith(f"{parent}_") and c != parent]
        
        if not sub_cols:
            print(f"  ↳ Le secteur {parent.upper()} n'a pas de sous-catégories détaillées. Il est qualifié tel quel pour l'IRF.")
            # On récupère sa meilleure performance pour le classement IRF
            best_lag_val = max([extract_abs_coef(df_table1.loc[parent, col_mapper[l]]) for l in range(1, 7)])
            top_candidates_for_irf.append({'sector': f"{parent}_zscore", 'abs_coef': best_lag_val, 'df': df_phase1})
            continue

        # Si on a des sous-catégories, on fait le LASSO entre elles
        df_phase2, z_cols_p2 = prep_gdelt_data(df_geo_region, sub_cols, df_macro, macro_col)
        row_labels_p2 = [c.replace('_zscore', '') for c in z_cols_p2] + ["_INERTIE_MACRO_"]
        df_table2 = pd.DataFrame("0.00", index=row_labels_p2, columns=lag_range)
        
        best_sub_coef = 0
        best_sub_name = None
        
        for lag in lag_range:
            X_dict = {col: df_phase2[col].shift(lag) for col in z_cols_p2}
            if lag > 0: X_dict["_INERTIE_MACRO_"] = df_phase2[macro_col].shift(lag)
                
            df_lag = pd.DataFrame(X_dict, index=df_phase2.index)
            df_lag[macro_col] = df_phase2[macro_col]
            df_lag = df_lag.dropna()
            
            if len(df_lag) < 20: continue
                
            features = list(X_dict.keys())
            coefs, p_vals = run_lasso_horizon(df_lag, features, macro_col)
            
            for i, feat in enumerate(features):
                row_name = "_INERTIE_MACRO_" if feat == "_INERTIE_MACRO_" else feat.replace('_zscore', '')
                coef_val = coefs[i]
                pval = p_vals[feat]
                df_table2.loc[row_name, lag] = format_coef(coef_val, pval)
                
                # Sauvegarde du meilleur candidat (Seulement dans le futur, p < 0.05)
                if feat != "_INERTIE_MACRO_" and lag > 0 and pval < 0.05:
                    if abs(coef_val) > best_sub_coef:
                        best_sub_coef = abs(coef_val)
                        best_sub_name = feat

        df_table2.loc["_INERTIE_MACRO_", [l for l in lag_range if l <= 0]] = "N/A"
        df_table2 = df_table2.rename(columns=col_mapper)[list(col_mapper.values())]
        
        display(df_table2.style.map(apply_strict_color_code).set_properties(**{
            'font-size': '11px', 'padding': '4px', 'text-align': 'center', 'border': '1px solid #e0e0e0'
        }))
        
        if best_sub_name:
            print(f"  🏆 Meilleure Narration détectée : {best_sub_name.replace('_zscore', '').upper()} (Impact: {best_sub_coef:.2f})")
            top_candidates_for_irf.append({'sector': best_sub_name, 'abs_coef': best_sub_coef, 'df': df_phase2})
        else:
            print("  ⚠ Aucune sous-catégorie n'a survécu à l'isolation.")

    # =========================================================================
    # 📌 PHASE 3 : MODÉLISATION DE CHOC (VAR / IRF) SUR L'ÉLITE
    # =========================================================================
    if not top_candidates_for_irf:
        print("\n❌ Aucun sous-indicateur qualifié pour les IRF.")
        return

    print("\n\n" + "▼"*90)
    print("📌 PHASE 3 : SIMULATION DE CRISE (FONCTIONS DE RÉPONSE IMPULSIONNELLE)")
    print("Objectif : Visualiser l'impact économique sur 12 mois d'un choc sur les meilleurs narratifs.")
    print("▼"*90)

    # Trier pour ne prendre que le Top 2 absolu (Évite de saturer l'écran de graphiques)
    top_candidates_for_irf = sorted(top_candidates_for_irf, key=lambda x: x['abs_coef'], reverse=True)[:5]

    for cand in top_candidates_for_irf:
        best_sector = cand['sector']
        clean_name = best_sector.replace('_zscore', '')
        print(f"\n💥 Choc Simulatif (+1 Écart-type) sur le récit : {clean_name.upper()}")
        
        # On utilise le DataFrame de la Phase 2 où les Z-scores étaient déjà calculés proprement
        df_var = cand['df'][[best_sector, macro_col]].dropna()
        
        try:
            model = VAR(df_var)
            results = model.fit(maxlags=3, ic='aic')
            irf = results.irf(12)
            
            lower_bound_matrix, upper_bound_matrix = irf.err_band_sz1(orth=True, signif=0.05)
            effect_on_macro = irf.orth_irfs[:, 1, 0] # Choc sur index 0 (GDELT) -> Effet sur index 1 (Macro)
            lower_bound = lower_bound_matrix[:, 1, 0]
            upper_bound = upper_bound_matrix[:, 1, 0]
            
            plt.figure(figsize=(10, 4))
            plt.plot(range(13), effect_on_macro, color='#c62828' if effect_on_macro[1] < 0 else '#1565c0', linewidth=2.5, label=f"Impact sur l'Inflation YoY")
            plt.fill_between(range(13), lower_bound, upper_bound, color='gray', alpha=0.15, label="Intervalle Confiance (95%)")
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            
            plt.title(f"Propagation économique suite à un choc médiatique '{clean_name}'", fontsize=12)
            plt.xlabel("Mois après le choc", fontweight='bold')
            plt.ylabel("Impact (%)", fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Erreur lors du calcul de l'IRF pour {clean_name}: {e}")

    print("\n" + "="*90)
    print(f"✅ FIN DE L'AUDIT POUR {region.upper()}")
    print("="*90 + "\n")