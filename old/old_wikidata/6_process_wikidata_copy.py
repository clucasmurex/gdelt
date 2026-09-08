import os
import glob
import json
import re
import pandas as pd
import tldextract
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# --- CONFIGURATION ---
WIKIDATA_DIR = "data/wikidata"
OUTPUT_DIR = "data/domains"
JSON_MAPPING_PATH = "/data/gdelt/gdelt_sources_mapping.json"
MAX_WORKERS = 16

# --- 1. EXCLUSIONS SUR LES LABELS WIKIDATA (medialabel + typelabel) ---
# Sécurisé : "hinduism|monastery|theolog" au lieu de "hindu" (pour sauver The Hindu)
EXCLUDED_LABEL_KEYWORDS = [
    # Éducation & Science pure
    "university", "college", "school", "high school", "student", "academic journal",
    "scientific journal", "peer-reviewed", "proceedings",
    # Sport & Divertissement
    "sports", "sports news", "tabloid", "video games", "gaming", "gamer", 
    "movie", "cinema", "film", "entertainment", "pop culture", "music", "celebrit",
    # Religion & Communautaire (Sécurisé !)
    "church", "religion", "catholic", "christian", "baptist", "jewish", "islamic", 
    "muslim", "hinduism", "monastery", "theolog", "lgbt", "gay",
    # Local & Formats spécifiques
    "municipal newsletter", "children's newspaper", "local newspaper", "fake news", "satire"
]

EXCLUSION_LABEL_PATTERN = '|'.join(re.escape(kw) for kw in EXCLUDED_LABEL_KEYWORDS)

# --- 2. EXCLUSIONS SUR LE NOM DE DOMAINE (URL / Website) ---
# Sécurisé : "[-.]deal|deal[-.]" ignore "ideal.es" ; "\.edu$" élimine MIT et Cornell
EXCLUDED_DOMAIN_PATTERNS = [
    r"\.edu$", r"university", r"college", r"school",  # Éducation
    r"gaming|gamer|esports?|videojuego|playstation|nintendo|xbox|3djuegos|kotaku|ign\.|gamespot",  # Gaming
    r"crypto|bitcoin|casino|betting|poker|apuestas|odds|gambling|token|nft[-.]|[-.]nft",  # Crypto & Paris
    r"horoscope|astrolog|zodiac|tarot",  # Astrologie
    r"coupon|promocod|discount|[-.]deal|deal[-.]|dailydeal|hotdeal|shoponline|boutique",  # Coupons (Sécurisé !)
    r"^ij[a-z]{2,}\.org|journalof|sciedu|springer|ieee|sciencedirect"  # Revues académiques
]

EXCLUSION_DOMAIN_PATTERN = '|'.join(EXCLUDED_DOMAIN_PATTERNS)

worker_source_mapping = {}
worker_tld_extractor = None

def init_worker(json_path):
    """Initialise le mapping JSON et l'extracteur TLD dans la mémoire de chaque worker."""
    global worker_source_mapping, worker_tld_extractor
    worker_tld_extractor = tldextract.TLDExtract()
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            worker_source_mapping = data.get("source_to_id", {})
    except Exception as e:
        print(f"Erreur lors du chargement du JSON: {e}")
        worker_source_mapping = {}

def clean_url_to_domain(url):
    """Nettoie l'URL brute pour extraire le netloc de base en minuscules."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
        
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc.lower()
    except:
        return None

def get_gdelt_match(domain):
    """Applique la recherche en cascade en 3 étapes pour trouver le domaine et l'ID dans le JSON."""
    if not domain:
        return None
    
    # 1. Test du domaine exact
    if domain in worker_source_mapping:
        return worker_source_mapping[domain], domain
        
    # 2. Test via tldextract (sans le deprecation warning)
    ext = worker_tld_extractor(domain)
    top_domain = ext.top_domain_under_public_suffix
    if top_domain and top_domain in worker_source_mapping:
        return worker_source_mapping[top_domain], top_domain
        
    # 3. Test via découpage direct (fallback)
    parts = domain.split('.')
    if len(parts) > 2:
        root_domain = ".".join(parts[-2:])
        if root_domain in worker_source_mapping:
            return worker_source_mapping[root_domain], root_domain
            
    return None


def process_csv(filepath):
    global worker_source_mapping
    
    filename = os.path.basename(filepath)
    region = filename[:2].upper()
    
    csv_domains = set()
    matched_records = []
    
    try:
        use_cols_func = lambda col: col.lower() in ['medialabel', 'typelabel', 'countrylabel', 'inception', 'website']
        df = pd.read_csv(filepath, usecols=use_cols_func)
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        print(f"Erreur de lecture pour {filepath}: {e}")
        return region, csv_domains, matched_records

    # 1. Nettoyage initial et extraction du domaine EN PREMIER
    if 'website' not in df.columns:
        return region, csv_domains, matched_records
    
    df = df.dropna(subset=['website']).copy()
    df['raw_domain'] = df['website'].apply(clean_url_to_domain)
    df = df.dropna(subset=['raw_domain']).copy()
    
    if df.empty:
        return region, csv_domains, matched_records

    # 2. 🔥 LE DOUBLE BOUCLIER D'EXCLUSION (Sur le Texte ET sur l'URL)
    # A. Check sur les labels textuels (medialabel + typelabel)
    text_to_check = df['medialabel'].fillna('').astype(str) + " " + df['typelabel'].fillna('').astype(str)
    mask_label_excluded = text_to_check.str.contains(EXCLUSION_LABEL_PATTERN, case=False, na=False)
    
    # B. Check direct sur le nom de domaine (élimine mit.edu, ign.com, gamespot.com...)
    mask_domain_excluded = df['raw_domain'].str.contains(EXCLUSION_DOMAIN_PATTERN, case=False, regex=True, na=False)
    
    # On élimine la ligne si elle échoue sur l'un OU l'autre des boucliers
    df_filtered = df[~(mask_label_excluded | mask_domain_excluded)].copy()
    
    if df_filtered.empty:
        return region, csv_domains, matched_records

    # 3. Déduplication interne par nom de domaine
    df_dedup = df_filtered.drop_duplicates(subset=['raw_domain'])
    csv_domains = set(df_dedup['raw_domain'])
    
    # 4. Recherche d'intersection avec le JSON GDELT
    for _, row in df_dedup.iterrows():
        match_result = get_gdelt_match(row['raw_domain'])
        if match_result:
            gdelt_id, matched_domain = match_result
            matched_records.append({
                'id': gdelt_id,
                'domain': matched_domain,
                'medialabel': row.get('medialabel', None),
                'typelabel': row.get('typelabel', None),
                'countrylabel': row.get('countrylabel', None),
                'inception': row.get('inception', None)
            })
            
    return region, csv_domains, matched_records

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(WIKIDATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {WIKIDATA_DIR}.")
        return

    print(f"{len(csv_files)} fichiers CSV à traiter avec {MAX_WORKERS} workers max...\n")
    print(f"{'RÉGION':<8} | {'FICHIER':<30} | {'MATCHS':<10} | {'TOTAL CSV':<10} | {'COUVERTURE':<10}")
    print("-" * 77)
    
    aggregated_csv_domains = defaultdict(set)
    # Dictionnaire pour dédupliquer les métadonnées par domaine lors de la fusion (ex: EU1 + EU2)
    aggregated_matches = defaultdict(dict)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(JSON_MAPPING_PATH,)) as executor:
        futures = {executor.submit(process_csv, fp): fp for fp in csv_files}
        
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                region, csv_domains, matched_records = future.result()
                
                aggregated_csv_domains[region].update(csv_domains)
                
                # Ajout des résultats et déduplication inter-fichiers sur le 'domain'
                for record in matched_records:
                    dom = record['domain']
                    if dom not in aggregated_matches[region]:
                        aggregated_matches[region][dom] = record
                
                nb_csv = len(csv_domains)
                nb_match = len(matched_records)
                ratio = (nb_match / nb_csv * 100) if nb_csv > 0 else 0.0
                print(f"[{region:<6}] | {os.path.basename(filepath):<30} | {nb_match:<10} | {nb_csv:<10} | {ratio:.1f}%")
                
            except Exception as e:
                print(f"Erreur avec {filepath}: {e}")

    # --- BILAN GLOBAL PAR RÉGION ---
    print("\n" + "="*77)
    print("BILAN GLOBAL PAR RÉGION (Après exclusions et déduplication)")
    print("="*77)
    print(f"{'RÉGION':<10} | {'DOMAINES FILTRÉS (CSV)':<25} | {'TROUVÉS DANS JSON':<20} | {'TAUX':<10}")
    print("-" * 77)

    
    # 1. Calculer les statistiques pour toutes les régions
    summary_stats = []
    for region in aggregated_csv_domains.keys():
        total_csv_region = len(aggregated_csv_domains[region])
        total_match_region = len(aggregated_matches[region])
        ratio_region = (total_match_region / total_csv_region * 100) if total_csv_region > 0 else 0.0
        summary_stats.append((region, total_csv_region, total_match_region, ratio_region))
        
    # 2. Trier la liste par le taux (index 3 du tuple), en ordre décroissant
    summary_stats.sort(key=lambda x: x[3], reverse=True)
    
    # 3. Afficher les résultats triés
    for region, total_csv_region, total_match_region, ratio_region in summary_stats:
        print(f"{region:<10} | {total_csv_region:<25} | {total_match_region:<20} | {ratio_region:.1f}%")
    
    # --- SAUVEGARDE EN PARQUET ---
    print("\nÉcriture des fichiers Parquet enrichis de la whitelist...")
    for region, matches_dict in aggregated_matches.items():
        if matches_dict:
            # Création du DataFrame avec toutes les colonnes enrichies
            df_out = pd.DataFrame(list(matches_dict.values()))
            
            # Réordonner proprement les colonnes
            cols_order = ['id', 'domain', 'medialabel', 'typelabel', 'countrylabel', 'inception']
            df_out = df_out[cols_order]
            
            # NOUVEAU : Convertir les valeurs nulles en chaînes vides, puis forcer le type texte
            # Cela empêche Pandas de typer les colonnes vides en Float64
            df_out = df_out.fillna('').astype(str)
            
            output_path = os.path.join(OUTPUT_DIR, f"domains_{region}.parquet")
            df_out.to_parquet(output_path, index=False)
            print(f"  -> {output_path} : {len(df_out)} domaines enregistrés avec métadonnées")
        else:
            print(f"  -> {region} : Aucune correspondance après filtrage, fichier ignoré.")

if __name__ == "__main__":
    main()