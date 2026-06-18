# fileName: gdelt_monthly_pipeline.py
import os
import sys
import shutil
import urllib.request
import subprocess
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def parse_zip_worker(args):
    """
    Ouvre un ZIP local en mémoire de manière ultra-optimisée.
    Ne charge STRICTEMENT que les colonnes demandées pour économiser la RAM.
    """
    zip_filepath, is_translated = args
    
    target_indices = [
        0,   # GKGRECORDID
        1,   # V2.1DATE
        2,   # V2SOURCECOLLECTIONIDENTIFIER
        3,   # V2SOURCECOMMONNAME
        4,   # V2DOCUMENTIDENTIFIER
        7,   # V1THEMES
        8,   # V2ENHANCEDTHEMES
        9,   # V1LOCATIONS
        10,  # V2ENHANCEDLOCATIONS
        11,  # V1PERSONS
        13,  # V1ORGANIZATIONS
        15,  # V1.5TONE
        17,  # V2GCAM
        26   # V2.1TRANSLATIONINFO
    ]

    column_names = [
        "GKGRECORDID",
        "DATE",
        "SourceCollectionIdentifier",
        "SourceCommonName",
        "DocumentIdentifier",
        "Themes",
        "EnhancedThemes",
        "Locations",
        "EnhancedLocations",
        "Persons",
        "Organizations",
        "Tone_Raw",
        "GCAM",
        "TranslationInfo"
    ]
    
    try:
        with zipfile.ZipFile(zip_filepath) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                # usecols force Pandas à ignorer le reste du fichier géant à la lecture
                df = pd.read_csv(
                    f, sep='\t', header=None, 
                    usecols=target_indices,
                    names=column_names,
                    dtype=str, # Optimise le parsing initial en évitant le typage automatique
                    encoding='utf-8', 
                    on_bad_lines='skip'
                )
                
                # --- OPTIMISATION DU STOCKAGE DE LA SÉRIE TEMPORELLE ---
                # On nettoie immédiatement la colonne Tone pour économiser des dizaines de Go de texte
                tone_split = df['Tone_Raw'].str.split(',', expand=False)
                
                # Extraction du sentiment net (index 0) et conversion en float léger
                df['Tone'] = tone_split.str[0]
                df['Tone'] = pd.to_numeric(df['Tone'], errors='coerce').fillna(0.0).astype('float32')
                
                # Extraction du Word Count (index 6) et conversion en entier léger
                df['WordCount'] = tone_split.str[6]
                df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce').fillna(0).astype('int32')
                
                # Ajout du flag de traduction (0 = Anglais natif, 1 = Traduit) en format ultra-léger (1 octet)
                df['translated'] = int(is_translated)
                df['translated'] = df['translated'].astype('uint8')
                
                # On supprime la lourde chaîne de caractères brute
                df.drop(columns=['Tone_Raw'], inplace=True)
                
        return df
    except Exception:
        # En cas de fichier ZIP corrompu, on renvoie None pour ne pas bloquer le pipeline
        return None

def download_file_worker(url, temp_dir):
    """Télécharge un ZIP unitaire via wget de manière furtive et rapide."""
    filename = url.split('/')[-1]
    filepath = os.path.join(temp_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return "EXISTS"

    cmd = [
        "wget", "--tries=3", "--waitretry=1", "--quiet",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-O", filepath, url
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return "DOWNLOADED"
        if os.path.exists(filepath): os.remove(filepath)
        return "FAILED"
    except Exception:
        if os.path.exists(filepath): os.remove(filepath)
        return "TIMEOUT"

class GDELTRollingPipeline:
    def __init__(self, temp_dir, final_dir, net_workers=16, cpu_workers=64):
        self.temp_dir = Path(temp_dir)
        self.final_dir = Path(final_dir)
        self.net_workers = net_workers
        self.cpu_workers = cpu_workers
        
        # Utilisation d'un dictionnaire { url: is_translated (True/False) } pour dédoublonner à la source
        self.valid_gkg_urls = {}
        
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.final_dir.mkdir(exist_ok=True, parents=True)

    def _fetch_master_list(self, url, is_translation_list=False):
        """Méthode interne pour parser une liste GDELT spécifique."""
        count = 0
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str: continue
                    file_url = line_str.split(' ')[-1]
                    if 'gkg.csv.zip' in file_url:
                        # Si le fichier existe déjà et qu'on traite la liste translingue, 
                        # elle écrase la version anglaise car elle apporte la valeur ajoutée de la traduction.
                        self.valid_gkg_urls[file_url] = is_translation_list
                        count += 1
            return count
        except Exception as e:
            print(f"💥 Erreur lors de la récupération de {url} : {e}")
            return 0

    def load_master_lists(self):
        """Récupère et fusionne la liste officielle et la liste translingue de GDELT."""
        print("📋 Chargement de la Master File List GDELT (English)...")
        english_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        eng_count = self._fetch_master_list(english_url, is_translation_list=False)
        print(f"  ✓ {eng_count:,} fichiers GKG anglophones identifiés.")

        print("📋 Chargement de la Master File List GDELT (Translingual)...")
        trans_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"
        trans_count = self._fetch_master_list(trans_url, is_translation_list=True)
        print(f"  ✓ {trans_count:,} fichiers GKG translingues identifiés.")
        
        print(f"🎯 Total après dédoublonnage strict : {len(self.valid_gkg_urls):,} fichiers uniques prêts à être traités.")

    def generate_months_list(self, start_date, end_date):
        """Génère la liste des chaînes YYYYMM de manière robuste."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        months = []
        current = datetime(start.year, start.month, 1)
        
        while current <= end:
            months.append(current.strftime('%Y%m'))
            next_month_approx = current + timedelta(days=32)
            current = datetime(next_month_approx.year, next_month_approx.month, 1)
            
        return months

    def process_pipeline(self, start_date, end_date):
        self.load_master_lists()
        months_to_process = self.generate_months_list(start_date, end_date)
        
        print(f"\n🚀 SÉCURISATION DU PIPELINE ROULANT MULTILINGUE (MODE ULTRA-COMPACT)")
        print(f"📂 Zone Tampon Éphémère : {self.temp_dir}")
        print(f"💾 Base Parquet Finale   : {self.final_dir}\n")

        for idx, month_str in enumerate(months_to_process, 1):
            parquet_filename = f"gdelt_{month_str[:4]}-{month_str[4:]}.parquet"
            parquet_filepath = self.final_dir / parquet_filename
            
            if parquet_filepath.exists():
                print(f"⏭️  [{idx}/{len(months_to_process)}] Mois {month_str} déjà converti. Passé.")
                continue

            print(f"🔄 [{idx}/{len(months_to_process)}] Traitement du mois : {month_str}")
            
            # Sélection des URLs associées au mois en cours
            month_urls = [url for url in self.valid_gkg_urls.keys() if url.split('/')[-1].startswith(month_str)]
            total_files = len(month_urls)
            
            if total_files == 0:
                print(f"  ⚠ Aucun fichier pour le mois {month_str}.")
                continue

            # ÉTAPE 1 : Téléchargement dans la zone tampon
            print(f"  📥 Téléchargement de {total_files} fichiers (.zip) en zone tampon...")
            success, skipped, failed = 0, 0, 0
            
            with ThreadPoolExecutor(max_workers=self.net_workers) as net_executor:
                futures = {net_executor.submit(download_file_worker, url, str(self.temp_dir)): url for url in month_urls}
                for future in as_completed(futures):
                    status = future.result()
                    if status == "DOWNLOADED": success += 1
                    elif status == "EXISTS": skipped += 1
                    else: failed += 1
            print(f"  ✓ Fin téléchargement (Nouveaux: {success} | Présents: {skipped} | Échecs: {failed})")

            # ÉTAPE 2 : Extraction sélective & Compilation multi-processus
            print(f"  🗜️  Parsing sélectif et compression Parquet via {self.cpu_workers} cœurs...")
            
            # Préparation des arguments pour le ProcessPool (chemin_local, is_translated)
            worker_tasks = []
            for url in month_urls:
                filename = url.split('/')[-1]
                local_path = self.temp_dir / filename
                if local_path.exists() and local_path.stat().st_size > 0:
                    is_translated = self.valid_gkg_urls[url]
                    worker_tasks.append((str(local_path), is_translated))

            month_dfs = []
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, task): task[0] for task in worker_tasks}
                for future in as_completed(cpu_futures):
                    res = future.result()
                    if isinstance(res, pd.DataFrame):
                        month_dfs.append(res)

            # ÉTAPE 3 : Écriture finale
            if month_dfs:
                full_month_df = pd.concat(month_dfs, ignore_index=True)
                
                # Suppression des doublons basés sur la clé primaire de l'article GKG
                full_month_df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                
                # Écriture Parquet avec compression Snappy
                full_month_df.to_parquet(parquet_filepath, index=False, compression='snappy')
                print(f"  🟢 Fichier créé : {parquet_filename} ({len(full_month_df):,} articles enregistrés).")
                
                del full_month_df
                del month_dfs
            else:
                print(f"  ❌ Échec : Aucun article extrait pour le mois {month_str}")

            # ÉTAPE 4 : Purge immédiate de l'espace tampon
            print(f"  🧹 Libération de l'espace disque de la zone tampon...")
            for f in self.temp_dir.glob('*'):
                try:
                    os.remove(f)
                except Exception:
                    pass
            print(f"  ✨ Zone tampon vidée.\n")

        print("🏆 PIPELINE EXÉCUTÉ AVEC SUCCÈS ! TA BASE DE DONNÉES UNIFIÉE EST PRÊTE.")

if __name__ == "__main__":
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',    
        final_dir='./gdelt_parquet_db',   
        net_workers=16,                   
        cpu_workers=64                    
    )
    
    pipeline.process_pipeline(
        start_date='2015-02-19',
        end_date='2026-06-16'
    )