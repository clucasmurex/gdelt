# fileName: gdelt_monthly_pipeline.py
import os
import sys
import shutil
import urllib.request
import subprocess
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def parse_zip_worker(zip_filepath):
    """
    Ouvre un ZIP local en mémoire de manière ultra-optimisée.
    Ne charge STRICTEMENT que les colonnes demandées pour économiser la RAM.
    """
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
        26   # V2.1TRANSLATIONINFO (approx. fin du schéma standard)
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
        "Tone_Raw", # Gardé pour le split
        "GCAM",
        "TranslationInfo"
    ]
    
    try:
        with zipfile.ZipFile(zip_filepath) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                df = pd.read_csv(
                    f, sep='\t', header=None, 
                    usecols=target_indices,
                    names=column_names,
                    dtype=str, 
                    encoding='utf-8', 
                    on_bad_lines='skip'
                )
                
                # --- OPTIMISATION DU STOCKAGE DE LA SÉRIE TEMPORELLE ---
                tone_split = df['Tone_Raw'].str.split(',', expand=False)
                
                df['Tone'] = tone_split.str[0]
                df['Tone'] = pd.to_numeric(df['Tone'], errors='coerce').fillna(0.0).astype('float32')
                
                df['WordCount'] = tone_split.str[6]
                df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce').fillna(0).astype('int32')
                
                df.drop(columns=['Tone_Raw'], inplace=True)
                
        return df
    except Exception:
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
    def __init__(self, temp_dir, final_dir, net_workers=8, cpu_workers=32):
        self.temp_dir = Path(temp_dir)
        self.final_dir = Path(final_dir)
        self.net_workers = net_workers
        self.cpu_workers = cpu_workers
        self.valid_gkg_urls = set()
        
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.final_dir.mkdir(exist_ok=True, parents=True)

    def load_master_list(self):
        """Récupère la liste officielle des fichiers existants sur GDELT."""
        print("📋 Chargement de la Master File List GDELT...")
        master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        try:
            req = urllib.request.Request(master_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str: continue
                    url = line_str.split(' ')[-1]
                    if 'gkg.csv.zip' in url:
                        self.valid_gkg_urls.add(url)
            print(f"✓ {len(self.valid_gkg_urls):,} fichiers GKG uniques répertoriés.")
        except Exception as e:
            print(f"💥 Erreur critique Master List : {e}")
            sys.exit(1)

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
        self.load_master_list()
        months_to_process = self.generate_months_list(start_date, end_date)
        
        print(f"\n🚀 PIPELINE MENSUEL (OPTIMISATION DE COMPRESSION ZSTD FORCE 12)")
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
            month_urls = [url for url in self.valid_gkg_urls if url.split('/')[-1].startswith(month_str)]
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
            # --- CORRECTION DE LA LIGNE 164 CI-DESSOUS ---
            print(f"  🗜️  Parsing sélectif et compression Parquet via {self.cpu_workers} cœurs...")
            local_zips = list(self.temp_dir.glob('*.zip'))
            month_dfs = []

            with ProcessPoolExecutor(max_workers=self.cpu_workers) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, str(p)): p for p in local_zips}
                for future in as_completed(cpu_futures):
                    res = future.result()
                    if isinstance(res, pd.DataFrame):
                        month_dfs.append(res)

            # ÉTAPE 3 : Écriture finale (MODIFIÉE POUR LE MAX DE COMPRESSION)
            if month_dfs:
                full_month_df = pd.concat(month_dfs, ignore_index=True)
                full_month_df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                
                print("  🗜️  Passage de Pandas à PyArrow et compression ZSTD (Niveau 12)...")
                # 1. Conversion du DataFrame en table PyArrow native
                table = pa.Table.from_pandas(full_month_df, preserve_index=False)
                
                # 2. Écriture directe via le moteur de PyArrow avec un niveau ZSTD agressif
                pq.write_table(
                    table, 
                    parquet_filepath, 
                    compression='zstd', 
                    compression_level=12, 
                    use_dictionary=True
                )
                
                print(f"  🟢 Fichier créé : {parquet_filename} ({len(full_month_df):,} articles).")
                
                del table
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

        print("🏆 PIPELINE EXÉCUTÉ AVEC SUCCÈS !")

if __name__ == "__main__":
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',    
        final_dir='./gdelt_parquet_db',   
        net_workers=8,        # Doux avec le réseau
        cpu_workers=24        # Sympa avec les collègues sur la machine !
    )
    
    pipeline.process_pipeline(
        start_date='2015-10-01',
        end_date='2026-06-16'
    )