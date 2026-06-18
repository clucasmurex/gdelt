# fileName: gdelt_monthly_pipeline.py
import os
import sys
import shutil
import urllib.request
import subprocess
import zipfile
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# --- SYSTÈME DE DICTIONNAIRE GLOBAL DE SOURCES ---
# Centralise et associe chaque nom de domaine (ex: nytimes.com) à un ID unique (int32)
SOURCE_DICT_PATH = "./gdelt_sources_mapping.json"
source_to_id = {}
id_to_source = {}
next_source_id = 1

def load_source_dictionary():
    global source_to_id, id_to_source, next_source_id
    if os.path.exists(SOURCE_DICT_PATH):
        try:
            with open(SOURCE_DICT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                source_to_id = data.get("source_to_id", {})
                id_to_source = {int(k): v for k, v in data.get("id_to_source", {}).items()}
                if source_to_id:
                    next_source_id = max(int(v) for v in source_to_id.values()) + 1
        except Exception as e:
            print(f"⚠️ Impossible de charger le dictionnaire de sources : {e}")

def save_source_dictionary():
    global source_to_id, id_to_source
    try:
        with open(SOURCE_DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump({"source_to_id": source_to_id, "id_to_source": id_to_source}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder le dictionnaire de sources : {e}")

def get_or_create_source_id(source_name):
    global next_source_id
    if not isinstance(source_name, str) or not source_name:
        return 0  # 0 désigne une source inconnue ou manquante
    if source_name in source_to_id:
        return source_to_id[source_name]
    
    current_id = next_source_id
    source_to_id[source_name] = current_id
    id_to_source[current_id] = source_name
    next_source_id += 1
    return current_id


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
        8,   # V2ENHANCEDTHEMES
        10,  # V2ENHANCEDLOCATIONS
        11,  # V1PERSONS
        13,  # V1ORGANIZATIONS
        15,  # V1.5TONE
        26   # V2.1TRANSLATIONINFO (approx. fin du schéma standard)
    ]

    column_names = [
        "GKGRECORDID",
        "DATE",
        "SourceCollectionIdentifier",
        "SourceCommonName",
        "DocumentIdentifier",
        "EnhancedThemes",
        "EnhancedLocations",
        "Persons",
        "Organizations",
        "Tone_Raw", 
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
                
                # --- OPTIMISATION DRUSTIQUE DES TYPES PRIMITIFS ---
                tone_split = df['Tone_Raw'].str.split(',', expand=False)
                df['Tone'] = tone_split.str[0]
                df['Tone'] = pd.to_numeric(df['Tone'], errors='coerce').fillna(0.0).astype('float16')
                
                df['WordCount'] = tone_split.str[6]
                df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce').fillna(0).astype('int32')
                df.drop(columns=['Tone_Raw'], inplace=True)
                
                df['SourceCollectionIdentifier'] = df['SourceCollectionIdentifier'].astype('category')
                
                # SÉCURITÉ ANTI-DOUBLONS : Nettoyage local immédiat pour chaque lot de 15 min
                df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                
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
        load_source_dictionary() # Chargement initial du JSON externe au démarrage

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
        
        print(f"\n🚀 PIPELINE MENSUEL ULTRA-STREAMING (DICTIONNAIRE INT-JSON ENGAGÉ)")
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

            # ÉTAPE 2 & 3 : Extraction, Dédoublonnage, Encodage des sources & Streaming direct
            print(f"  🗜️  Parsing, Mapping ID et Streaming direct sur disque via {self.cpu_workers} cœurs...")
            local_zips = list(self.temp_dir.glob('*.zip'))
            
            writer = None
            schema = None
            articles_comptes = 0

            # max_tasks_per_child=5 force la libération régulière de la RAM des processus enfants
            with ProcessPoolExecutor(max_workers=self.cpu_workers, max_tasks_per_child=5) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, str(p)): p for p in local_zips}
                
                for future in as_completed(cpu_futures):
                    res = future.result()
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        
                        # --- ENCODAGE DYNAMIQUE ET SÉCURISÉ DES SOURCES ---
                        # Extraction des domaines uniques de ce chunk de 15 minutes
                        unique_sources = res['SourceCommonName'].dropna().unique()
                        for src in unique_sources:
                            get_or_create_source_id(src)
                        
                        # Remplacement du texte de la source par son ID numérique
                        res['SourceCommonName_ID'] = res['SourceCommonName'].map(source_to_id).fillna(0).astype('int32')
                        res.drop(columns=['SourceCommonName'], inplace=True)
                        
                        # Conversion immédiate en table binaire PyArrow
                        table = pa.Table.from_pandas(res, preserve_index=False)
                        articles_comptes += len(res)
                        
                        # Initialisation du fichier Parquet au tout premier bloc reçu
                        if writer is None:
                            schema = table.schema
                            writer = pq.ParquetWriter(
                                parquet_filepath, 
                                schema, 
                                compression='zstd', 
                                compression_level=12, # Compression maximale force 12
                                use_dictionary=True
                            )
                        
                        # Écriture immédiate du bloc sur le disque (libération de la RAM principale)
                        writer.write_table(table)
                        
                        del res
                        del table
            
            # ÉTAPE 3.5 : Sauvegarde physique du dictionnaire JSON mis à jour pour ce mois-ci
            save_source_dictionary()

            # Fermeture propre du fichier Parquet mensuel
            if writer is not None:
                writer.close()
                print(f"  🟢 Fichier créé en streaming : {parquet_filename} ({articles_comptes:,} articles enregistrés).")
            else:
                print(f"  ❌ Échec : Aucun article extrait pour le mois {month_str}")

            # ÉTAPE 4 : Purge immédiate de l'espace tampon
            print(f"  🧹 Libération de l'espace disque de la zone tampon...")
            for f in self.temp_dir.glob('*'):
                try: os.remove(f)
                except Exception: pass
            print(f"  ✨ Zone tampon vidée.\n")

        print("🏆 PIPELINE EXÉCUTÉ AVEC SUCCÈS ! TOUS LES PARAMÈTRES SONT AU VERT.")

if __name__ == "__main__":
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',    
        final_dir='./gdelt_parquet_db',   
        net_workers=8,        # Doux avec le réseau
        cpu_workers=24        # Courtois avec les collègues connectés sur la machine
    )
    
    pipeline.process_pipeline(
        start_date='2015-03-01',
        end_date='2015-03-03'
    )