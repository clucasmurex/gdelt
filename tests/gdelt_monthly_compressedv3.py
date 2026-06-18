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
        except Exception:
            # Si le fichier est vide ou corrompu, on l'initialise à blanc proprement
            source_to_id = {}
            id_to_source = {}
            next_source_id = 1

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
        return 0
    if source_name in source_to_id:
        return source_to_id[source_name]
    
    current_id = next_source_id
    source_to_id[source_name] = current_id
    id_to_source[current_id] = source_name
    next_source_id += 1
    return current_id


def parse_zip_worker(args):
    """
    Ouvre un ZIP local en mémoire de manière ultra-optimisée.
    Ajoute le flag de traduction translingue.
    """
    zip_filepath, is_translated = args
    
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
        26   # V2.1TRANSLATIONINFO
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
                
                # Extraction & Typage optimisé (float16 / int32)
                tone_split = df['Tone_Raw'].str.split(',', expand=False)
                df['Tone'] = tone_split.str[0]
                df['Tone'] = pd.to_numeric(df['Tone'], errors='coerce').fillna(0.0).astype('float16')
                
                df['WordCount'] = tone_split.str[6]
                df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce').fillna(0).astype('int32')
                df.drop(columns=['Tone_Raw'], inplace=True)
                
                # Flag de traduction (0 = Anglais d'origine, 1 = Traduit via la base Translingual)
                df['translated'] = int(is_translated)
                df['translated'] = df['translated'].astype('uint8')
                
                df['SourceCollectionIdentifier'] = df['SourceCollectionIdentifier'].astype('category')
                df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                
        return df
    except Exception:
        return None

def download_file_worker(url, temp_dir):
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
    def __init__(self, temp_dir, final_dir, net_workers=24, cpu_workers=48):
        self.temp_dir = Path(temp_dir)
        self.final_dir = Path(final_dir)
        self.net_workers = net_workers
        self.cpu_workers = cpu_workers
        self.valid_gkg_urls = {} # URL -> is_translated (True/False)
        
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.final_dir.mkdir(exist_ok=True, parents=True)
        load_source_dictionary()

    def _fetch_master_list(self, url, is_translation_list=False):
        count = 0
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str: continue
                    file_url = line_str.split(' ')[-1]
                    if 'gkg.csv.zip' in file_url:
                        # Si doublon, la liste translingue écrase l'autre pour conserver la métadonnée de traduction
                        self.valid_gkg_urls[file_url] = is_translation_list
                        count += 1
            return count
        except Exception as e:
            print(f"💥 Erreur lors de la récupération de {url} : {e}")
            return 0

    def load_master_lists(self):
        print("📋 Chargement de la Master File List GDELT (English)...")
        english_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        eng_count = self._fetch_master_list(english_url, is_translation_list=False)
        print(f"  ✓ {eng_count:,} fichiers GKG anglophones identifiés.")

        print("📋 Chargement de la Master File List GDELT (Translingual)...")
        trans_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"
        trans_count = self._fetch_master_list(trans_url, is_translation_list=True)
        print(f"  ✓ {trans_count:,} fichiers GKG translingues identifiés.")
        
        print(f"🎯 Total combiné après dédoublonnage strict : {len(self.valid_gkg_urls):,} fichiers uniques.")

    def generate_months_list(self, start_date, end_date):
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
        
        print(f"\n🚀 PIPELINE MENSUEL MULTILINGUE ULTRA-PERFORMANT (48 CŒURS ENGAGÉS)")
        print(f"📂 Zone Tampon Éphémère : {self.temp_dir}")
        print(f"💾 Base Parquet Finale   : {self.final_dir}\n")

        for idx, month_str in enumerate(months_to_process, 1):
            parquet_filename = f"gdelt_{month_str[:4]}-{month_str[4:]}.parquet"
            parquet_filepath = self.final_dir / parquet_filename
            
            if parquet_filepath.exists():
                print(f"⏭️  [{idx}/{len(months_to_process)}] Mois {month_str} déjà converti. Passé.")
                continue

            print(f"🔄 [{idx}/{len(months_to_process)}] Traitement du mois : {month_str}")
            
            month_urls = [url for url in self.valid_gkg_urls.keys() if url.split('/')[-1].startswith(month_str)]
            total_files = len(month_urls)
            
            if total_files == 0:
                print(f"  ⚠ Aucun fichier pour le mois {month_str}.")
                continue

            # ÉTAPE 1 : Téléchargement parallèle intensif
            print(f"  📥 Téléchargement parallélisé de {total_files} fichiers (.zip) en zone tampon...")
            success, skipped, failed = 0, 0, 0
            
            with ThreadPoolExecutor(max_workers=self.net_workers) as net_executor:
                futures = {net_executor.submit(download_file_worker, url, str(self.temp_dir)): url for url in month_urls}
                for future in as_completed(futures):
                    status = future.result()
                    if status == "DOWNLOADED": success += 1
                    elif status == "EXISTS": skipped += 1
                    else: failed += 1
            print(f"  ✓ Fin téléchargement (Nouveaux: {success} | Présents: {skipped} | Échecs: {failed})")

            # ÉTAPE 2 & 3 : Extraction Parallèle Lourde & Écriture Vectorisée par Lots Réglés
            print(f"  🗜️  Parsing parallèle et Streaming haute performance...")
            
            worker_tasks = []
            for url in month_urls:
                filename = url.split('/')[-1]
                local_path = self.temp_dir / filename
                if local_path.exists() and local_path.stat().st_size > 0:
                    is_translated = self.valid_gkg_urls[url]
                    worker_tasks.append((str(local_path), is_translated))

            writer = None
            schema = None
            articles_comptes = 0
            chunk_dfs = []
            chunk_size_trigger = 200  # Lots élargis à 200 pour maximiser l'usage des 500 Go de RAM

            with ProcessPoolExecutor(max_workers=self.cpu_workers, max_tasks_per_child=20) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, task): task[0] for task in worker_tasks}
                
                for future in as_completed(cpu_futures):
                    res = future.result()
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        chunk_dfs.append(res)
                        
                    if len(chunk_dfs) >= chunk_size_trigger:
                        batch_df = pd.concat(chunk_dfs, ignore_index=True)
                        chunk_dfs.clear()
                        
                        unique_sources = batch_df['SourceCommonName'].dropna().unique()
                        for src in unique_sources:
                            get_or_create_source_id(src)
                        
                        batch_df['SourceCommonName_ID'] = batch_df['SourceCommonName'].map(source_to_id).fillna(0).astype('int32')
                        batch_df.drop(columns=['SourceCommonName'], inplace=True)
                        batch_df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                        
                        table = pa.Table.from_pandas(batch_df, preserve_index=False)
                        articles_comptes += len(batch_df)
                        
                        if writer is None:
                            schema = table.schema
                            writer = pq.ParquetWriter(
                                parquet_filepath, schema, 
                                compression='zstd', compression_level=12, 
                                use_dictionary=True
                            )
                        
                        writer.write_table(table)
                        del batch_df
                        del table

                # Traitement du reliquat final
                if chunk_dfs:
                    batch_df = pd.concat(chunk_dfs, ignore_index=True)
                    chunk_dfs.clear()
                    
                    unique_sources = batch_df['SourceCommonName'].dropna().unique()
                    for src in unique_sources:
                        get_or_create_source_id(src)
                    
                    batch_df['SourceCommonName_ID'] = batch_df['SourceCommonName'].map(source_to_id).fillna(0).astype('int32')
                    batch_df.drop(columns=['SourceCommonName'], inplace=True)
                    batch_df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)
                    
                    table = pa.Table.from_pandas(batch_df, preserve_index=False)
                    articles_comptes += len(batch_df)
                    
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(parquet_filepath, schema, compression='zstd', compression_level=12, use_dictionary=True)
                    
                    writer.write_table(table)
                    del batch_df
                    del table

            save_source_dictionary()

            if writer is not None:
                writer.close()
                print(f"  🟢 Fichier unifié créé : {parquet_filename} ({articles_comptes:,} articles enregistrés).")
            else:
                print(f"  ❌ Échec : Aucun article extrait pour le mois {month_str}")

            # ÉTAPE 4 : Purge immédiate de l'espace tampon
            print(f"  🧹 Libération de l'espace disque de la zone tampon...")
            for f in self.temp_dir.glob('*'):
                try: os.remove(f)
                except Exception: pass
            print(f"  ✨ Zone tampon vidée.\n")

        print("🏆 ENSEMBLE DES MOIS MULTILINGUES EXTRAITS AVEC SUCCÈS !")

if __name__ == "__main__":
    # N'oubliez pas d'initialiser ou de vider gdelt_sources_mapping.json s'il plante au démarrage !
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',    
        final_dir='./gdelt_parquet_db',   
        net_workers=24,       # Pipeline réseau élargi
        cpu_workers=48        # Profil Haute Performance (laisse 16 cœurs aux collègues)
    )
    
    pipeline.process_pipeline(
        start_date='2015-02-19',
        end_date='2015-02-28'
    )