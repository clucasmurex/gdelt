# fileName: gdelt_monthly_pipeline.py
import os
import sys
import time
import queue
import shutil
import threading
import urllib.request
import zipfile
import json
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
from collections import defaultdict
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
        return 0
    if source_name in source_to_id:
        return source_to_id[source_name]
    current_id = next_source_id
    source_to_id[source_name] = current_id
    id_to_source[current_id] = source_name
    next_source_id += 1
    return current_id


def parse_zip_worker(zip_filepath):
    """
    Parse un ZIP GKG et renvoie les données sérialisées en Arrow IPC (bytes).

    POURQUOI ARROW IPC ET PAS UN DATAFRAME PANDAS ?
    ProcessPoolExecutor transfère les résultats via pickle. Un DataFrame pandas
    avec des colonnes texte longues (EnhancedThemes, EnhancedLocations…) est
    très coûteux à pickler : ~7-8 secondes de surcoût IPC par fichier, soit
    ~95 % du temps total de "parsing".
    Arrow IPC est le format de sérialisation natif de PyArrow : buffers binaires
    sans copie inutile, désérialisation quasi-instantanée côté réception.

    NB : V2.1TRANSLATIONINFO est à l'index 25 (26 = V2.1EXTRASXML).
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
        25,  # V2.1TRANSLATIONINFO
    ]
    column_names = [
        "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
        "DocumentIdentifier", "EnhancedThemes", "EnhancedLocations",
        "Persons", "Organizations", "Tone_Raw", "TranslationInfo",
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
                    on_bad_lines='skip',
                    engine='pyarrow'
                )

        is_translingual = 1 if '.translation.' in os.path.basename(zip_filepath) else 0
        df['IsTranslingual'] = is_translingual
        df['IsTranslingual'] = df['IsTranslingual'].astype('int8')

        tone_split      = df['Tone_Raw'].str.split(',', expand=False)
        df['Tone']      = pd.to_numeric(tone_split.str[0], errors='coerce').fillna(0.0).astype('float16')
        df['WordCount'] = pd.to_numeric(tone_split.str[6], errors='coerce').fillna(0).astype('int32')
        df.drop(columns=['Tone_Raw'], inplace=True)

        df['SourceCollectionIdentifier'] = pd.to_numeric(
            df['SourceCollectionIdentifier'], errors='coerce'
        ).fillna(0).astype('int8')

        for col in ['GKGRECORDID', 'DATE', 'DocumentIdentifier',
                    'EnhancedThemes', 'EnhancedLocations',
                    'Persons', 'Organizations', 'TranslationInfo']:
            df[col] = df[col].fillna('')

        df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)

        if df.empty:
            return None

        # Conversion en Arrow puis sérialisation IPC.
        # Pickle d'un DataFrame string → plusieurs secondes de surcoût.
        # Arrow IPC du même contenu → quelques dizaines de ms.
        table = pa.Table.from_pandas(df, preserve_index=False)
        del df
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as ipc_writer:
            ipc_writer.write_table(table)
        return sink.getvalue()   # pa.Buffer, sérialisé via pickle comme bytes bruts

    except Exception:
        return None


def download_file_worker(url, temp_dir, timeout=15, max_retries=3):
    filename = url.split('/')[-1]
    filepath = os.path.join(temp_dir, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return "EXISTS"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, \
                 open(filepath, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return "DOWNLOADED"
            return "FAILED"
        except Exception:
            if os.path.exists(filepath):
                try: os.remove(filepath)
                except Exception: pass
            time.sleep(0.5 * (attempt + 1))
    return "FAILED"


def writer_thread_fn(write_queue, parquet_filepath, zstd_level, row_group_size, result_queue):
    """
    Thread dédié à l'écriture Parquet, découplé du thread principal.

    Reçoit des pa.Table via write_queue, les accumule jusqu'à row_group_size
    lignes, puis flush (concat + write) en une passe.
    Sentinel None = fin du flux.
    """
    writer        = None
    buffer_tables = []
    buffer_rows   = 0
    articles      = 0
    failed        = 0

    def flush():
        nonlocal writer, articles, failed
        if not buffer_tables:
            return
        try:
            merged = pa.concat_tables(buffer_tables)
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_filepath,
                    merged.schema,
                    compression='zstd',
                    compression_level=zstd_level,
                    use_dictionary=True,
                    write_batch_size=65_536,
                )
            writer.write_table(merged)
            articles += merged.num_rows
            del merged
        except Exception as e:
            failed += 1
            print(f"    ⚠️ Flush ignoré (schéma incompatible) : {e}")
        buffer_tables.clear()

    while True:
        table = write_queue.get()
        if table is None:
            flush()
            break
        buffer_tables.append(table)
        buffer_rows += table.num_rows
        if buffer_rows >= row_group_size:
            flush()
            buffer_rows = 0

    if writer is not None:
        writer.close()

    result_queue.put((articles, failed))


class GDELTRollingPipeline:
    def __init__(self, temp_dir, final_dir, net_workers=32, cpu_workers=50,
                 zstd_compression_level=3, row_group_size=500_000,
                 write_queue_maxsize=30):
        self.temp_dir = Path(temp_dir)
        self.final_dir = Path(final_dir)
        self.net_workers = net_workers
        self.cpu_workers = cpu_workers
        self.zstd_compression_level = zstd_compression_level
        self.row_group_size = row_group_size
        self.write_queue_maxsize = write_queue_maxsize
        self.urls_by_month = defaultdict(list)

        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.final_dir.mkdir(exist_ok=True, parents=True)
        load_source_dictionary()

    def load_master_list(self):
        print("📋 Chargement des Master File Lists GDELT (standard + translingue)...")
        master_lists = [
            ("standard",    "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"),
            ("translingue", "http://data.gdeltproject.org/gdeltv2/masterfilelist-translation.txt"),
        ]
        total = 0
        any_success = False
        for label, master_url in master_lists:
            count_before = total
            try:
                req = urllib.request.Request(master_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    for line in response:
                        line_str = line.decode('utf-8').strip()
                        if not line_str: continue
                        url = line_str.split(' ')[-1]
                        if 'gkg.csv.zip' in url:
                            month_key = url.split('/')[-1][:6]
                            self.urls_by_month[month_key].append(url)
                            total += 1
                any_success = True
                print(f"  ✓ Liste {label} : {total - count_before:,} fichiers GKG.")
            except Exception as e:
                print(f"  ⚠️ Impossible de charger la liste {label} : {e}")

        if not any_success:
            print("💥 Erreur critique : aucune master list chargée.")
            sys.exit(1)
        print(f"✓ Total combiné : {total:,} fichiers GKG sur {len(self.urls_by_month)} mois.")

    def generate_months_list(self, start_date, end_date):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end   = datetime.strptime(end_date,   '%Y-%m-%d')
        months, current = [], datetime(start.year, start.month, 1)
        while current <= end:
            months.append(current.strftime('%Y%m'))
            nxt = current + timedelta(days=32)
            current = datetime(nxt.year, nxt.month, 1)
        return months

    def process_pipeline(self, start_date, end_date):
        self.load_master_list()
        months_to_process = self.generate_months_list(start_date, end_date)

        print(f"\n🚀 PIPELINE MENSUEL — Arrow IPC + producteur-consommateur")
        print(f"   zstd niv.{self.zstd_compression_level} | "
              f"row_group {self.row_group_size:,} | "
              f"{self.cpu_workers} CPU | {self.net_workers} net")
        print(f"📂 Zone Tampon : {self.temp_dir}")
        print(f"💾 Parquet DB  : {self.final_dir}\n")

        for idx, month_str in enumerate(months_to_process, 1):
            parquet_filename = f"gdelt_{month_str[:4]}-{month_str[4:]}.parquet"
            parquet_filepath = self.final_dir / parquet_filename

            if parquet_filepath.exists():
                print(f"⏭️  [{idx}/{len(months_to_process)}] {month_str} déjà converti.")
                continue

            print(f"🔄 [{idx}/{len(months_to_process)}] Traitement du mois : {month_str}")
            month_urls  = self.urls_by_month.get(month_str, [])
            total_files = len(month_urls)
            if total_files == 0:
                print(f"  ⚠ Aucun fichier pour {month_str}.")
                continue

            # ── ÉTAPE 1 : Téléchargement ──────────────────────────────────────
            print(f"  📥 Téléchargement de {total_files} fichiers (.zip)...")
            t0 = time.time()
            success, skipped, failed, done = 0, 0, 0, 0
            progress_step = max(1, total_files // 20)
            with ThreadPoolExecutor(max_workers=self.net_workers) as net_executor:
                futures = {net_executor.submit(download_file_worker, url, str(self.temp_dir)): url
                           for url in month_urls}
                for future in as_completed(futures):
                    status = future.result()
                    done += 1
                    if status == "DOWNLOADED": success += 1
                    elif status == "EXISTS":   skipped += 1
                    else:                      failed  += 1
                    if done % progress_step == 0 or done == total_files:
                        print(f"    ... {done}/{total_files} (↓{success} | ✓{skipped} | ✗{failed})")
            print(f"  ✓ Téléchargement en {time.time() - t0:.0f}s")

            # ── ÉTAPE 2 : Parsing parallèle (Arrow IPC) + écriture asynchrone ─
            print(f"  🗜️  Parsing ({self.cpu_workers} workers, Arrow IPC) + écriture asynchrone...")
            t1 = time.time()
            local_zips        = list(self.temp_dir.glob('*.zip'))
            total_zips        = len(local_zips)
            progress_step_cpu = max(1, total_zips // 20)

            write_queue  = queue.Queue(maxsize=self.write_queue_maxsize)
            result_queue = queue.Queue()

            wt = threading.Thread(
                target=writer_thread_fn,
                args=(write_queue, parquet_filepath,
                      self.zstd_compression_level, self.row_group_size,
                      result_queue),
                daemon=True,
            )
            wt.start()

            chunks_done = 0
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, str(p)): p
                               for p in local_zips}

                for future in as_completed(cpu_futures):
                    buf = future.result()   # pa.Buffer Arrow IPC, ou None
                    chunks_done += 1

                    if buf is not None:
                        # Désérialisation Arrow IPC : quasi-instantanée (zero-copy)
                        reader = pa.ipc.open_stream(buf)
                        table  = reader.read_all()
                        del buf

                        # Mapping source → ID (dict Python, thread principal)
                        src_col = table.column('SourceCommonName')
                        unique_sources = src_col.unique().to_pylist()
                        for src in unique_sources:
                            if src:
                                get_or_create_source_id(src)

                        # Remplacement de la colonne texte par les IDs numériques
                        ids = pa.array(
                            [source_to_id.get(s, 0) if s else 0
                             for s in src_col.to_pylist()],
                            type=pa.int32()
                        )
                        table = table.set_column(
                            table.schema.get_field_index('SourceCommonName'),
                            'SourceCommonName_ID',
                            ids
                        )

                        write_queue.put(table)

                    if chunks_done % progress_step_cpu == 0 or chunks_done == total_zips:
                        print(f"    ... {chunks_done}/{total_zips} parsés "
                              f"| queue writer : {write_queue.qsize()} tables")

            write_queue.put(None)   # sentinel
            wt.join()

            articles_comptes, chunks_failed = result_queue.get()

            # ── ÉTAPE 3 : Sauvegarde du dictionnaire ──────────────────────────
            save_source_dictionary()

            if articles_comptes > 0:
                print(f"  🟢 {parquet_filename} — "
                      f"{articles_comptes:,} articles en {time.time() - t1:.0f}s.")
                if chunks_failed:
                    print(f"  ⚠️ {chunks_failed} chunk(s) ignoré(s).")
            else:
                print(f"  ❌ Aucun article extrait pour {month_str}")

            # ── ÉTAPE 4 : Purge de la zone tampon ─────────────────────────────
            print(f"  🧹 Purge zone tampon...")
            for f in self.temp_dir.glob('*'):
                try: os.remove(f)
                except Exception: pass
            print(f"  ✨ Zone tampon vidée.\n")

        print("🏆 PIPELINE EXÉCUTÉ AVEC SUCCÈS !")


if __name__ == "__main__":
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',
        final_dir='./gdelt_parquet_dbv2',
        net_workers=32,
        cpu_workers=50,
        zstd_compression_level=6,
        row_group_size=500_000,
        write_queue_maxsize=30,
    )

    pipeline.process_pipeline(
        start_date='2025-06-01',
        end_date='2025-06-30',
    )