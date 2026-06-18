# fileName: gdelt_monthly_pipeline.py
import os
import sys
import time
import shutil
import urllib.request
import zipfile
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
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
    Ouvre un ZIP local en mémoire de manière optimisée.
    Ne charge que les colonnes réellement utilisées en aval, pour économiser
    RAM et temps CPU (la colonne TranslationInfo a été retirée car jamais
    utilisée par la suite du pipeline).
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

                # --- OPTIMISATION DES TYPES PRIMITIFS ---
                tone_split = df['Tone_Raw'].str.split(',', expand=False)
                df['Tone'] = tone_split.str[0]
                df['Tone'] = pd.to_numeric(df['Tone'], errors='coerce').fillna(0.0).astype('float16')

                df['WordCount'] = tone_split.str[6]
                df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce').fillna(0).astype('int32')
                df.drop(columns=['Tone_Raw'], inplace=True)

                # V2SOURCECOLLECTIONIDENTIFIER est un petit entier (1, 2, 3...).
                # On le convertit directement en int8 plutôt qu'en 'category' :
                # un dtype 'category' peut produire un type Arrow "dictionary"
                # dont l'index (int8/int16) varie selon le nombre de catégories
                # vues dans CHAQUE chunk, ce qui peut faire planter le
                # ParquetWriter par incompatibilité de schéma entre deux écritures.
                df['SourceCollectionIdentifier'] = pd.to_numeric(
                    df['SourceCollectionIdentifier'], errors='coerce'
                ).fillna(0).astype('int8')

                # Toutes les colonnes texte restantes doivent être de vraies
                # chaînes (jamais NaN), sinon un chunk avec une colonne
                # entièrement vide ferait inférer un type Arrow "null" et
                # casserait l'écriture Parquet dès le chunk suivant.
                for col in ['GKGRECORDID', 'DATE', 'DocumentIdentifier',
                            'EnhancedThemes', 'EnhancedLocations',
                            'Persons', 'Organizations']:
                    df[col] = df[col].fillna('')

                # SÉCURITÉ ANTI-DOUBLONS : nettoyage local immédiat pour chaque lot de 15 min
                df.drop_duplicates(subset=['GKGRECORDID'], inplace=True)

        return df
    except Exception:
        return None


def download_file_worker(url, temp_dir, timeout=15, max_retries=3):
    """
    Télécharge un ZIP unitaire avec urllib natif (pas de sous-processus).
    Beaucoup moins de surcoût que de lancer un binaire externe par fichier,
    et les erreurs réseau remontent immédiatement au lieu de dépendre de la
    présence de `wget` sur la machine.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(temp_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return "EXISTS"

    req = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )

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


class GDELTRollingPipeline:
    def __init__(self, temp_dir, final_dir, net_workers=16, cpu_workers=24,
                 zstd_compression_level=6):
        self.temp_dir = Path(temp_dir)
        self.final_dir = Path(final_dir)
        self.net_workers = net_workers
        self.cpu_workers = cpu_workers
        self.zstd_compression_level = zstd_compression_level
        self.urls_by_month = defaultdict(list)

        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.final_dir.mkdir(exist_ok=True, parents=True)
        load_source_dictionary()  # Chargement initial du JSON externe au démarrage

    def load_master_list(self):
        """Récupère la liste officielle des fichiers existants sur GDELT et les
        regroupe directement par mois (évite de refiltrer un set global à
        chaque itération de la boucle principale)."""
        print("📋 Chargement de la Master File List GDELT...")
        master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        total = 0
        try:
            req = urllib.request.Request(master_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    url = line_str.split(' ')[-1]
                    if 'gkg.csv.zip' in url:
                        filename = url.split('/')[-1]
                        month_key = filename[:6]  # YYYYMM
                        self.urls_by_month[month_key].append(url)
                        total += 1
            print(f"✓ {total:,} fichiers GKG répertoriés, répartis sur {len(self.urls_by_month)} mois.")
        except Exception as e:
            print(f"💥 Erreur critique Master List (vérifiez votre connectivité réseau) : {e}")
            sys.exit(1)

    def generate_months_list(self, start_date, end_date):
        """Génère la liste des chaînes YYYYMM de manière robuste.

        ATTENTION : la granularité de ce pipeline est MENSUELLE. Si
        start_date/end_date ne couvrent que quelques jours dans un même mois,
        c'est tout de même le mois entier qui sera téléchargé et traité
        (c'est voulu par design, mais ça explique un temps d'exécution bien
        plus long qu'attendu si vous pensiez ne traiter que quelques jours)."""
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

        print(f"\n🚀 PIPELINE MENSUEL STREAMING (DICTIONNAIRE INT-JSON ENGAGÉ)")
        print(f"📂 Zone Tampon Éphémère : {self.temp_dir}")
        print(f"💾 Base Parquet Finale   : {self.final_dir}\n")

        for idx, month_str in enumerate(months_to_process, 1):
            parquet_filename = f"gdelt_{month_str[:4]}-{month_str[4:]}.parquet"
            parquet_filepath = self.final_dir / parquet_filename

            if parquet_filepath.exists():
                print(f"⏭️  [{idx}/{len(months_to_process)}] Mois {month_str} déjà converti. Passé.")
                continue

            print(f"🔄 [{idx}/{len(months_to_process)}] Traitement du mois : {month_str}")

            month_urls = self.urls_by_month.get(month_str, [])
            total_files = len(month_urls)

            if total_files == 0:
                print(f"  ⚠ Aucun fichier pour le mois {month_str}.")
                continue

            # ÉTAPE 1 : Téléchargement dans la zone tampon
            print(f"  📥 Téléchargement de {total_files} fichiers (.zip) en zone tampon...")
            t0 = time.time()
            success, skipped, failed, done = 0, 0, 0, 0
            progress_step = max(1, total_files // 20)

            with ThreadPoolExecutor(max_workers=self.net_workers) as net_executor:
                futures = {net_executor.submit(download_file_worker, url, str(self.temp_dir)): url for url in month_urls}
                for future in as_completed(futures):
                    status = future.result()
                    done += 1
                    if status == "DOWNLOADED": success += 1
                    elif status == "EXISTS": skipped += 1
                    else: failed += 1
                    if done % progress_step == 0 or done == total_files:
                        print(f"    ... {done}/{total_files} traités "
                              f"(Nouveaux: {success} | Présents: {skipped} | Échecs: {failed})")

            print(f"  ✓ Fin téléchargement en {time.time() - t0:.0f}s "
                  f"(Nouveaux: {success} | Présents: {skipped} | Échecs: {failed})")

            # ÉTAPE 2 & 3 : Extraction, Dédoublonnage, Encodage des sources & Streaming direct
            print(f"  🗜️  Parsing, Mapping ID et Streaming direct sur disque via {self.cpu_workers} cœurs...")
            t1 = time.time()
            local_zips = list(self.temp_dir.glob('*.zip'))
            total_zips = len(local_zips)
            progress_step_cpu = max(1, total_zips // 20)

            writer = None
            schema = None
            articles_comptes = 0
            chunks_done = 0
            chunks_failed = 0

            # Pas de max_tasks_per_child ici : le pool est déjà recréé à chaque
            # mois, donc forcer un redémarrage des processus toutes les
            # quelques tâches ne fait que payer le coût (ré-import de pandas /
            # pyarrow à chaque fois) sans bénéfice réel.
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as cpu_executor:
                cpu_futures = {cpu_executor.submit(parse_zip_worker, str(p)): p for p in local_zips}

                for future in as_completed(cpu_futures):
                    res = future.result()
                    chunks_done += 1

                    if isinstance(res, pd.DataFrame) and not res.empty:
                        # --- ENCODAGE DYNAMIQUE ET SÉCURISÉ DES SOURCES ---
                        unique_sources = res['SourceCommonName'].dropna().unique()
                        for src in unique_sources:
                            get_or_create_source_id(src)

                        res['SourceCommonName_ID'] = res['SourceCommonName'].map(source_to_id).fillna(0).astype('int32')
                        res.drop(columns=['SourceCommonName'], inplace=True)

                        table = pa.Table.from_pandas(res, preserve_index=False)
                        del res

                        try:
                            if writer is None:
                                schema = table.schema
                                writer = pq.ParquetWriter(
                                    parquet_filepath,
                                    schema,
                                    compression='zstd',
                                    compression_level=self.zstd_compression_level,
                                    use_dictionary=True
                                )
                            writer.write_table(table)
                            articles_comptes += table.num_rows
                        except Exception as e:
                            chunks_failed += 1
                            print(f"    ⚠️ Chunk ignoré (incompatibilité de schéma) : {e}")

                        del table

                    if chunks_done % progress_step_cpu == 0 or chunks_done == total_zips:
                        print(f"    ... {chunks_done}/{total_zips} fichiers traités "
                              f"({articles_comptes:,} articles écrits jusqu'ici)")

            # ÉTAPE 3.5 : Sauvegarde physique du dictionnaire JSON mis à jour pour ce mois-ci
            save_source_dictionary()

            # Fermeture propre du fichier Parquet mensuel
            if writer is not None:
                writer.close()
                print(f"  🟢 Fichier créé en streaming : {parquet_filename} "
                      f"({articles_comptes:,} articles enregistrés en {time.time() - t1:.0f}s).")
                if chunks_failed:
                    print(f"  ⚠️ {chunks_failed} chunk(s) ignoré(s) pour incompatibilité de schéma.")
            else:
                print(f"  ❌ Échec : Aucun article extrait pour le mois {month_str}")

            # ÉTAPE 4 : Purge immédiate de l'espace tampon
            print(f"  🧹 Libération de l'espace disque de la zone tampon...")
            for f in self.temp_dir.glob('*'):
                try: os.remove(f)
                except Exception: pass
            print(f"  ✨ Zone tampon vidée.\n")

        print("🏆 PIPELINE EXÉCUTÉ AVEC SUCCÈS !")


if __name__ == "__main__":
    pipeline = GDELTRollingPipeline(
        temp_dir='./gdelt_buffer_temp',
        final_dir='./gdelt_parquet_db',
        net_workers=16,       # urllib natif = pas besoin d'être aussi conservateur qu'avec wget
        cpu_workers=24,       # Courtois avec les collègues connectés sur la machine
        zstd_compression_level=6,  # Bon compromis vitesse/taille (12 était inutilement agressif)
    )

    pipeline.process_pipeline(
        start_date='2015-02-19',
        end_date='2026-06-17'   # ⚠️ rappel : ceci traite TOUT le mois de avril 2015, pas seulement 3 jours
    )