# fileName: gdelt_optimized_pipeline.py
import gdelt
import json
import sqlite3
import gzip
import os
from datetime import datetime, timedelta
from pathlib import Path

class GDELTDailyPipeline:
    """
    Pipeline GDELT conçu pour le traitement jour par jour (Granularité Daily).
    Télécharge, stocke immédiatement, libère la RAM, et passe au jour suivant.
    """
    
    def __init__(self, db_path='gdelt_daily_database.db', output_dir='gdelt_daily_chunks'):
        self.gd = gdelt.gdelt(version=2)
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stats = {'total_days_processed': 0, 'total_articles': 0}
        self.init_database()

    def init_database(self):
        """Initialise SQLite avec le mode WAL pour des écritures ultra-rapides sans blocage."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('PRAGMA journal_mode = WAL;')
        c.execute('PRAGMA synchronous = NORMAL;')
        c.execute('PRAGMA cache_size = -10000;') # Limite stricte à ~10 Mo de RAM pour le cache
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                gkg_record_id TEXT PRIMARY KEY,
                date_published INTEGER,
                source TEXT,
                url TEXT,
                themes TEXT,
                persons TEXT,
                organizations TEXT,
                tone REAL
            ) WITHOUT ROWID;
        ''')
        conn.commit()
        conn.close()
        print(f"✓ Base de données initialisée (Mode Daily) : {self.db_path}")

    def generate_days(self, start_date, end_date):
        """Génère les jours un par un sous forme de strings."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start
        
        while current <= end:
            yield current.strftime('%Y-%m-%d')
            current += timedelta(days=1)

    def fetch_single_day(self, date_str, table='gkg'):
        """Télécharge les données d'un seul jour et les renvoie sous forme de liste."""
        try:
            # GDELT v2 attend le format 'YYYY MM DD'
            gdelt_date_format = date_str.replace('-', ' ')
            
            results_json = self.gd.Search([gdelt_date_format], table=table, output='json', coverage=True)
            
            if not results_json or results_json.strip() == "":
                return []
                
            return json.loads(results_json)
        except Exception as e:
            print(f"  ✗ Erreur de téléchargement pour le {date_str}: {str(e)}")
            return []

    def save_day_to_sqlite(self, date_str, articles):
        """Insère les articles d'une journée dans SQLite au sein d'une seule transaction."""
        if not articles:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('BEGIN TRANSACTION;')
        
        inserted_count = 0
        for article in articles:
            try:
                tone_str = article.get('V2Tone', '')
                tone = float(tone_str.split(',')[0]) if tone_str else 0.0
                
                c.execute('''
                    INSERT OR IGNORE INTO articles
                    (gkg_record_id, date_published, source, url, themes, persons, organizations, tone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.get('GKGRECORDID'),
                    article.get('DATE'),
                    article.get('SourceCommonName'),
                    article.get('DocumentIdentifier'),
                    article.get('Themes'),
                    article.get('V2Persons'),
                    article.get('V2Organizations'),
                    tone
                ))
                inserted_count += 1
            except Exception:
                continue
                
        conn.commit()
        conn.close()
        return inserted_count

    def save_day_to_gzip(self, date_str, articles, table):
        """Sauvegarde les articles d'une journée dans un fichier .json.gz unique."""
        if not articles:
            return
            
        filename = f"{table}_{date_str.replace('-', '')}.json.gz"
        filepath = self.output_dir / filename
        
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False)

    def run_daily_pipeline(self, start_date, end_date, mode='sqlite', table='gkg'):
        """Exécute la boucle jour par jour (Télécharge -> Stocke -> Nettoie -> Suivant)."""
        print(f"\n🚀 Lancement du Pipeline JOURNALIER : {start_date} → {end_date}")
        
        for day in self.generate_days(start_date, end_date):
            print(f"📅 Traitement du {day}...", end='', flush=True)
            
            # 1. Téléchargement de la journée (Seule cette journée occupe la RAM temporairement)
            day_data = self.fetch_single_day(day, table=table)
            articles_count = len(day_data)
            
            if articles_count > 0:
                # 2. Stockage immédiat selon le mode choisi
                if mode == 'sqlite':
                    inserted = self.save_day_to_sqlite(day, day_data)
                    print(f" ✓ {inserted:,} articles insérés en BDD.")
                elif mode == 'files':
                    self.save_day_to_gzip(day, day_data, table)
                    print(f" ✓ Fichier .gz créé ({articles_count:,} articles).")
                
                self.stats['total_articles'] += articles_count
            else:
                print(" ⚠ Aucun article trouvé.")
            
            # 3. LE NETTOYAGE (Crucial) : On vide la variable et on force Python à libérer la RAM
            del day_data
            self.stats['total_days_processed'] += 1

        print(f"\n📊 --- FIN DU TRAITEMENT ---")
        print(f" Nombre de jours traités : {self.stats['total_days_processed']}")
        print(f" Total d'articles cumulés : {self.stats['total_articles']:,}")

if __name__ == "__main__":
    pipeline = GDELTDailyPipeline(db_path='gdelt_2025_daily.db', output_dir='gdelt_daily_zips')
    
    # Lancez l'exécution jour par jour ici
    pipeline.run_daily_pipeline(
        start_date='2025-01-01', 
        end_date='2025-12-31', 
        mode='files',  # 'sqlite' ou 'files'
        table='gkg'
    )