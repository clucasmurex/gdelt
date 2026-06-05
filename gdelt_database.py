import gdelt
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import gzip

class GDELTDatabaseManager:
    """
    Gestion GDELT scalable avec SQLite
    Ideal pour 10+ ans de données (milliards d'articles)
    """
    
    def __init__(self, db_path='gdelt_data.db'):
        self.db_path = db_path
        self.gd = gdelt.gdelt(version=2)
        self.init_database()
    
    def init_database(self):
        """Créer la structure de base de données"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table principale
        c.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gkg_record_id TEXT UNIQUE,
                date_published INTEGER,
                source TEXT,
                url TEXT,
                themes TEXT,
                persons TEXT,
                organizations TEXT,
                tone REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index pour recherches rapides
        c.execute('CREATE INDEX IF NOT EXISTS idx_date ON articles(date_published)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_source ON articles(source)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_persons ON articles(persons)')
        
        # Table de métadonnées
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Base de données initialisée: {self.db_path}")
    
    def insert_articles(self, articles, batch_size=1000):
        """
        Insérer les articles par batch pour optimiser la performance
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        inserted = 0
        skipped = 0
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            for article in batch:
                try:
                    # Extraire le ton (sentiment)
                    tone_str = article.get('V2Tone', '')
                    tone = float(tone_str.split(',')[0]) if tone_str else 0
                    
                    c.execute('''
                        INSERT OR IGNORE INTO articles
                        (gkg_record_id, date_published, source, url, themes, 
                         persons, organizations, tone)
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
                    inserted += 1
                except Exception as e:
                    skipped += 1
            
            conn.commit()
        
        conn.close()
        print(f"  ✓ Insérés: {inserted} | Dupliqués: {skipped}")
        return inserted
    
    def bulk_import(self, start_date, end_date, interval='month'):
        """
        Import massif: date_start → date_end
        """
        print(f"\n{'='*60}")
        print(f"Import GDELT: {start_date} → {end_date}")
        print(f"{'='*60}\n")
        
        from gdelt_bulk_query import GDELTBulkAnalyzer
        analyzer = GDELTBulkAnalyzer()
        date_ranges = analyzer.date_range(start_date, end_date, interval)
        
        total_articles = 0
        
        for i, (start, end) in enumerate(date_ranges, 1):
            print(f"[{i}/{len(date_ranges)}] {start} → {end}")
            
            # Requête GDELT
            try:
                start_gdelt = start.replace('-', ' ')
                end_gdelt = end.replace('-', ' ')
                
                results_json = self.gd.Search(
                    [start_gdelt, end_gdelt],
                    table='gkg',
                    output='json',
                    coverage=True
                )
                
                results = json.loads(results_json)
                
                if results:
                    inserted = self.insert_articles(results)
                    total_articles += inserted
            
            except Exception as e:
                print(f"  ✗ Erreur: {str(e)}")
        
        print(f"\n✓ TERMINÉ: {total_articles:,} articles importés")
    
    def query_by_person(self, person_name):
        """Rechercher tous les articles mentionnant une personne"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT date_published, source, url, tone 
            FROM articles 
            WHERE persons LIKE ? 
            ORDER BY date_published DESC
        ''', (f'%{person_name}%',))
        
        results = c.fetchall()
        conn.close()
        
        return results
    
    def query_by_organization(self, org_name):
        """Rechercher tous les articles mentionnant une organisation"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT date_published, source, url, tone 
            FROM articles 
            WHERE organizations LIKE ? 
            ORDER BY date_published DESC
        ''', (f'%{org_name}%',))
        
        results = c.fetchall()
        conn.close()
        
        return results
    
    def get_stats(self):
        """Obtenir les statistiques de la base"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM articles')
        total = c.fetchone()[0]
        
        c.execute('SELECT MIN(date_published), MAX(date_published) FROM articles')
        min_date, max_date = c.fetchone()
        
        c.execute('SELECT COUNT(DISTINCT source) FROM articles')
        sources = c.fetchone()[0]
        
        c.execute('SELECT AVG(tone) FROM articles')
        avg_tone = c.fetchone()[0]
        
        conn.close()
        
        return {
            'total_articles': total,
            'date_range': (min_date, max_date),
            'unique_sources': sources,
            'avg_tone': avg_tone
        }
    
    def export_to_csv(self, query, output_file, limit=None):
        """Exporter les résultats en CSV pour analyse"""
        import csv
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if limit:
            query += f' LIMIT {limit}'
        
        c.execute(query)
        rows = c.fetchall()
        
        if rows:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'source', 'url', 'tone'])
                writer.writerows(rows)
            
            print(f"✓ Exporté: {output_file} ({len(rows)} lignes)")
        
        conn.close()

# Utilisation
if __name__ == "__main__":
    db_manager = GDELTDatabaseManager('gdelt_2024.db')
    
    # OPTION 1: Import sur 6 mois
    # db_manager.bulk_import('2024-01-01', '2024-06-30', interval='month')
    
    # OPTION 2: Statistiques
    stats = db_manager.get_stats()
    print(f"\nStatistiques:")
    print(f"  Articles: {stats['total_articles']:,}")
    print(f"  Période: {stats['date_range']}")
    print(f"  Sources uniques: {stats['unique_sources']}")
    print(f"  Ton moyen: {stats['avg_tone']:.2f}")
    
    # OPTION 3: Recherches
    # results = db_manager.query_by_person('Trump')
    # print(f"Articles mentionnant Trump: {len(results)}")
    # for date, source, url, tone in results[:5]:
    #     print(f"  {date} - {source} (Ton: {tone})")
    
    # OPTION 4: Export
    # db_manager.export_to_csv(
    #     'SELECT date_published, source, url, tone FROM articles WHERE tone < -5',
    #     'negative_articles.csv'
    # )