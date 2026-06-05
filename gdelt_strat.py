# ============================================================================
# OPTION 1: PETITS VOLUMES (< 1 an) → JSON + fichiers compressés
# ============================================================================

# Avantages:
# ✓ Simple, pas de dépendances DB
# ✓ Idéal pour < 5 millions d'articles
# ✓ Facile à partager les fichiers

from gdelt_bulk_query import GDELTBulkAnalyzer

analyzer = GDELTBulkAnalyzer(output_dir='gdelt_2024')
analyzer.bulk_query(
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='month',  # 12 requêtes
    compress=True      # Gzip pour économiser l'espace
)

# Résultat: 12 fichiers JSON.gz (~100-500 MB chacun)


# ============================================================================
# OPTION 2: VOLUMES MOYENS (1-5 ans) → SQLite (RECOMMANDÉ)
# ============================================================================

# Avantages:
# ✓ Recherches rapides avec indexation
# ✓ Requêtes SQL complexes
# ✓ Gestion efficace (< 100 GB)
# ✓ Pas de serveur externe

from gdelt_database import GDELTDatabaseManager

db = GDELTDatabaseManager('gdelt_2022_2024.db')

# Import sur 3 ans par trimestre (12 requêtes)
db.bulk_import('2022-01-01', '2024-12-31', interval='quarter')

# Recherches
trump_articles = db.query_by_person('Trump')
print(f"Articles sur Trump: {len(trump_articles)}")

# Exporter des résultats
db.export_to_csv(
    'SELECT date_published, source, tone FROM articles WHERE tone < -8',
    'negative_news.csv'
)


# ============================================================================
# OPTION 3: VOLUMES MASSIFS (5-20+ ans) → PostgreSQL + Pandas
# ============================================================================

# Pour 10+ ans: ~1-50 milliards d'articles
# → Besoin d'une vraie base de données distribuée

# Installation:
# pip install psycopg2-binary pandas sqlalchemy

import pandas as pd
import sqlalchemy as sa

# Configuration PostgreSQL (sur serveur distant si possible)
db_url = 'postgresql://user:password@localhost/gdelt'
engine = sa.create_engine(db_url)

def import_gdelt_to_postgres(start_date, end_date, interval='quarter'):
    """Import massif avec chunks Pandas"""
    from gdelt_bulk_query import GDELTBulkAnalyzer
    import gzip
    
    analyzer = GDELTBulkAnalyzer()
    date_ranges = analyzer.date_range(start_date, end_date, interval)
    
    for start, end in date_ranges:
        print(f"Traitement: {start} → {end}")
        
        # Requête GDELT
        start_gdelt = start.replace('-', ' ')
        end_gdelt = end.replace('-', ' ')
        
        gd = __import__('gdelt').gdelt(version=2)
        results_json = gd.Search(
            [start_gdelt, end_gdelt],
            table='gkg',
            output='json',
            coverage=True
        )
        
        results = __import__('json').loads(results_json)
        
        # Convertir en DataFrame
        df = pd.json_normalize(results)
        
        # Nettoyer les colonnes
        df = df[['GKGRECORDID', 'DATE', 'SourceCommonName', 'DocumentIdentifier', 
                 'V2Persons', 'V2Organizations', 'V2Tone']]
        df.columns = ['gkg_id', 'date', 'source', 'url', 'persons', 'orgs', 'tone']
        
        # Extraire le ton (premier nombre)
        df['tone'] = df['tone'].str.split(',').str[0].astype(float)
        
        # Insérer par batch
        df.to_sql('articles', engine, if_exists='append', index=False, chunksize=10000)
        print(f"  ✓ {len(df):,} articles importés")

# Lancer l'import
import_gdelt_to_postgres('2014-01-01', '2024-12-31', interval='quarter')


# ============================================================================
# OPTION 4: ULTRA-MASSIF (20+ ans) → Parquet + Spark (pour data science)
# ============================================================================

# Installation:
# pip install pyspark pyarrow

from pyspark.sql import SparkSession
import json

def import_gdelt_to_parquet(start_date, end_date):
    """Pour 20+ ans: utiliser Apache Spark"""
    spark = SparkSession.builder.appName("GDELT").getOrCreate()
    
    from gdelt_bulk_query import GDELTBulkAnalyzer
    analyzer = GDELTBulkAnalyzer()
    date_ranges = analyzer.date_range(start_date, end_date, interval='quarter')
    
    all_data = []
    
    for start, end in date_ranges:
        print(f"Requête: {start} → {end}")
        
        start_gdelt = start.replace('-', ' ')
        end_gdelt = end.replace('-', ' ')
        
        gd = __import__('gdelt').gdelt(version=2)
        results_json = gd.Search([start_gdelt, end_gdelt], table='gkg', output='json')
        results = json.loads(results_json)
        
        all_data.extend(results)
        
        # Quand on atteint 100k articles, sauvegarder un chunk Parquet
        if len(all_data) > 100000:
            df = spark.createDataFrame(all_data)
            df.write.mode('append').parquet(f'gdelt/{start}_{end}.parquet')
            all_data = []

# Parquet: format columnar optimisé pour analytics
# + Compression automatique
# + Intégration Spark/Pandas/DuckDB
# + Excellent pour aggregations et filtering


# ============================================================================
# STRATÉGIE RÉSUMÉE
# ============================================================================

"""
📊 CHOIX PAR VOLUME:

1-100 millions articles (< 1 an):
   └─ JSON compressé (fichiers)
   └─ Traitement: Pandas

100M - 1 milliard articles (1-5 ans):
   └─ SQLite
   └─ Traitement: SQL + Pandas

1-10 milliards articles (5-20 ans):
   └─ PostgreSQL ou MySQL
   └─ Partitioning par année/mois
   └─ Traitement: SQL + Pandas/Polars

10B+ articles (20+ ans):
   └─ Parquet + Spark/DuckDB
   └─ Cloud storage (S3, GCS)
   └─ Data warehouse (BigQuery, Redshift)
   └─ Traitement: Spark SQL, PySpark


⚡ OPTIMISATIONS CLÉS:

1. Batcher les requêtes:
   - Par jour/semaine pour petits volumes
   - Par mois/trimestre pour moyens volumes
   - Par trimestre/année pour gros volumes

2. Filtrer à la source:
   - Limiter par pays/langue
   - Filtrer les sources (médias importants)
   - Exclure les doublons

3. Stocker intelligemment:
   - Compresser (Gzip, Parquet)
   - Indexer les colonnes fréquemment cherchées
   - Partitionner par date

4. Paralléliser:
   - Requêtes GDELT en parallèle (attention rate limits!)
   - Traitement Spark
   - Uploads batch BD
"""

# ============================================================================
# EXEMPLE COMPLET: 5 ANS avec SQLite
# ============================================================================

if __name__ == "__main__":
    
    # 1. Setup
    from gdelt_database import GDELTDatabaseManager
    db = GDELTDatabaseManager('gdelt_full_5years.db')
    
    # 2. Import par trimestres (20 requêtes seulement)
    print("📥 Importation 5 ans de données...")
    db.bulk_import('2020-01-01', '2020-12-31', interval='quarter')
    
    # 3. Statistiques
    print("\n📊 Statistiques:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 4. Analyses rapides
    print("\n🔍 Recherches:")
    
    # Top organisations mentionnées
    conn = __import__('sqlite3').connect('gdelt_full_5years.db')
    c = conn.cursor()
    c.execute('''
        SELECT organizations, COUNT(*) as count 
        FROM articles 
        WHERE organizations IS NOT NULL
        GROUP BY organizations 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    print("Top organisations:")
    for org, count in c.fetchall():
        print(f"  {org}: {count:,} articles")
    conn.close()
    
    # 5. Export
    db.export_to_csv(
        '''SELECT date_published, source, url, tone 
           FROM articles 
           ORDER BY date_published DESC''',
        'gdelt_full_5years.csv',
        limit=100000
    )
    
    print("\n✅ Analyse terminée!")