import gdelt
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import gzip

class GDELTBulkAnalyzer:
    """
    Requête GDELT par plages de dates avec gestion des flux massifs
    """
    
    def __init__(self, output_dir='gdelt_data'):
        self.gd = gdelt.gdelt(version=2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stats = {'total_articles': 0, 'files_created': 0}
    
    def date_range(self, start_date, end_date, interval='month'):
        """
        Génère une liste de plages de dates
        interval: 'day', 'week', 'month', 'quarter'
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        ranges = []
        current = start
        
        while current <= end:
            if interval == 'day':
                next_date = current + timedelta(days=1)
            elif interval == 'week':
                next_date = current + timedelta(weeks=1)
            elif interval == 'month':
                # Aller au 1er du mois suivant
                if current.month == 12:
                    next_date = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    next_date = current.replace(month=current.month + 1, day=1)
            elif interval == 'quarter':
                next_month = current.month + 3
                if next_month > 12:
                    next_date = current.replace(year=current.year + 1, month=next_month - 12, day=1)
                else:
                    next_date = current.replace(month=next_month, day=1)
            
            # Ne pas dépasser la date de fin
            if next_date > end:
                next_date = end + timedelta(days=1)
            
            ranges.append((current.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
            current = next_date
        
        return ranges
    
    def query_gdelt_range(self, start_date, end_date, table='gkg'):
        """
        Requête GDELT pour une plage de dates
        Format: YYYY-MM-DD
        """
        try:
            # Convertir au format GDELT (YYYY MM DD)
            start_gdelt = start_date.replace('-', ' ')
            end_gdelt = end_date.replace('-', ' ')
            
            print(f"  Requête: {start_gdelt} → {end_gdelt}...", end=' ')
            
            results_json = self.gd.Search(
                [start_gdelt, end_gdelt],
                table=table,
                output='json',
                coverage=True
            )
            
            results = json.loads(results_json)
            print(f"✓ {len(results)} articles")
            return results
            
        except Exception as e:
            print(f"✗ Erreur: {str(e)}")
            return []
    
    def save_chunk(self, data, chunk_name, compress=True):
        """
        Sauvegarde un chunk de données (JSON ou compressé)
        """
        filepath = self.output_dir / f"{chunk_name}.json"
        
        # Sauvegarder en JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        # Compresser si demandé
        if compress:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(filepath)
            filepath = f"{filepath}.gz"
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        else:
            size = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"    → Sauvegardé: {filepath} ({size:.2f} MB)")
        self.stats['files_created'] += 1
        return filepath
    
    def bulk_query(self, start_date, end_date, interval='month', table='gkg', compress=True):
        """
        Requête GDELT sur une large période avec sauvegarde par chunks
        
        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            interval: 'day', 'week', 'month', 'quarter'
            table: 'gkg', 'events', 'mentions'
            compress: Compresser les fichiers JSON
        """
        print(f"\n{'='*60}")
        print(f"GDELT Bulk Query: {start_date} → {end_date}")
        print(f"Intervalle: {interval} | Table: {table}")
        print(f"{'='*60}\n")
        
        date_ranges = self.date_range(start_date, end_date, interval)
        print(f"Total de requêtes: {len(date_ranges)}\n")
        
        all_results = []
        
        for i, (start, end) in enumerate(date_ranges, 1):
            print(f"[{i}/{len(date_ranges)}] {start}")
            
            # Requête pour cette plage
            results = self.query_gdelt_range(start, end, table=table)
            
            if results:
                all_results.extend(results)
                self.stats['total_articles'] += len(results)
                
                # Sauvegarder par chunk (tous les X articles ou tous les mois)
                if len(all_results) > 50000 or i == len(date_ranges):
                    chunk_name = f"{table}_{start.replace('-', '')}_{end.replace('-', '')}"
                    self.save_chunk(all_results, chunk_name, compress=compress)
                    all_results = []
        
        print(f"\n{'='*60}")
        print(f"✓ TERMINÉ")
        print(f"Total articles: {self.stats['total_articles']:,}")
        print(f"Fichiers créés: {self.stats['files_created']}")
        print(f"{'='*60}\n")
    
    def get_sample_articles(self, num_samples=5):
        """
        Charge et affiche des exemples d'articles des fichiers sauvegardés
        """
        json_files = list(self.output_dir.glob('*.json.gz')) + list(self.output_dir.glob('*.json'))
        
        if not json_files:
            print("Aucun fichier trouvé")
            return
        
        print(f"Fichiers trouvés: {len(json_files)}\n")
        
        for json_file in json_files[:1]:  # Charger le premier fichier
            print(f"Fichier: {json_file.name}")
            
            # Décompresser si nécessaire
            if str(json_file).endswith('.gz'):
                with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            print(f"Articles dans ce fichier: {len(data)}\n")
            
            # Afficher quelques articles
            for i, article in enumerate(data[:num_samples], 1):
                print(f"Article {i}:")
                print(f"  Source: {article.get('SourceCommonName')}")
                print(f"  Date: {article.get('DATE')}")
                print(f"  Personnes: {article.get('V2Persons', 'N/A')}")
                print(f"  Organisations: {article.get('V2Organizations', 'N/A')}")
                print()

# Utilisation
if __name__ == "__main__":
    analyzer = GDELTBulkAnalyzer(output_dir='gdelt_data_2024')
    
    # Exemple 1: Requête par mois (6 mois)
    analyzer.bulk_query(
        start_date='2024-01-01',
        end_date='2024-06-30',
        interval='month',
        table='gkg',
        compress=True
    )
    
    # Exemple 2: Requête par trimestre (1 an)
    # analyzer.bulk_query(
    #     start_date='2023-01-01',
    #     end_date='2023-12-31',
    #     interval='quarter',
    #     table='gkg'
    # )
    
    # Afficher des exemples
    analyzer.get_sample_articles(num_samples=3)