# fileName: gdelt_wget_archiver.py
import os
import sys
import urllib.request
import re
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class GDELTWgetArchiver:
    def __init__(self, output_dir='gdelt_raw_archive', max_workers=8):
        # On baisse à 8 ou 12 workers max. Comme wget ne lâche jamais l'affaire, 
        # mettre trop de workers saturerait le serveur GDELT inutilement.
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers
        self.valid_urls = set()

    def load_master_file_list(self):
        """Télécharge le registre officiel pour éviter de chercher dans le vide."""
        print("📋 Chargement de la Master File List GDELT...")
        master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        try:
            req = urllib.request.Request(master_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    url = line_str.split(' ')[-1]
                    if 'gkg.csv.zip' in url:
                        self.valid_urls.add(url)
            print(f"✓ {len(self.valid_urls):,} URLs GKG valides en mémoire.")
        except Exception as e:
            print(f"💥 Erreur master list : {e}")
            sys.exit(1)

    def filter_urls(self, start_date, end_date):
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        filtered = []
        for url in self.valid_urls:
            filename = url.split('/')[-1]
            match = re.match(r'^([0-9]{8})', filename)
            if match:
                file_date = datetime.strptime(match.group(1), '%Y%m%d')
                if start_obj <= file_date <= end_obj:
                    filtered.append(url)
        return sorted(filtered)

    def wget_download_worker(self, url):
        """Délègue le téléchargement à l'utilitaire système wget avec options de retry."""
        filename = url.split('/')[-1]
        filepath = os.path.join(self.output_dir, filename)

        # Si le fichier est déjà là et valide, on passe
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return "EXISTS", filename

        # Construction de la commande wget ultra-robuste :
        # --tries=15 : Réessaie 15 fois si le serveur timeout ou coupe
        # --waitretry=3 : Attend 3 secondes entre chaque essai pour laisser le serveur respirer
        # --quiet : N'inonde pas le terminal de logs inutiles
        cmd = [
            "wget",
            "--tries=15",
            "--waitretry=3",
            "--quiet",
            "-O", filepath,
            url
        ]

        try:
            # Exécute wget au niveau du système
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            
            if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return "DOWNLOADED", filename
            else:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return "FAILED", filename
        except subprocess.TimeoutExpired:
            if os.path.exists(filepath):
                os.remove(filepath)
            return "TIMEOUT", filename

    def download_history(self, start_date, end_date):
        self.load_master_file_list()
        urls = self.filter_urls(start_date, end_date)
        total_files = len(urls)
        
        print(f"\n🚀 Aspiration via WGET lancée ({total_files} fichiers au programme)...")
        
        success, skipped, failed = 0, 0, 0

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.wget_download_worker, url): url for url in urls}
                
                for i, future in enumerate(as_completed(futures), 1):
                    status, filename = future.result()
                    if status == "DOWNLOADED":
                        success += 1
                    elif status == "EXISTS":
                        skipped += 1
                    else:
                        failed += 1
                    
                    if i % 10 == 0 or i == total_files:
                        print(f"🔄 [{i}/{total_files}] | Récupérés : {success} | Déjà là : {skipped} | Échecs définitifs : {failed}")
        except KeyboardInterrupt:
            print("\n🛑 Interruption. Fermeture propre.")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)

if __name__ == "__main__":
    # On reste discret : 8 workers simultanés qui ne lâchent rien
    archiver = GDELTWgetArchiver(output_dir='gdelt_raw_archive', max_workers=8)
    
    # Relance sur ta période problématique de 2024
    archiver.download_history(
        start_date='2024-05-10',
        end_date='2024-05-30'
    )