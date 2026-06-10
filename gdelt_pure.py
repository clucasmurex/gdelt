# fileName: gdelt_archive_pure.py
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

class GDELTPureArchiver:
    def __init__(self, output_dir='gdelt_raw_archive', max_workers=16):
        # On se cale sur 16 à 24 workers max pour le réseau public de GDELT
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers

    def generate_gkg_urls_for_day(self, date_str):
        """Génère les 96 URLs de fichiers ZIP pour une journée complète."""
        base_url = "http://data.gdeltproject.org/gdeltv2/"
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        urls = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = f"{date_obj.year}{date_obj.month:02d}{date_obj.day:02d}{hour:02d}{minute:02d}00"
                urls.append(f"{base_url}{timestamp}.gkg.csv.zip")
        return urls

    def download_file_worker(self, url):
        """Télécharge un fichier ZIP unique et l'écrit directement sur le disque."""
        filename = url.split('/')[-1]
        filepath = os.path.join(self.output_dir, filename)

        # Skip si le fichier est déjà là (Reprise automatique après coupure)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return "EXISTS", filename

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            # Timeout réseau agressif de 12 secondes
            with urllib.request.urlopen(req, timeout=12) as response:
                with open(filepath, 'wb') as out_file:
                    # Écriture par blocs de 64 Ko en flux continu (Consommation RAM proche de 0)
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        out_file.write(chunk)
            return "DOWNLOADED", filename
        except Exception as e:
            # En cas de timeout ou erreur, on supprime le fichier incomplet s'il a été créé
            if os.path.exists(filepath):
                os.remove(filepath)
            return "ERROR", f"{filename} ({str(e)})"

    def download_history(self, start_date, end_date):
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Nombre total de jours à traiter
        total_days = (end_obj - current_date).days + 1
        
        print(f"📦 --- ARCHE GDELT BRUTE INITIÉE ---")
        print(f"📂 Dossier de stockage : {self.output_dir}")
        print(f"⏱️ Période : {start_date} à {end_date} ({total_days} jours)")
        print(f"🧵 Connexions simultanées : {self.max_workers}\n")

        day_count = 1
        while current_date <= end_obj:
            day_str = current_date.strftime('%Y-%m-%d')
            urls = self.generate_gkg_urls_for_day(day_str)
            
            success_downloads = 0
            skipped_files = 0
            failed_files = 0

            print(f"📅 [{day_count}/{total_days}] Téléchargement du {day_str}...", end='', flush=True)

            # ThreadPoolExecutor est parfait ici car c'est de l'I/O réseau pur (pas de calcul CPU)
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.download_file_worker, url): url for url in urls}
                    
                    for future in as_completed(futures):
                        status, info = future.result()
                        if status == "DOWNLOADED":
                            success_downloads += 1
                        elif status == "EXISTS":
                            skipped_files += 1
                        else:
                            failed_files += 1
            except KeyboardInterrupt:
                print("\n\n🛑 [Ctrl+C] Arrêt propre demandé par l'utilisateur.")
                print("🧹 Fermeture des connexions réseau en cours...")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(0)

            # Log de la journée
            print(f" Finis: {success_downloads} | Déjà là: {skipped_files} | Échecs/Timeouts: {failed_files}")
            
            current_date += timedelta(days=1)
            day_count += 1

        print(f"\n📊 --- ARCHIVAGE TERMINÉ ---")
        print(f"Tous les fichiers .zip bruts sont sécurisés dans : {self.output_dir}")

if __name__ == "__main__":
    # 16 à 20 workers max pour éviter de se faire kick par GDELT
    archiver = GDELTPureArchiver(output_dir='gdelt_raw_archive', max_workers=16)
    
    # Balance l'historique global que tu veux, tu peux couper quand tu veux avec Ctrl+C
    archiver.download_history(
        start_date='2024-05-10',
        end_date='2026-06-09'
    )