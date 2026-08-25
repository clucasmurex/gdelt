import json
import random
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Tirage de 100 sources pour vérification (FP/FN)")
    p.add_argument("--source_map", type=Path, default=Path("/data/gdelt/gdelt_sources_mapping.json"))
    p.add_argument("--retained_ids", type=Path, default=Path("liste_ids_retenus.txt"))
    p.add_argument("--output_file", type=Path, default=Path("./robustness_checks/check_100_sources_fp_fn.txt"))
    return p.parse_args()

def main():
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Charger les IDs retenus dans un "set" (ensemble) pour une recherche ultra-rapide
    print("[INFO] Chargement des identifiants retenus...")
    try:
        with open(args.retained_ids, "r", encoding="utf-8") as f:
            # On stocke sous forme d'entiers pour comparer facilement
            retained_ids = {int(line.strip()) for line in f if line.strip().isdigit()}
    except FileNotFoundError:
        print(f"[ERREUR] Le fichier {args.retained_ids} est introuvable.")
        return

    # 2. Charger le dictionnaire global des sources GDELT
    print("[INFO] Chargement du mapping des sources (JSON)...")
    try:
        with open(args.source_map, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERREUR] Le fichier {args.source_map} est introuvable.")
        return

    # Extraire le sous-dictionnaire "source_to_id"
    source_to_id = mapping_data.get("source_to_id", {})
    if not source_to_id:
        print("[ERREUR] La clé 'source_to_id' est introuvable ou vide dans le JSON.")
        return

    # 3. Tirage aléatoire de 100 sources
    print("[INFO] Tirage aléatoire de 100 domaines...")
    all_sources = list(source_to_id.items())  # Crée une liste de tuples : [("law360.com", 1), ...]
    sample_size = min(200, len(all_sources))  # Sécurité au cas où il y aurait moins de 100 sources
    
    sampled_sources = random.sample(all_sources, sample_size)

    # 4. Vérification et écriture des résultats
    print(f"[INFO] Génération du fichier d'annotation : {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("=== ROBUSTNESS CHECK : FAUX POSITIFS / FAUX NÉGATIFS ===\n")
        f.write("Vérifiez manuellement :\n")
        f.write("- Faux Négatifs : Les sources 'Rejetées' qui auraient dû être gardées.\n")
        f.write("- Faux Positifs : Les sources 'Retenues' qui auraient dû être jetées.\n\n")
        
        for i, (domain, src_id) in enumerate(sampled_sources, 1):
            # On vérifie si l'ID de la source tirée au sort fait partie de ta liste blanche
            if src_id in retained_ids:
                status = "OUI (Retenue)"
            else:
                status = "NON (Rejetée)"
                
            # Formatage aligné pour faciliter la lecture humaine
            f.write(f"[{i:03d}] {status:>15} | ID: {src_id:>8} | Domaine: {domain}\n")

    print("\n[SUCCÈS] Opération terminée. Bon courage pour l'annotation !")

if __name__ == "__main__":
    main()