import os
import time
import requests

# Point de terminaison SPARQL extrait du code HTML de query.wikidata.org
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# COUNTRIES = {
#     "GB": "Q145",   # Royaume-Uni
# }

# Dictionnaire associant les codes ISO aux identifiants Wikidata (Q-ID)
def get_target_countries():
    print("Récupération de la liste des pays depuis Wikidata...")
    query = """
    SELECT ?iso ?qid WHERE {
      ?country wdt:P31 wd:Q3624078 ;  # Est un état souverain
               wdt:P297 ?iso .        # Code ISO 3166-1 alpha-2
               
      # Extraction du Q-ID depuis l'URL de l'entité
      BIND(REPLACE(STR(?country), ".*Q", "Q") AS ?qid)
      
      # Exclure les membres de l'Union Européenne (Q458)
      FILTER NOT EXISTS { ?country wdt:P463 wd:Q458 . }
      
      # Exclure la France, les USA et la Suisse
      FILTER (?iso NOT IN ("FR", "US", "CH", "GB"))
    }
    """
    headers = {
        "User-Agent": "MediaDataFetcher/1.0",
        "Accept": "application/sparql-results+json"
    }
    
    response = requests.get("https://query.wikidata.org/sparql", params={'query': query}, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    countries = {}
    
    for item in data['results']['bindings']:
        iso = item['iso']['value']
        qid = item['qid']['value']
        countries[iso] = qid
        
    print(f"{len(countries)} pays cibles identifiés.")
    return countries

# Vous remplacez simplement votre dictionnaire statique par l'appel à cette fonction :
COUNTRIES = get_target_countries()

# Dictionnaire associant vos noms de fichiers aux types de médias Wikidata
MEDIA_TYPES = {
    "news": "Q11032",       # journaux
    "agencies": "Q192283",  # agences de presse
    "websites": "Q1193236"  # sites d'infos
}

# Modèle de la requête SPARQL avec des placeholders (%s) pour le pays et le type
QUERY_TEMPLATE = """
SELECT DISTINCT
  ?mediaLabel
  ?typeLabel
  ?countryLabel
  ?inception
  ?website
WHERE {
  ?media wdt:P31 ?type ;
         wdt:P17 wd:%s ; 
         wdt:P856 ?website .

  ?type wdt:P279* wd:%s . 

  OPTIONAL {
    ?media wdt:P571 ?inception .
  }

  FILTER NOT EXISTS {
    ?media wdt:P576 ?endDate
  }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
}
"""

def setup_directory():
    # Détermine le chemin absolu vers Documents/gdelt/data/wikidata
    # os.path.expanduser("~") gère automatiquement le dossier utilisateur (Windows/Mac/Linux)
    base_dir = os.path.expanduser("~/Documents")
    target_dir = os.path.join(base_dir, "gdelt", "data", "wikidata")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def fetch_and_save_data(iso_code, country_qid, output_dir):
    for media_name, media_qid in MEDIA_TYPES.items():
        print(f"Interrogation de Wikidata pour {iso_code} ({media_name})...")
        
        # Injection des variables dans le template
        query = QUERY_TEMPLATE % (country_qid, media_qid)
        
        # Définition des headers : le User-Agent est obligatoire sur Wikidata
        # 'Accept': 'text/csv' permet de télécharger le fichier directement au bon format
        headers = {
            "User-Agent": "MediaDataFetcher/1.0 (clucas@murex.com)",
            "Accept": "text/csv"
        }
        
        try:
            response = requests.get(SPARQL_ENDPOINT, params={'query': query}, headers=headers)
            response.raise_for_status() # Lève une exception si le statut HTTP indique une erreur
            
            # Création du nom de fichier et du chemin complet
            filename = f"{iso_code}_{media_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Écriture du fichier CSV
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            print(f"✔ Sauvegardé avec succès : {filepath}")
            
        except requests.exceptions.RequestException as e:
            print(f"✖ Erreur lors de la requête pour {iso_code}_{media_name} : {e}")
        
        # Pause obligatoire de 2 secondes entre les requêtes pour respecter
        # les conditions d'utilisation de l'API Wikidata et éviter d'être banni
        time.sleep(2)

def main():
    output_dir = setup_directory()
    print(f"Dossier de destination : {output_dir}\n")
    
    for iso_code, country_qid in COUNTRIES.items():
        fetch_and_save_data(iso_code, country_qid, output_dir)
        print("-" * 40)
        
    print("Processus terminé !")

if __name__ == "__main__":
    main()