import re

def extract_theme_chains(input_file, strict_keywords, flexible_keywords, exclude_keywords, output_file):
    """
    Finds lines containing keywords based on strict/flexible rules,
    but IMMEDIATELY excludes any line containing any keyword from exclude_keywords.
    """
    extracted_items = []
    
    # 1. Préparation de la Blacklist (mots à exclure)
    if exclude_keywords:
        exclude_pattern = re.compile(
            '|'.join(re.escape(kw.upper()) for kw in exclude_keywords)
        )
    else:
        exclude_pattern = None

    # 2. Préparation des mots stricts (\b garantit les limites du mot)
    if strict_keywords:
        strict_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw.upper()) for kw in strict_keywords) + r')\b'
        )
    else:
        strict_pattern = None

    # 3. Préparation des mots flexibles
    if flexible_keywords:
        flexible_pattern = re.compile(
            '|'.join(re.escape(kw.upper()) for kw in flexible_keywords)
        )
    else:
        flexible_pattern = None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line_content = line.strip()
                line_upper = line_content.upper()
                
                # --- ÉTAPE 1 : Le Garde-Fou (Exclusion) ---
                # Si la ligne contient un mot banni, on l'ignore DIRECTEMENT
                if exclude_pattern and exclude_pattern.search(line_upper):
                    continue  # Passe à la ligne suivante du fichier
                
                # --- ÉTAPE 2 : Recherche des mots-clés ---
                match_found = False
                
                # Vérification des mots stricts
                if strict_pattern and strict_pattern.search(line_upper):
                    match_found = True
                
                # Vérification des mots flexibles (si pas déjà validé)
                if not match_found and flexible_pattern and flexible_pattern.search(line_upper):
                    match_found = True
                
                # Si la ligne est validée, on nettoie et on stocke
                if match_found:
                    clean_chain = re.sub(r'(?:\s*_\s*|\s*)\d+$', '', line_content)
                    if clean_chain:
                        extracted_items.append(clean_chain)
        
        # Sauvegarde et affichage des résultats
        if extracted_items:
            print(f"--- Extracted {len(extracted_items)} items (trailing numbers removed): ---")
            with open(output_file, 'w', encoding='utf-8') as out_file:
                for item in extracted_items:
                    print(item)
                    out_file.write(item + "\n")
            print(f"\n[Success] Cleaned list saved to '{output_file}'")
        else:
            print("No themes found matching the criteria.")
            
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please check the file path.")

# --- Execution ---

# 1. Mots à exclure impérativement (Blacklist)
my_exclude_keywords = ["TAX_WORLD", "TAX_DISEASE", "TAX_TERROR_GROUP", "TAX_ETHNICITY", "TAX_RELIGION", "TAX_AGRICULHARMINSECTS", "ECON_WORLDCURRENCIES", "TAX_POLITICAL_PART"]

# 2. Mots "seuls" (Pas de Madagascar pour GAS)
my_strict_keywords = ["SOIL" ]

# 3. Mots qui acceptent les dérivés (Bank, Banking...)
my_flexible_keywords = ["AGRIC", "FARM", "CROP", "FOOD", "HARVEST", "RURAL", "PLANT", "CULTIVAT", "IRRIGATION", "FERTILIZE", "MEAT", "RICE", "WHEAT", "CORN", "GRAIN", "SOY", "AGRI", "AGRO"]

extract_theme_chains(
    input_file="themes.txt", 
    strict_keywords=my_strict_keywords,
    flexible_keywords=my_flexible_keywords,
    exclude_keywords=my_exclude_keywords,  # <--- Nouvelle liste passée ici
    output_file="agriculture_themes.txt"
)