import re

def extract_theme_chains(input_file, keyword, output_file):
    """
    Finds lines containing the keyword, strips off any trailing numbers 
    at the very end of the line, and saves each theme to a custom output file.
    """
    extracted_items = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line_content = line.strip()
                
                # Check if the keyword exists in the line (case-insensitive)
                if keyword.upper() in line_content.upper():
                    
                    # Removes trailing underscores, spaces, and numbers at the very end
                    clean_chain = re.sub(r'(?:\s*_\s*|\s*)\d+$', '', line_content)
                    
                    if clean_chain:
                        extracted_items.append(clean_chain)
        
        # Save and print results
        if extracted_items:
            print(f"--- Extracted {len(extracted_items)} items (trailing numbers removed): ---")
            
            with open(output_file, 'w', encoding='utf-8') as out_file:
                for item in extracted_items:
                    print(item)  # Returns à la ligne in console
                    out_file.write(item + "\n")  # Returns à la ligne in file
                    
            print(f"\n[Success] Cleaned list saved to '{output_file}'")
        else:
            print(f"No themes found containing the keyword '{keyword}'.")
            
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please check the file path.")

# --- Execution ---
# You can now change all three parameters here:
extract_theme_chains(
    input_file="themes.txt", 
    keyword="TRADE", 
    output_file="trade_themes.txt"  # <--- Change your output file name here!
)