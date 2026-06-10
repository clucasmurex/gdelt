# fileName: gdelt_diagnostic.py
import subprocess

# Une URL typique du 10 mai 2024
test_url = "http://data.gdeltproject.org/gdeltv2/20240510000000.gkg.csv.zip"
test_output = "diagnostic_test.zip"

print("🔍 --- DIAGNOSTIC CONNEXION GDELT ---")

# Test 1 : Wget classique (ce qu'on faisait)
print("\n1. Test avec Wget classique...")
cmd_classic = ["wget", "-O", test_output, test_url]
result_classic = subprocess.run(cmd_classic, capture_output=True, text=True)
print(f"Code de retour : {result_classic.returncode}")
# Affiche les dernières lignes de l'erreur si ça a échoué
if result_classic.returncode != 0:
    print("Détail de l'erreur système :")
    print(result_classic.stderr[-500:]) 

# Test 2 : Wget déguisé en navigateur (Le correctif)
print("\n2. Test avec Wget déguisé en navigateur (User-Agent modifié)...")
cmd_stealth = [
    "wget", 
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", 
    "-O", test_output, 
    test_url
]
result_stealth = subprocess.run(cmd_stealth, capture_output=True, text=True)
print(f"Code de retour : {result_stealth.returncode}")
if result_stealth.returncode == 0:
    print("🟢 Succès ! Le déguisement en navigateur fonctionne.")
    if os.path.exists(test_output):
        os.remove(test_output)
else:
    print("🔴 Échec même avec le déguisement.")