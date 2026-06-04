import gdelt
import json

print("Initialisation GDELT v2...")
gd = gdelt.gdelt(version=2)

print("Recherche...")
results_json = gd.Search(['2026 06 03'], table='gkg', output='json', coverage=True)

# Parser le JSON
print("Parsing du JSON...")
results = json.loads(results_json)

print(f"\n=== RÉSULTATS CORRECTS ===")
print(f"Type: {type(results)}")
print(f"Nombre d'éléments: {len(results)}")

if results:
    print(f"\n✓ Premier résultat:")
    print(json.dumps(results[0], indent=2, ensure_ascii=False))
    
    print(f"\n✓ Clés disponibles:")
    print(list(results[0].keys()))
else:
    print("⚠ Aucun résultat")