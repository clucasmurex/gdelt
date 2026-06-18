import pandas as pd

# Charge uniquement les 5 premières lignes du fichier pour vérifier la structure
df = pd.read_parquet("./gdelt_parquet_db/gdelt_2015-04.parquet")

print("📏 Dimensions (Lignes, Colonnes) :", df.shape)
print("\n📋 Échantillon des données :")
print(df[['Organizations', 'Tone', 'WordCount', 'SourceCommonName_ID']].head())