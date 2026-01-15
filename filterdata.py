import pandas as pd

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_CSV = "flickr_data2.csv"
OUTPUT_CSV = "flickr_data2_clean.csv"

# ======================================================
# LECTURE DU CSV
# ======================================================
# low_memory=False évite les erreurs de type sur gros fichiers
df = pd.read_csv(INPUT_CSV, low_memory=False)

print(f"Nombre de lignes initial : {len(df)}")

# ======================================================
# NETTOYAGE DES COLONNES
# ======================================================
# Supprime les espaces autour des noms de colonnes
df.columns = df.columns.str.strip()

# Supprime les colonnes vides dues aux ,,, à la fin des lignes
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ======================================================
# CONVERSION DES TYPES
# ======================================================

# Conversion latitude / longitude en float
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")

# Colonnes de dates à convertir en numérique
date_columns = [
    "date_taken_minute",
    "date_taken_hour",
    "date_taken_day",
    "date_taken_month",
    "date_taken_year",
    "date_upload_minute",
    "date_upload_hour",
    "date_upload_day",
    "date_upload_month",
    "date_upload_year",
]

# Conversion robuste : valeurs invalides → NaN
for col in date_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ======================================================
# FONCTION D'AFFICHAGE DES FILTRES
# ======================================================
def report_filter(name, before, after, removed_df):
    removed = before - after
    print(f" {name}")
    print(f" Lignes supprimées : {removed}")
    if removed > 0:
        print("   Exemple de ligne supprimée :")
        print(removed_df.iloc[0])
    else:
        print("   Aucune ligne supprimée")

# ======================================================
# SUPPRESSION DES DOUBLONS
# ======================================================
before = len(df)

# Doublons stricts sur toutes les colonnes
duplicates = df[df.duplicated()]
df = df.drop_duplicates()

report_filter("Doublons", before, len(df), duplicates)

# ======================================================
# FILTRE GÉOGRAPHIQUE (LYON LARGE)
# ======================================================
# Latitude ≈ 45 ± 5
# Longitude ≈ 4 ± 5
before = len(df)

invalid_geo = df[
    df["lat"].isna() | df["long"].isna() |
    (df["lat"] < 40) | (df["lat"] > 50) |
    (df["long"] < -1) | (df["long"] > 9)
]

df = df.drop(invalid_geo.index)

report_filter("Coordonnées hors zone Lyon", before, len(df), invalid_geo)

# ======================================================
# FILTRE DES DATES INCOHÉRENTES
# ======================================================
before = len(df)

invalid_dates = df[
    df[date_columns].isna().any(axis=1) |
    (df["date_taken_minute"] < 0) | (df["date_taken_minute"] > 60) |
    (df["date_taken_hour"] < 0) | (df["date_taken_hour"] > 24) |
    (df["date_taken_day"] < 1) | (df["date_taken_day"] > 31) |
    (df["date_taken_month"] < 1) | (df["date_taken_month"] > 12) |
    (df["date_taken_year"] < 1839) | (df["date_taken_year"] > 2026) |  #1839 = date 1er photo
    (df["date_upload_minute"] < 0) | (df["date_upload_minute"] > 60) |
    (df["date_upload_hour"] < 0) | (df["date_upload_hour"] > 24) |
    (df["date_upload_day"] < 1) | (df["date_upload_day"] > 31) |
    (df["date_upload_month"] < 1) | (df["date_upload_month"] > 12) |
    (df["date_upload_year"] < 1990) | (df["date_upload_year"] > 2026) 
]

df = df.drop(invalid_dates.index)

report_filter("Dates incohérentes", before, len(df), invalid_dates)

# ======================================================
# SAUVEGARDE dans un nouveau CSV
# ======================================================
df.to_csv(OUTPUT_CSV, index=False)

print("\n Nettoyage terminé")
print(f"Nombre de lignes finales : {len(df)}")
print(f"Fichier sauvegardé : {OUTPUT_CSV}")
