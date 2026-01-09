import pandas as pd
import folium

# Charger le CSV
df = pd.read_csv("flickr_data2.csv")

# Nettoyer les noms de colonnes (IMPORTANT)
df.columns = df.columns.str.strip()

# Prendre les 10 premières lignes
df_10 = df.head(10)

# Carte centrée sur Lyon
map_lyon = folium.Map(
    location=[45.7640, 4.8357],
    zoom_start=12
)

# Ajouter les points
for _, row in df_10.iterrows():
    folium.Marker(
        location=[row["lat"], row["long"]],
        popup=row["title"] if pd.notna(row["title"]) else "Sans titre"
    ).add_to(map_lyon)

# Sauvegarde
map_lyon.save("lyon_points.html")

print("Carte créée : lyon_points.html")
