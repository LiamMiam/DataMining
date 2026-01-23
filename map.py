import pandas as pd
import numpy as np
import folium #pour la carte leaflet
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull #pour faire un polygone autour des clusters

from name import process_texts_and_extract_keywords, format_keywords


# Charger le CSV déjà nettoyé
df = pd.read_csv("flickr_data2_clean.csv")


print(f"Total points : {len(df)}")

# Sous-échantillon pour clustering (100 000 points max)
n_cluster_sample = min(100000, len(df))
df_cluster_sample = df.sample(n=n_cluster_sample, random_state=42)
coords = df_cluster_sample[["lat", "long"]].to_numpy()

# Clustering DBSCAN
kms_per_radian = 6371.0088
epsilon = 0.05 / kms_per_radian  # ~50 mètres → plus précis pour éviter gros clusters
min_samples = 250                  # ignorer les petits clusters < 250 points

db = DBSCAN(
    eps=epsilon,
    min_samples=min_samples,
    algorithm="ball_tree",
    metric="haversine"
)

labels = db.fit_predict(np.radians(coords))
df_cluster_sample["cluster"] = labels

# Garder seulement les vrais clusters (>=100 points)
df_valid_clusters = df_cluster_sample[df_cluster_sample["cluster"] != -1]

print(f"Clusters trouvés : {df_valid_clusters['cluster'].nunique()}")

# Carte Folium
map_lyon = folium.Map(location=[45.7640, 4.8357], zoom_start=13)

# Sous-échantillon aléatoire global (200 points)
df_display = df.sample(
    n=min(1000, len(df)),
    random_state=42
)


for _, row in df_display.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        color="blue",
        fill=True,
        fill_opacity=0.7
    ).add_to(map_lyon)

# Polygones autour des clusters
for cluster_id in df_valid_clusters["cluster"].unique():
    points = df_valid_clusters[df_valid_clusters["cluster"] == cluster_id][["lat", "long"]].to_numpy()

    # Sélection du cluster
    cluster_df = df_valid_clusters[df_valid_clusters["cluster"] == cluster_id]

    # Concaténation title + tags (gestion des NaN)
    texts = (
        cluster_df["title"].fillna("").astype(str)
        + " "
        + cluster_df["tags"].fillna("").astype(str)
    ).tolist()

    # Extraction des mots-clés
    raw_keywords = process_texts_and_extract_keywords(texts)

    
    keywords_str = format_keywords(raw_keywords, max_words=5)

    if len(points) < 3:
        continue

    # Petit jitter pour éviter les points identiques
    points += np.random.normal(0, 1e-7, points.shape)

    try:
        hull = ConvexHull(points)
        polygon = [[points[v][0], points[v][1]] for v in hull.vertices]
        polygon.append(polygon[0])  # fermer le polygone

        # Popup avec HTML plus joli
        popup_html = f"""
        <b>Cluster {cluster_id}</b><br>
        Points: {len(points)}<br>
        <b>Keywords:</b> {keywords_str}
        """

        folium.Polygon(
            locations=polygon,
            color="red",
            fill=True,
            fill_opacity=0.2,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(map_lyon)
    except Exception as e:
        print(f"Impossible de créer polygone cluster {cluster_id}: {e}")
        continue


# Sauvegarde
map_lyon.save("lyon_clusters_filtered.html")
print("Carte générée : lyon_clusters_filtered.html")





#cluster utilisateurs différents sur une même zone géographique