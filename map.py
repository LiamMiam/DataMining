import pandas as pd
import numpy as np
import folium #pour la carte leaflet
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull #pour faire un polygone autour des clusters

# Charger le CSV déjà nettoyé
df = pd.read_csv("flickr_data2_clean.csv")


print(f"Total points : {len(df)}")

# Sous-échantillon pour clustering (100 000 points max)
n_cluster_sample = min(100000, len(df))
df_cluster_sample = df.sample(n=n_cluster_sample, random_state=42)
coords = df_cluster_sample[["lat", "long"]].to_numpy()

# Clustering DBSCAN
kms_per_radian = 6371.0088
epsilon = 0.03 / kms_per_radian  # ~30 mètres → plus précis pour éviter gros clusters
min_samples = 100                  # ignorer les petits clusters < 100 points

db = DBSCAN(
    eps=epsilon,
    min_samples=min_samples,
    algorithm="ball_tree",
    metric="haversine"
)

labels = db.fit_predict(np.radians(coords))
df_cluster_sample["cluster"] = labels

# Garder seulement les vrais clusters (>=25 points)
df_valid_clusters = df_cluster_sample[df_cluster_sample["cluster"] != -1]

print(f"Clusters trouvés : {df_valid_clusters['cluster'].nunique()}")

# Carte Folium
map_lyon = folium.Map(location=[45.7640, 4.8357], zoom_start=13)

# Sous-échantillon pour affichage (200 points max)
df_display = df_valid_clusters.sample(n=min(200, len(df_valid_clusters)), random_state=42)

for _, row in df_display.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        popup=f"Cluster {row['cluster']}",
        color="blue",
        fill=True,
        fill_opacity=0.7
    ).add_to(map_lyon)

# Polygones autour des clusters
for cluster_id in df_valid_clusters["cluster"].unique():
    points = df_valid_clusters[df_valid_clusters["cluster"] == cluster_id][["lat", "long"]].to_numpy()

    if len(points) < 3:
        continue

    # Petit jitter pour éviter les points identiques
    points += np.random.normal(0, 1e-7, points.shape)

    try:
        hull = ConvexHull(points)
        polygon = [[points[v][0], points[v][1]] for v in hull.vertices]
        polygon.append(polygon[0])  # fermer le polygone

        folium.Polygon(
            locations=polygon,
            color="red",
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id} ({len(points)} points)"
        ).add_to(map_lyon)
    except Exception as e:
        print(f"Impossible de créer polygone cluster {cluster_id}: {e}")
        continue

# Sauvegarde
map_lyon.save("lyon_clusters_filtered.html")
print("Carte générée : lyon_clusters_filtered.html")
