import pandas as pd
import numpy as np
import folium
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Charger le CSV nettoyé
df = pd.read_csv("flickr_data2_clean.csv")

print(f"Total points : {len(df)}")

# Sous-échantillon pour clustering (5 000 points max - hierarchical est TRÈS gourmand en mémoire)
n_cluster_sample = min(5000, len(df))
df_cluster_sample = df.sample(n=n_cluster_sample, random_state=42)
coords = df_cluster_sample[["lat", "long"]].to_numpy()

print(f"Échantillon pour clustering : {n_cluster_sample} points")
print("Démarrage du clustering hiérarchique...")

# Clustering hiérarchique (Agglomerative Clustering)
n_clusters = 15  # nombre de clusters souhaités
distance_threshold = None  

hierarchical = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='euclidean',  
    linkage='ward' 
)

labels = hierarchical.fit_predict(coords)
df_cluster_sample["cluster"] = labels

print(f"Clusters créés : {df_cluster_sample['cluster'].nunique()}")

# Statistiques par cluster
cluster_stats = df_cluster_sample.groupby('cluster').size().sort_values(ascending=False)
print("\nTaille des clusters :")
print(cluster_stats)

# Créer un dendrogramme (arbre hiérarchique)
print("\nGénération du dendrogramme...")
linkage_matrix = linkage(coords[:1000], method='ward')  # Seulement 1000 points pour vitesse

plt.figure(figsize=(15, 8))
plt.title('Dendrogramme - Clustering Hiérarchique')
plt.xlabel('Index des points')
plt.ylabel('Distance')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
print("Dendrogramme sauvegardé : dendrogram.png")
plt.close()

# Carte Folium
map_lyon = folium.Map(location=[45.7640, 4.8357], zoom_start=13)

# Couleurs pour différencier les clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
          'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
          'gray', 'black', 'lightgray']

# Sous-échantillon pour affichage (500 points max)
df_display = df_cluster_sample.sample(n=min(500, len(df_cluster_sample)), random_state=42)

for _, row in df_display.iterrows():
    color = colors[int(row["cluster"]) % len(colors)]
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=5,
        popup=f"Cluster {row['cluster']}",
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(map_lyon)

# Polygones autour des clusters
for cluster_id in df_cluster_sample["cluster"].unique():
    points = df_cluster_sample[df_cluster_sample["cluster"] == cluster_id][["lat", "long"]].to_numpy()

    if len(points) < 3:
        continue

    # Petit jitter pour éviter les points identiques
    points += np.random.normal(0, 1e-7, points.shape)

    try:
        hull = ConvexHull(points)
        polygon = [[points[v][0], points[v][1]] for v in hull.vertices]
        polygon.append(polygon[0])  # fermer le polygone

        color = colors[int(cluster_id) % len(colors)]
        folium.Polygon(
            locations=polygon,
            color=color,
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id} ({len(points)} points)"
        ).add_to(map_lyon)
    except Exception as e:
        print(f"Impossible de créer polygone cluster {cluster_id}: {e}")
        continue

# Ajouter les centres de clusters
for cluster_id in df_cluster_sample["cluster"].unique():
    points = df_cluster_sample[df_cluster_sample["cluster"] == cluster_id][["lat", "long"]]
    center_lat = points["lat"].mean()
    center_long = points["long"].mean()
    
    folium.Marker(
        location=[center_lat, center_long],
        popup=f"Centre Cluster {cluster_id}<br>{len(points)} photos",
        icon=folium.Icon(color=colors[int(cluster_id) % len(colors)], icon='info-sign')
    ).add_to(map_lyon)

# Sauvegarde
map_lyon.save("lyon_hierarchical_clusters.html")
print("\nCarte générée : lyon_hierarchical_clusters.html")
