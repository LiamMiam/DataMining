import numpy as np
import pandas as pd
import folium
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class KMeansCustom:

    
    def __init__(self, n_clusters=30, max_iterations=100, random_state=42, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.tolerance = tolerance
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.iterations_count_ = 0
    
    def _initialize_centers(self, X):
        # Choisir aléatoirement n_clusters points comme centres initiaux
        np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X, centers):
        # Distance euclidienne de chaque point vers chaque centre
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        # Assigner chaque point au centre le plus proche
        return np.argmin(distances, axis=0)
    
    def _update_centers(self, X, labels):
        # Recalculer les centres comme moyenne des points du cluster
        new_centers = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 
            else self.cluster_centers_[k]
            for k in range(self.n_clusters)
        ])
        return new_centers
    
    def _compute_inertia(self, X, labels, centers):
        # Somme des distances au carré (mesure de qualité du clustering)
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k])**2)
        return inertia
    
    def fit(self, X):
        X = np.asarray(X)
        centers = self._initialize_centers(X)
        self.cluster_centers_ = centers
        
        print(" Démarrage K-Means...")
        
        for iteration in range(self.max_iterations):
            labels = self._assign_clusters(X, centers)
            inertia = self._compute_inertia(X, labels, centers)
            new_centers = self._update_centers(X, labels)
            
            # Mesurer la stabilité des centres
            centers_shift = np.sqrt(np.sum((new_centers - centers)**2))
            
            if iteration % 10 == 0:
                print(f"   Iter {iteration + 1:3d} | Inertie: {inertia:10.2f} | Shift: {centers_shift:.6f}")
            
            # Arrêter si convergence
            if centers_shift < self.tolerance:
                print(f"✓ Convergence à l'itération {iteration + 1}")
                break
            
            centers = new_centers
        
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.iterations_count_ = iteration + 1
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X, self.cluster_centers_)


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df, df[["lat", "long"]].to_numpy()


def print_stats(df, kmeans):
    print(f"\n{'='*50}")
    print(f"✓ K-Means terminé !")
    print(f"  • Inertie finale : {kmeans.inertia_:.2f}")
    print(f"  • Itérations : {kmeans.iterations_count_}")
    print(f"{'='*50}")
    
    cluster_stats = df["cluster"].value_counts().sort_index()
    print("\n Taille des clusters :")
    for cluster_id, count in cluster_stats.items():
        print(f"   Cluster {cluster_id:2d}: {count:4d} photos")


# Charge les données et lance le clustering
N_CLUSTERS = 30
MIN_CLUSTER_SIZE = 50  # Garder seulement les clusters "gros"
CSV_FILE = "flickr_data2_clean.csv"

df, coords = load_data(CSV_FILE)

print(f" Données chargées : {len(df)} points")

kmeans = KMeansCustom(n_clusters=N_CLUSTERS, max_iterations=100, random_state=42)
kmeans.fit(coords)

df["cluster"] = kmeans.labels_

# Filtrer les clusters trop petits (points isolés)
cluster_sizes = df["cluster"].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index
df_filtered = df[df["cluster"].isin(valid_clusters)].copy()

print(f"\n Filtrage des petits clusters :")
print(f"  • Clusters gardés : {len(valid_clusters)}")
print(f"  • Points abandonnés : {len(df) - len(df_filtered)}")

print_stats(df_filtered, kmeans)

def create_map(df, kmeans, colors, sample_size=500):
    """Crée la carte Folium avec clusters et centres"""
    map_lyon = folium.Map(location=[45.764, 4.8357], zoom_start=13)
    
    # Afficher les points (sous-échantillon pour performance)
    df_display = df.sample(n=min(sample_size, len(df)), random_state=42)
    for _, row in df_display.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=4,
            color=colors[int(row["cluster"]) % len(colors)],
            fill=True,
            fill_opacity=0.6,
            popup=f"Cluster {int(row['cluster'])}"
        ).add_to(map_lyon)
    
    # Afficher les centres
    for i, (lat, lon) in enumerate(kmeans.cluster_centers_):
        folium.Marker(
            location=[lat, lon],
            popup=f"Centre {i}",
            icon=folium.Icon(color="black", icon="star")
        ).add_to(map_lyon)
    
    return map_lyon


def add_cluster_boundaries(map_obj, df, colors, n_clusters):
    for cluster_id in range(n_clusters):
        points = df[df["cluster"] == cluster_id][["lat", "long"]].to_numpy()
        
        if len(points) < 3:
            continue
        
        # Petit bruit pour éviter les points identiques
        points = points + np.random.normal(0, 1e-7, points.shape)
        
        try:
            hull = ConvexHull(points)
            polygon = [[points[v][0], points[v][1]] for v in hull.vertices]
            polygon.append(polygon[0])  # Fermer le polygone
            
            folium.Polygon(
                locations=polygon,
                color=colors[cluster_id % len(colors)],
                fill=True,
                fill_opacity=0.2,
                weight=2,
                popup=f"Cluster {cluster_id}"
            ).add_to(map_obj)
        except Exception as e:
            print(f"  Cluster {cluster_id}: {e}")


# ========== CARTOGRAPHIE ==========
COLORS = [
    "red", "blue", "green", "purple", "orange", 
    "darkred", "cadetblue", "darkgreen", "darkblue", 
    "pink", "gray", "black", "lightblue", "lightgreen", "beige"
]

print("\n Création de la carte...")

map_lyon = create_map(df, kmeans, COLORS)
add_cluster_boundaries(map_lyon, df, COLORS, N_CLUSTERS)

map_lyon.save("lyon_kmeans_clusters.html")
print("✓ Carte sauvegardée : lyon_kmeans_clusters.html")
