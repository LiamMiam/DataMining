import pandas as pd
import numpy as np
import folium  # pour la carte leaflet
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull  # pour faire un polygone autour des clusters

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from Flicker.textMining.NER import process_texts_and_extract_keywords, format_keywords
# ======================================================
# CONFIG
# ======================================================
RECLUSTER_THRESHOLD = 5000      # si un cluster dépasse ce nombre de points -> on relance DBSCAN dessus
MAX_PASSES = 5                 # nombre max de passes (sécurité anti-boucle infinie)

kms_per_radian = 6371.0088
epsilon = 0.05 / kms_per_radian      # DBSCAN global ~50m
min_samples = 250

# Paramètres de re-clustering (plus stricts)
# (tu peux ajuster, ex 30m puis 20m puis 15m, etc.)
eps_shrink_factor = 0.7             # à chaque passe, eps *= 0.7
min_samples_shrink_factor = 0.85    # à chaque passe, min_samples *= 0.85 (descend doucement)

MIN_USERS = 25
USER_COL = "user"

# ======================================================
# LOAD
# ======================================================
df = pd.read_csv("Flicker/data/flickr_data2_clean.csv")
print(f"Total points : {len(df)}")

# Sous-échantillon pour clustering (100 000 points max)
n_cluster_sample = min(100000, len(df))
df_cluster_sample = df.sample(n=n_cluster_sample, random_state=42).copy()
coords = df_cluster_sample[["lat", "long"]].to_numpy()

# ======================================================
# DBSCAN PASS 1
# ======================================================
db = DBSCAN(
    eps=epsilon,
    min_samples=min_samples,
    algorithm="ball_tree",
    metric="haversine"
)

labels = db.fit_predict(np.radians(coords))
df_cluster_sample["cluster"] = labels


# ======================================================
# MULTI-PASS RECLUSTERING FOR BIG CLUSTERS
# ======================================================
def recluster_big_clusters(df_in: pd.DataFrame,
                           cluster_col: str,
                           base_eps: float,
                           base_min_samples: int,
                           threshold: int,
                           max_passes: int) -> pd.DataFrame:
    """
    Re-fait DBSCAN en plusieurs passes sur les clusters trop gros.
    À chaque passe : on prend les clusters > threshold et on relance DBSCAN dessus
    avec eps plus petit et min_samples un peu plus petit.
    """
    df_work = df_in.copy()
    df_work[cluster_col] = df_work[cluster_col].astype(int)

    # On va assigner des IDs uniques aux nouveaux sous-clusters
    next_cluster_id = int(df_work[cluster_col].max()) + 1

    eps_pass = base_eps
    min_samples_pass = base_min_samples

    for p in range(1, max_passes + 1):
        # tailles des clusters (hors bruit)
        sizes = df_work[df_work[cluster_col] != -1].groupby(cluster_col).size()
        big_ids = sizes[sizes > threshold].index.tolist()

        if not big_ids:
            print(f"[Recluster] Stop: plus aucun cluster > {threshold} (pass {p}).")
            break

        print(f"[Recluster] Pass {p}: {len(big_ids)} cluster(s) > {threshold} points "
              f"(eps={eps_pass * kms_per_radian * 1000:.1f}m, min_samples={min_samples_pass})")

        # Pour chaque gros cluster, relance DBSCAN localement
        for cid in big_ids:
            mask = df_work[cluster_col] == cid
            sub_coords = df_work.loc[mask, ["lat", "long"]].to_numpy()

            # DBSCAN local
            sub_labels = DBSCAN(
                eps=eps_pass,
                min_samples=int(min_samples_pass),
                algorithm="ball_tree",
                metric="haversine"
            ).fit_predict(np.radians(sub_coords))

            # Remapping labels locaux -> IDs globaux uniques
            uniq = [l for l in np.unique(sub_labels) if l != -1]
            mapping = {l: (next_cluster_id + i) for i, l in enumerate(uniq)}
            next_cluster_id += len(uniq)

            # Construit les nouveaux ids: bruit local -> -1
            new_ids = np.array([mapping.get(l, -1) for l in sub_labels], dtype=int)

            # Remplace le cluster initial par les nouveaux sous-clusters
            df_work.loc[mask, cluster_col] = new_ids

        # Rétrécit les paramètres pour la passe suivante (plus fin)
        eps_pass *= eps_shrink_factor
        min_samples_pass = max(20, int(min_samples_pass * min_samples_shrink_factor))

    return df_work


df_cluster_sample = recluster_big_clusters(
    df_in=df_cluster_sample,
    cluster_col="cluster",
    base_eps=epsilon,
    base_min_samples=min_samples,
    threshold=RECLUSTER_THRESHOLD,
    max_passes=MAX_PASSES
)

# ======================================================
# FILTER VALID CLUSTERS (remove noise)
# ======================================================
df_valid_clusters = df_cluster_sample[df_cluster_sample["cluster"] != -1].copy()
print(f"Clusters trouvés (après multi-pass) : {df_valid_clusters['cluster'].nunique()}")

# ======================================================
# FILTER BY MIN DISTINCT USERS
# ======================================================
users_per_cluster = df_valid_clusters.groupby("cluster")[USER_COL].nunique()
valid_cluster_ids = users_per_cluster[users_per_cluster >= MIN_USERS].index
df_valid_clusters = df_valid_clusters[df_valid_clusters["cluster"].isin(valid_cluster_ids)].copy()

print(f"Clusters gardés (>= {MIN_USERS} users) : {df_valid_clusters['cluster'].nunique()}")

# ======================================================
# MAP FOLIUM
# ======================================================
map_lyon = folium.Map(location=[45.7640, 4.8357], zoom_start=13)

# Sous-échantillon aléatoire global (1000 points)
df_display = df.sample(n=min(1000, len(df)), random_state=42)

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
    cluster_df = df_valid_clusters[df_valid_clusters["cluster"] == cluster_id]
    points = cluster_df[["lat", "long"]].to_numpy()

    # Keywords sur titres
    texts = cluster_df["title"].fillna("").astype(str).tolist()
    raw_keywords = process_texts_and_extract_keywords(texts)
    keywords_str = format_keywords(raw_keywords, max_words=3)

    if len(points) < 3:
        continue

    points = points + np.random.normal(0, 1e-7, points.shape)

    try:
        hull = ConvexHull(points)
        polygon = [[points[v][0], points[v][1]] for v in hull.vertices]
        polygon.append(polygon[0])

        popup_html = f"""
        <b>Cluster {cluster_id}</b><br>
        Points: {len(points)}<br>
        Users: {cluster_df[USER_COL].nunique()}<br>
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
# Sauvegarde
map_lyon.save("Flicker/DBscan/dbscanReccursif.html")
print("Carte générée : Flicker/DBscan/dbscanReccursif.html")