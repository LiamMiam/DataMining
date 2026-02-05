import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# CONFIGURATION
# ======================================================
CSV_PATH = "flickr_data2_clean.csv"
MAX_POINTS = 100000          # sous-échantillon pour accélérer
K_MIN = 1
K_MAX = 60
RANDOM_STATE = 42

# ======================================================
# LECTURE DU CSV
# ======================================================
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Nombre de lignes lues : {len(df)}")

# ======================================================
# SÉLECTION DES FEATURES
# ======================================================
# Ici : clustering spatial uniquement
df = df[["lat", "long"]].dropna()

print(f"Points valides (lat/long) : {len(df)}")

# ======================================================
# SOUS-ÉCHANTILLONNAGE
# ======================================================
if len(df) > MAX_POINTS:
    df = df.sample(n=MAX_POINTS, random_state=RANDOM_STATE)
    print(f"Sous-échantillonnage à {MAX_POINTS} points")

X = df.to_numpy()

# ======================================================
# NORMALISATION (bonne pratique)
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# MÉTHODE DU COUDE (ELBOW)
# ======================================================
wcss = []
k_values = range(K_MIN, K_MAX + 1)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=RANDOM_STATE
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # WCSS

# ======================================================
# AFFICHAGE DU GRAPHIQUE
# ======================================================
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker="o")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("WCSS / Inertia")
plt.title("Méthode du coude (Elbow Method) – KMeans")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()