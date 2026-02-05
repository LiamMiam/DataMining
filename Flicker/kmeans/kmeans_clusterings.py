import numpy as np
import pandas as pd
import folium
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans

# Charger les données propres
df = pd.read_csv("flickr_data2_clean.csv")

coords = df[["lat", "long"]]

# K-means
k = 15
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

# Carte
map_lyon = folium.Map(location=[45.764, 4.8357], zoom_start=13)

colors = [
    "red","blue","green","purple","orange","darkred","cadetblue",
    "darkgreen","darkblue","pink","gray","black","lightblue","lightgreen","beige"
]

# Sous-échantillon pour affichage (performance)
df_display = df.sample(n=800, random_state=42)

for _, row in df_display.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=4,
        color=colors[row["cluster"] % len(colors)],
        fill=True,
        fill_opacity=0.6,
        popup=f"K-means cluster {row['cluster']}"
    ).add_to(map_lyon)

# 👉 CENTRES DES CLUSTERS (IMPORTANT)
centers = kmeans.cluster_centers_

for i, (lat, lon) in enumerate(centers):
    folium.Marker(
        location=[lat, lon],
        popup=f"Centre K-means {i}",
        icon=folium.Icon(color="black", icon="star")
    ).add_to(map_lyon)

# Polygones de Voronoi pour délimiter les clusters autour des centres
def voronoi_finite_polygons_2d(vor, radius=None):
    # Adapte https://stackoverflow.com/a/20678647 pour fermer les régions infinies
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Mappage point->arêtes
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruit chaque région
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # Région infinie -> fermer
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            # Point à mi-distance
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_region = np.asarray(new_region)
        vs = np.asarray(new_vertices)
        c = vs[new_region].mean(axis=0)
        angles = np.arctan2(vs[new_region][:, 1] - c[1], vs[new_region][:, 0] - c[0])
        new_region = new_region[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# Construit le Voronoi et découpe par un cadre englobant la ville
centers_xy = np.column_stack((centers[:, 1], centers[:, 0]))  # x=lon, y=lat
vor = Voronoi(centers_xy)
regions, vertices = voronoi_finite_polygons_2d(vor)

padding = 0.05  # env. ~5 km
lat_min, lat_max = coords["lat"].min() - padding, coords["lat"].max() + padding
lon_min, lon_max = coords["long"].min() - padding, coords["long"].max() + padding
bbox = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])

for idx, region in enumerate(regions):
    polygon_points = vertices[region]
    poly = Polygon(polygon_points)
    poly = poly.intersection(bbox)
    if poly.is_empty:
        continue

    # Folium attend lon, lat
    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _feat, c=colors[idx % len(colors)]: {
            "fillColor": c,
            "color": c,
            "weight": 2,
            "fillOpacity": 0.15,
        },
        name=f"Cluster zone {idx}",
    ).add_to(map_lyon)

map_lyon.save("lyon_kmeans_clusters.html")
print("Carte K-means générée")
