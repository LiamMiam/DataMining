import pandas as pd
import numpy as np
import json
import folium
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from Flicker.textMining.NER import process_texts_and_extract_keywords, format_keywords

# =========================
# CONFIG
# =========================
CSV_PATH = "data/flickr_data2_clean.csv"
OUT_HTML = "lyon_dbscan_by_year_slider.html"

YEAR_COL = "date_taken_year"   # adapte si besoin
USER_COL = "user"

MAX_POINTS_PER_YEAR = 30000    # pour ne pas exploser (optionnel)
MIN_POINTS_CLUSTER = 50       # dbscan min_samples
EPS_METERS = 50                # eps en mètres

MIN_USERS = 5                 # filtre cluster

kms_per_radian = 6371.0088
epsilon = (EPS_METERS / 1000) / kms_per_radian

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.dropna(subset=["lat", "long", YEAR_COL]).copy()
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df = df.dropna(subset=[YEAR_COL])
df[YEAR_COL] = df[YEAR_COL].astype(int)

years = sorted(df[YEAR_COL].unique())
print("Années dispo :", years)

# =========================
# BUILD MAP BASE
# =========================
m = folium.Map(location=[45.7640, 4.8357], zoom_start=13)

# =========================
# DBSCAN PER YEAR -> store polygons
# =========================
year_layers = {}

for y in years:
    dfy = df[df[YEAR_COL] == y].copy()
    if len(dfy) < MIN_POINTS_CLUSTER:
        continue

    # (Optionnel) limiter taille annuelle
    if len(dfy) > MAX_POINTS_PER_YEAR:
        dfy = dfy.sample(n=MAX_POINTS_PER_YEAR, random_state=42)

    coords = dfy[["lat", "long"]].to_numpy()

    db = DBSCAN(
        eps=epsilon,
        min_samples=MIN_POINTS_CLUSTER,
        algorithm="ball_tree",
        metric="haversine"
    )
    labels = db.fit_predict(np.radians(coords))
    dfy["cluster"] = labels

    dfy = dfy[dfy["cluster"] != -1].copy()
    if dfy.empty:
        continue

    # filtre min users
    if USER_COL in dfy.columns:
        users_per_cluster = dfy.groupby("cluster")[USER_COL].nunique()
        valid_ids = users_per_cluster[users_per_cluster >= MIN_USERS].index
        dfy = dfy[dfy["cluster"].isin(valid_ids)].copy()

    if dfy.empty:
        continue

    polygons = []

    for cid in sorted(dfy["cluster"].unique()):
        cluster_df = dfy[dfy["cluster"] == cid]
        points = cluster_df[["lat", "long"]].to_numpy()

        if len(points) < 3:
            continue

        # keywords
        texts = cluster_df["title"].fillna("").astype(str).tolist() if "title" in cluster_df.columns else []
        raw_keywords = process_texts_and_extract_keywords(texts) if texts else []
        keywords_str = format_keywords(raw_keywords, max_words=3) if raw_keywords else ""

        # jitter
        points = points + np.random.normal(0, 1e-7, points.shape)

        try:
            hull = ConvexHull(points)
            polygon = [[float(points[v][0]), float(points[v][1])] for v in hull.vertices]
            polygon.append(polygon[0])

            popup_html = (
                f"<b>Year {y}</b><br>"
                f"<b>Cluster {cid}</b><br>"
                f"Points: {len(cluster_df)}<br>"
                + (f"Users: {cluster_df[USER_COL].nunique()}<br>" if USER_COL in cluster_df.columns else "")
                + (f"<b>Keywords:</b> {keywords_str}" if keywords_str else "")
            )

            polygons.append({
                "polygon": polygon,
                "popup": popup_html
            })
        except:
            continue

    if polygons:
        year_layers[str(y)] = polygons
        print(f"Year {y}: {len(polygons)} clusters")

# =========================
# Inject JS + Slider
# =========================
year_layers_json = json.dumps(year_layers)

slider_min = int(min(map(int, year_layers.keys()))) if year_layers else 0
slider_max = int(max(map(int, year_layers.keys()))) if year_layers else 0
slider_default = slider_min

map_var = m.get_name()  # IMPORTANT: nom réel de la map folium

html = f"""
<div style="
 position: fixed; bottom: 20px; left: 20px; z-index: 9999;
 background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px;
 box-shadow: 0 2px 8px rgba(0,0,0,0.2);
 font-family: Arial; font-size: 14px;
">
  <div><b>Année:</b> <span id="yearLabel">{slider_default}</span></div>
  <input id="yearSlider" type="range" min="{slider_min}" max="{slider_max}" value="{slider_default}" step="1" style="width: 240px;">
  <div style="font-size:12px; color:#666;">Glisse pour afficher les clusters DBSCAN</div>
</div>

<script>
const YEAR_LAYERS = {year_layers_json};
let currentLayerGroup = null;

function showYear(year) {{
  document.getElementById("yearLabel").innerText = year;

  if (currentLayerGroup) {{
    {map_var}.removeLayer(currentLayerGroup);
  }}

  const items = YEAR_LAYERS[String(year)] || [];
  currentLayerGroup = L.layerGroup();

  items.forEach(item => {{
    const poly = L.polygon(item.polygon, {{
      color: "red",
      fillColor: "red",
      fillOpacity: 0.2
    }});
    poly.bindPopup(item.popup);
    poly.addTo(currentLayerGroup);
  }});

  currentLayerGroup.addTo({map_var});
}}

const slider = document.getElementById("yearSlider");
slider.addEventListener("input", (e) => {{
  showYear(parseInt(e.target.value));
}});

showYear({slider_default});
</script>
"""

m.get_root().html.add_child(folium.Element(html))
m.save(OUT_HTML)
print("Carte générée :", OUT_HTML)