# %% [markdown]
# # Part 3, Task 2 — Unsupervised Machine Learning
#
# No accident dataset is available, so instead:
# - **K-means**: cluster *traffic conditions* themselves (hour, weather
#   severity, traffic volume) and interpret what each cluster represents.
# - **Association rule mining**: discretise time-of-day, weekday type and
#   weather, then mine rules that predict congestion level, reporting the
#   highest-lift rules.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import PART3_DIR, get_full_dataset

FIG_DIR = PART3_DIR / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = get_full_dataset()
print(df.shape)

# %% [markdown]
# ## K-means clustering of traffic conditions
#
# A weather severity score is built as a simple ordinal scale (0 = clear
# through 4 = severe/low-visibility), since raw category labels can't be fed
# to K-means directly and this keeps the ordering meaningful for a distance-
# based algorithm (an arbitrary integer-encoded category would falsely imply
# "Snow" is "closer" to "Rain" than to "Clear" only because of encoding
# order — an ordinal *severity* scale avoids that problem).

# %%
SEVERITY_MAP = {
    "Clear": 0,
    "Clouds": 1,
    "Drizzle": 2, "Mist": 2, "Haze": 2,
    "Rain": 3, "Fog": 3, "Smoke": 3,
    "Snow": 4, "Thunderstorm": 4, "Squall": 4,
}
df["weather_severity"] = df["weather_main"].map(SEVERITY_MAP)

cluster_features = df[["hour", "weather_severity", "traffic_volume"]].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

# Choose k via silhouette score across a small candidate range
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
print("Silhouette scores by k:", {k: round(v, 4) for k, v in sil_scores.items()})

best_k = max(sil_scores, key=sil_scores.get)
print(f"Selected k = {best_k} (highest silhouette score)")

# %%
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = df.groupby("cluster")[["hour", "weather_severity", "traffic_volume"]].mean().round(1)
cluster_summary["n_hours"] = df["cluster"].value_counts()
print(cluster_summary)

# %% [markdown]
# ### Cluster interpretation
# (Generated programmatically below from each cluster's mean hour/severity/
# volume, so the write-up always matches whatever `best_k` silhouette
# analysis actually selects.)

# %%
def describe_cluster(row) -> str:
    time_desc = (
        "overnight/off-peak" if row["hour"] < 6 or row["hour"] > 21
        else "daytime/commute-adjacent" if row["hour"] < 16
        else "evening peak"
    )
    weather_desc = (
        "clear/mild" if row["weather_severity"] < 1
        else "cloudy" if row["weather_severity"] < 2
        else "wet/reduced-visibility" if row["weather_severity"] < 3
        else "severe weather"
    )
    volume_desc = (
        "low" if row["traffic_volume"] < 2000
        else "moderate" if row["traffic_volume"] < 4500
        else "high"
    )
    return f"{time_desc} hours, typically {weather_desc} conditions, {volume_desc} traffic volume (~{row['traffic_volume']:.0f} vehicles/hr)"

for cluster_id, row in cluster_summary.iterrows():
    print(f"Cluster {cluster_id} ({int(row['n_hours'])} hours): {describe_cluster(row)}")

# %%
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df["hour"], df["traffic_volume"], c=df["cluster"], cmap="tab10", alpha=0.3, s=8)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Traffic volume")
ax.set_title(f"K-means clusters of traffic conditions (k={best_k})")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
ax.add_artist(legend1)
fig.tight_layout()
fig.savefig(FIG_DIR / "kmeans_clusters.png", dpi=150)
print(f"Saved cluster plot to {FIG_DIR / 'kmeans_clusters.png'}")

# %% [markdown]
# ## Association rule mining
#
# Discretise time-of-day, weekday type and weather into categorical "items",
# combine with the existing `congestion_category`, one-hot encode into a
# transaction table, then mine frequent itemsets and rules with Apriori —
# reporting the rules with the highest **lift** (how much more likely the
# consequent is, given the antecedent, versus chance) that predict a
# congestion level.

# %%
def time_of_day_bin(hour: int) -> str:
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    return "Evening"

basket_df = pd.DataFrame({
    "time_of_day": df["hour"].apply(time_of_day_bin),
    "day_type": np.where(df["is_weekend"] == 1, "Weekend", "Weekday"),
    "weather": df["weather_main"],
    "congestion": df["congestion_category"],
})

transactions = pd.get_dummies(basket_df)
print(f"Transaction table: {transactions.shape[0]} transactions, {transactions.shape[1]} possible items")

frequent_itemsets = apriori(transactions, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Keep only rules whose consequent is a congestion level — those are the
# actionable "under these conditions, expect this congestion level" rules.
congestion_cols = {f"congestion_{c}" for c in df["congestion_category"].unique()}
rules_congestion = rules[rules["consequents"].apply(lambda s: bool(set(s) & congestion_cols))]
rules_congestion = rules_congestion.sort_values("lift", ascending=False)

top_rules = rules_congestion.head(8)[["antecedents", "consequents", "support", "confidence", "lift"]]
top_rules["antecedents"] = top_rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
top_rules["consequents"] = top_rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
print(top_rules.round(3).to_string(index=False))

# %% [markdown]
# The table above is dominated by high-lift **low-congestion / nighttime**
# rules (statistically strong, but operationally unsurprising). The rules
# below isolate **High/Severe congestion** consequents specifically — these
# are the operationally actionable ones for the mobility team.

# %%
high_congestion_cols = {"congestion_High", "congestion_Severe"}
rules_high = rules[rules["consequents"].apply(lambda s: bool(set(s) & high_congestion_cols))]
rules_high = rules_high.sort_values("lift", ascending=False)

top_high_rules = rules_high.head(5)[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
top_high_rules["antecedents"] = top_high_rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
top_high_rules["consequents"] = top_high_rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
print(top_high_rules.round(3).to_string(index=False))

# %% [markdown]
# ### Plain-language interpretation
#
# Each rule reads as: *"When [antecedent conditions] hold, [consequent
# congestion level] is `lift` times more likely than it would be by chance."*
#
# The overall highest-lift rules are dominated by **Night + Low congestion**
# (statistically strong, but operationally unsurprising — nobody needs
# telling traffic is light at 3am). The second table isolates
# **High/Severe-congestion consequents** specifically, which is where the
# actionable insight is:
# - `{Weekday, Afternoon, Clouds} -> {Severe congestion}` — confidence ≈ 0.77,
#   lift ≈ 3.1. On a cloudy weekday afternoon, severe congestion is about
#   3x more likely than on a random hour, and it actually occurs on ~77% of
#   such afternoons.
# - `{Weekend, Afternoon, Clouds} -> {High congestion}` — confidence ≈ 0.85,
#   lift ≈ 3.4. Even weekend afternoons see elevated congestion under cloudy
#   skies, likely reflecting shopping/leisure travel rather than commuting.
#
# Both rules point the same direction as every other analysis in this
# capstone: **afternoon/commute-adjacent hours are the dominant congestion
# driver**, with cloudy weather as a secondary amplifying factor rather than
# a cause on its own — directly useful for the recommendation system (Task 5)
# and the MLOps alerting logic (Task 6).

# %%
if __name__ == "__main__":
    print(f"\nBest k for K-means: {best_k}")
    print(f"\nTop association rules:\n{top_rules.round(3).to_string(index=False)}")
