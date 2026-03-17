import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

# Créer un dossier pour les graphiques
os.makedirs("plots", exist_ok=True)

# 1. Chargement et Pré-traitement
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

results = []

# Fonction utilitaire pour plots PCA
def plot_pca(labels, title, filename):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8, legend='full')
    plt.title(title)
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=300)
    plt.close()

# Plot True Labels
plot_pca(y, "Vraies Étiquettes (Wine Dataset)", "true_labels")

# --- K-MEANS ---
# Elbow Method
inertias = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Nombre de clusters k')
plt.ylabel('Inertie')
plt.title('Méthode du Coude (Elbow Method) - K-Means')
plt.tight_layout()
plt.savefig(f"plots/kmeans_elbow.png", dpi=300)
plt.close()

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
y_kmeans = kmeans.fit_predict(X_scaled)
plot_pca(y_kmeans, "K-Means Clustering (k=3)", "kmeans_pca")
sil_kmeans = silhouette_score(X_scaled, y_kmeans)
results.append({'Algorithme': 'K-Means', 'Clusters': 3, 'Bruit': 'Non', 'Silhouette': sil_kmeans, 'Complexité': 'O(n * k * d * i)'})

# --- DBSCAN ---
dbscan = DBSCAN(eps=2.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
noise_db = list(y_dbscan).count(-1)
plot_pca(y_dbscan, f"DBSCAN Clustering (eps=2.5)", "dbscan_pca")
if n_clusters_db > 1:
    sil_db = silhouette_score(X_scaled, y_dbscan)
else:
    sil_db = "N/A"
results.append({'Algorithme': 'DBSCAN', 'Clusters': n_clusters_db, 'Bruit': f'Oui ({noise_db} points)', 'Silhouette': sil_db, 'Complexité': 'O(n \log n) ou O(n^2)'})

# --- HAC (Agglomerative Clustering) ---
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme (HAC)")
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))
plt.tight_layout()
plt.savefig(f"plots/hac_dendrogram.png", dpi=300)
plt.close()

hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hac = hac.fit_predict(X_scaled)
plot_pca(y_hac, "HAC Clustering (k=3)", "hac_pca")
sil_hac = silhouette_score(X_scaled, y_hac)
results.append({'Algorithme': 'HAC (Ward)', 'Clusters': 3, 'Bruit': 'Non', 'Silhouette': sil_hac, 'Complexité': 'O(n^3)'})

# --- GMM ---
gmm = GaussianMixture(n_components=3, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)
plot_pca(y_gmm, "Gaussian Mixture Models (k=3)", "gmm_pca")
sil_gmm = silhouette_score(X_scaled, y_gmm)
results.append({'Algorithme': 'GMM', 'Clusters': 3, 'Bruit': 'Non', 'Silhouette': sil_gmm, 'Complexité': 'O(n * k * d^2)'})

# --- OPTICS ---
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
y_optics = optics.fit_predict(X_scaled)

# Reachability Plot
space = np.arange(len(X_scaled))
reachability = optics.reachability_[optics.ordering_]
labels = optics.labels_[optics.ordering_]

plt.figure(figsize=(10, 5))
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    plt.plot(Xk, Rk, color, alpha=0.3)
plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
plt.ylabel('Distance de Reachability')
plt.title('Reachability Plot (OPTICS)')
plt.tight_layout()
plt.savefig(f"plots/optics_reachability.png", dpi=300)
plt.close()

plot_pca(y_optics, "OPTICS Clustering", "optics_pca")

n_clusters_opt = len(set(y_optics)) - (1 if -1 in y_optics else 0)
noise_opt = list(y_optics).count(-1)
if n_clusters_opt > 1:
    sil_opt = silhouette_score(X_scaled, y_optics)
else:
    sil_opt = "N/A"
results.append({'Algorithme': 'OPTICS', 'Clusters': n_clusters_opt, 'Bruit': f'Oui ({noise_opt} points)', 'Silhouette': sil_opt, 'Complexité': 'O(n \log n) ou O(n^2)'})

# Display Results Table
import pprint
print("=== RÉSULTATS COMPARATIFS ===")
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
