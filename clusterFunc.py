import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import load_iris

def perform_clustering(X, k_manual=3):
    """
    Performs Hierarchical Clustering and returns results in a dictionary.
    Also provides visualizations for various clustering metrics.
    HCA imparte datele initiale (X - ndarray) pe clustere cat mai diferite intre ele si cat mai similare intra cluster
    Clusterele NU sunt etichetate de la început.
    Algoritmul HCA le descoperă automat, fără a avea informații despre clasele reale din dataset
    Spre deosebire de LDA (Linear Discriminant Analysis), care are nevoie de etichete inițiale și optimizează separarea dintre clasele
    cunoscute, HCA doar descoperă structura naturală a datelor
    :param X: ndarray - Feature matrix for clustering.
    :param k_manual: Manually specified number of clusters.
    :return: Dictionary with all.py clustering results.
    """

    # ===================== 1. Compute Hierarchical Clustering (Linkage Matrix) =====================
    ##Calcul ierarhie (matricea ierarhie)
    linkage_matrix = linkage(X, method="ward")  # Ward method minimizes variance
    """
    output: matrice cu n-1 linii si 4 coloane:
    c1: cluster 1(punct sau cluster anterior)
    c2: cluster 2(punct sau cluster anterior)
    c3: distanta dintre c1 si c2. Se calculeaza folosind metoda Ward
    c4: nr puncte din clusterul nou format 
    """

    # ===================== 2. Find Optimal Partition (Using Linkage Distances) =====================
    ##Calcul partiție optimală (metoda Elbow pe distanțele de agregare)
    distances = linkage_matrix[:, 2]  #c3 din matricea de mai sus
    gaps = np.diff(distances)  # diferentele succesive din distances
    optimal_k = np.argmax(gaps) + 1  # indexul unde diferența este cea mai mare

    # ===================== 3. Compute Partitions =====================
    ##Calcul partiție oarecare (număr prestabilit de clusteri)
    labels_optimal = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
    labels_manual = fcluster(linkage_matrix, k_manual, criterion='maxclust')

    """
    fcluster() ia matricea de legături (linkage_matrix) și grupează instanțele în optimal_k clustere
    criterion='maxclust' inseamnă că forțăm exact optimal_k clustere.
    labels_optimal vector de etichete care spune în ce cluster a fost atribuită fiecare instanță. (1d array)
    """

    # ===================== 4. Compute Silhouette Scores =====================
    ##Calcul indecși Silhouette la nivel de partiție și de instanțe
    silhouette_optimal = silhouette_score(X, labels_optimal)
    silhouette_manual = silhouette_score(X, labels_manual)
    silhouette_values_optimal = silhouette_samples(X, labels_optimal)
    silhouette_values_manual = silhouette_samples(X, labels_manual)

    """
    Silhouette Score măsoară cât de bine sunt grupate punctele în clustere și cât de bine sunt separate între ele
    Scorurile mari înseamnă clustering mai bun, iar cele mici indică suprapunere între clustere
    s(i) = d1(i,cel mai apr cluster) - d2(i, restul pct din cluster) / max(d1,d2) ; intre -1 si 1
    s(i) -> 1 => pct bine incadrat in cluster
    s(i) -> 0 => pct la granita dintre clustere
    s(i) < 0 => pct este in clusterul gresit
    Returneaza un singur număr (medie pe toate instanțele)
    
    Silhouette samples returnează un vector cu scorurile Silhouette pentru fiecare observație
    """

    # ===================== Return Dictionary =====================
    return {
        "linkage_matrix": linkage_matrix,
        "optimal_k": optimal_k,
        "labels_optimal": labels_optimal,
        "labels_manual": labels_manual,
        "silhouette_optimal": silhouette_optimal,
        "silhouette_manual": silhouette_manual,
        "silhouette_values_optimal": silhouette_values_optimal,
        "silhouette_values_manual": silhouette_values_manual
    }

# ===================== Visualization Functions =====================

##Trasare plot dendrogramă cu evidențierea partiției (optimală și partiție-k)
def plot_dendrogram(linkage_matrix, labels, title):
    """ Plots the dendrogram with clusters colored. """
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=labels, color_threshold=np.max(linkage_matrix[:, 2]) * 0.7)
    plt.title(title)
    plt.xlabel("Instances")
    plt.ylabel("Linkage Distance")
    plt.show()

##Trasare plot Silhouette partiție (optimală și partiție-k)
def plot_silhouette(X, labels, title):
    """ Plots silhouette analysis for given partition labels. """
    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10
    plt.figure(figsize=(8, 6))

    for i in np.unique(labels):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()
        y_upper = y_lower + len(cluster_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals)
        y_lower = y_upper + 10

    plt.axvline(np.mean(silhouette_vals), linestyle="--", color="red")
    plt.title(title)
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Instances")
    plt.show()

##Trasare histograme clusteri pentru fiecare variabilă observată (partiție optimală și partiție-k)
def plot_cluster_histograms(X, labels, title):
    """ Plots histograms for each variable in the dataset, grouped by cluster. """
    df = pd.DataFrame(X, columns=[f"Var {i+1}" for i in range(X.shape[1])])
    df["Cluster"] = labels

    for column in df.columns[:-1]:  # Exclude the cluster column
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, hue="Cluster", kde=True, palette="Set1", alpha=0.5)
        plt.title(f"{title} - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

##Trasare plot partiție în axe principale (optimală și partiție-k)
def plot_pca_clusters(X, labels, title):
    """ Projects data using PCA and plots the clusters in 2D principal component space. """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set1", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend(title="Cluster")
    plt.show()

# ===================== Run Clustering on Sample Data =====================
iris = load_iris()
X = iris.data

clustering_results = perform_clustering(X, k_manual=3)

# ===================== Print Key Results =====================
print(f"Optimal Number of Clusters: {clustering_results['optimal_k']}")
print(f"Silhouette Score (Optimal k={clustering_results['optimal_k']}): {clustering_results['silhouette_optimal']:.4f}")
print(f"Silhouette Score (Manual k=3): {clustering_results['silhouette_manual']:.4f}")

# ===================== Generate Plots =====================
plot_dendrogram(clustering_results["linkage_matrix"], clustering_results["labels_optimal"], f"Dendrogram (Optimal k={clustering_results['optimal_k']})")
plot_dendrogram(clustering_results["linkage_matrix"], clustering_results["labels_manual"], f"Dendrogram (Manual k=3)")

plot_silhouette(X, clustering_results["labels_optimal"], f"Silhouette Analysis (Optimal k={clustering_results['optimal_k']})")
plot_silhouette(X, clustering_results["labels_manual"], f"Silhouette Analysis (Manual k=3)")

plot_cluster_histograms(X, clustering_results["labels_optimal"], f"Histograms (Optimal k={clustering_results['optimal_k']})")
plot_cluster_histograms(X, clustering_results["labels_manual"], f"Histograms (Manual k=3)")

plot_pca_clusters(X, clustering_results["labels_optimal"], f"PCA Projection (Optimal k={clustering_results['optimal_k']})")
plot_pca_clusters(X, clustering_results["labels_manual"], f"PCA Projection (Manual k=3)")
