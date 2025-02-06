import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from narwhals import DataFrame

np.set_printoptions(suppress=True)


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

alcohol =  pd.read_csv('DateIN/alcohol.csv')

alcohol.fillna(alcohol.select_dtypes(include='number').mean(), inplace=True)
alcohol.drop(["Code"], axis=1, inplace=True)
alcohol.set_index(alcohol["Entity"], inplace=True)
alcohol.drop(["Entity"], axis=1, inplace=True)
print(alcohol.head())

data = alcohol.to_numpy()
linkage_matrix = linkage(data, method='ward')
#print(linkage_matrix)

res1 = pd.DataFrame(linkage_matrix)
res1.rename(columns={0: "ID Cluster 1", 1: "ID Cluster 2", 2: "Distanta", 3: "Numar puncte cluster"}, inplace=True)
print(res1)

distances = linkage_matrix[:, :2]
gaps = np.diff(distances)
optimal_k = np.argmax(gaps) + 1

opt_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')

for i in range(len(opt_labels)):
    print("Punctul ", i, "se afla in clusterul ", opt_labels[i])

#Dendrogramma
plt.figure(figsize=(10,5))
dendrogram(linkage_matrix, labels = opt_labels)
plt.show()