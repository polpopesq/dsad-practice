import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

mortalitate = pd.read_csv('DateIN/Mortalitate.csv')

mortalitate.set_index(["Tara"], inplace=True)
mortalitate_arr = mortalitate.to_numpy()

X_std = StandardScaler().fit_transform(mortalitate_arr)

pca = PCA()
pca.fit(X_std)

eigenvalues = pca.explained_variance_
n_components = np.sum(eigenvalues > 1)

pca_optimal = PCA(n_components=n_components)
X_pca = pca_optimal.fit_transform(X_std)

df_scores = pd.DataFrame(X_pca)
df_scores.rename(columns={0: 'PC1', 1: "PC2", 2: "PC3"}, inplace=True)

df_scores.to_csv('dateOUT/PCASummary.csv')

print(df_scores.head())

def plot_2axe(X):
    x_vals = X[:,0]
    y_vals = X[:,1]
    plt.figure(figsize=(10,10))
    plt.scatter(x_vals, y_vals)
    plt.axvline(0, color='black', linestyle='--')
    plt.ylabel("PC2")
    plt.xlabel("PC1")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.show()

plot_2axe(X_pca)
print("Procent varianta explicata de cele 2 componente: ", (pca_optimal.explained_variance_ratio_[0] + pca_optimal.explained_variance_ratio_[1]) * 100, "%")