import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def perform_pca(X):
    """
    :param X: numpy ndarray

    PCA primeste un set de date si le reduce dimensionalitatea(proiectand intr-un nou spatiu)
    incercand sa pastreze cat mai multa varianta
    Aplica PCA pe datele standardizate și returnează toate valorile relevante:
    - Varianța explicată (Eigenvalues)
    - Componente principale (Eigenvectors)
    - Corelații factoriale (variabile observate - componente)
    - Cosinusuri (calitatea reprezentării variabilelor pe componente)
    - Contribuții (importanța fiecărei variabile în fiecare componentă)
    - Comunalități și varianță specifică
    - Scorurile PCA - noul spatiu vectorial sub forma de ndarray
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    """
    standardizare pentru a ajunge la medie 0 si dispersie 1
    """

    pca_full = PCA()
    scores_full = pca_full.fit_transform(X_scaled)
    """
    PCA inițial cu toate componentele pentru analiza Eigenvalues
    """

    eigenvalues = pca_full.explained_variance_  # Valorile proprii (câtă informație conține fiecare componentă)
    kaiser_mask = eigenvalues > 1  # Criteriul Kaiser (reținerea componentelor cu varianță > 1)
    n_comp_kaiser = np.sum(kaiser_mask)  # Numărul optim de componente conform Kaiser

    """
    eigenvalues = cat de multa varianta explica fiecare componenta principala
    kaiser_mask = vector de dim eigenvalues, kaiser_mask[i] = 1 daca eigenvalue[i] > 1 si 0 altfel
    n_comp_kaiser = numarul de componente care au eigenvalue > 1
    """

    pca_kaiser = PCA(n_components=n_comp_kaiser)
    scores_kaiser = pca_kaiser.fit_transform(X_scaled)

    """
    fit_transform(X_scaled) aplică PCA și proiectează datele în noul spațiu redus definit de numărul de componente selectat.
    ex - daca datele initiale erau:
    Observație	Var1 (X1)	Var2 (X2)	Var3 (X3)
    1	           0.8	       -1.2	       1.5
    2	           -0.5	       0.7	       -1.3
    3	           1.3	       -0.8	       0.9
    4	           -1.1	       1.3	       -0.5
    
    si n_components = 2, atunci:
    scores_kaiser = 
    Observație	 PC1	 PC2
    1	        1.25	-0.67
    2	        -0.90	0.45
    3	        1.05	-0.53
    4	        -1.40	0.75        (doar valorile, ndarray)
    """

    loadings = pca_full.components_
    corelatii = loadings * np.sqrt(eigenvalues[:, np.newaxis])  # Corelații variabile - componente

    """
    exemplu - date initiale:
    Observație	X1	X2	X3
    1	        4	2	5
    2	        7	5	9   
    3	        1	0	3
    
    loadings:
    Variabilă	PC1	 PC2	PC3
    X1	        0.7	 -0.4	0.5
    X2	        0.6	 0.8	-0.1
    X3	        0.3	 -0.2	0.9
    
    eigenvalues: [2.5, 1.2, 0.3]
    
    corelatii = loadings * sqrt(eigenvalues)
    in cazul nostru:
    Variabilă	    PC1 (1.58)	        PC2 (1.09)	        PC3 (0.55)
    X1	        0.7 × 1.58 = 1.11	-0.4 × 1.09 = -0.44	    0.5 × 0.55 = 0.28
    X2	        0.6 × 1.58 = 0.95	0.8 × 1.09 = 0.87	    -0.1 × 0.55 = -0.06
    X3	        0.3 × 1.58 = 0.47	-0.2 × 1.09 = -0.22	    0.9 × 0.55 = 0.50
    
    interpretare: corelatie -> 1 => variabila bine reprezentata de acea PC
                  corelatie -> 0 => variabila nu este bine explicata de acea PC
                  
    np.newaxis este folosit pentru a extinde dimensiunea lui eigenvalues, transformându-l într-un array bidimensional (3,1),
    astfel încât să fie compatibil cu loadings pentru operația de broadcasting.
    corelatii = corelațiile dintre variabilele observate și componentele principale
    """

    cos2 = loadings ** 2
    cos2 /= cos2.sum(axis=1, keepdims=True)  # Normalizare (suma pe componentă = 1)

    """
    cos2 ne arată cât de bine este proiectată o variabilă pe fiecare componentă (% din cat e proiectata pe toate componentele)
    ridicam la patrat fiecare loading si il impartim la suma pe linie pentru a avea suma pe linie 1 (normalizare)
    """

    comunalitati = cos2.sum(axis=0)
    """
    comunalitate = cat de bine e reprez in noul spatiu de toate PC (intre 0 si 1)
    """

    contributii = (loadings ** 2) / (eigenvalues[:, np.newaxis])
    contributii /= np.sum(contributii, axis=1, keepdims=True)  # Normalizare la suma 1

    """
    contributia fiecarei variabile la fiecare PC (procentul de varianță pe care acea variabilă îl aduce componentei respective)
    ~ce procent are fiecare variabila in explicarea fiecarei PC
    """

    return {
        "pca_full": pca_full,
        "pca_kaiser": pca_kaiser,
        "scores_full": scores_full,
        "scores_kaiser": scores_kaiser,
        "eigenvalues": eigenvalues,
        "corelatii": corelatii,
        "cos2": cos2,
        "comunalitati": comunalitati,
        "contributii": contributii
    }


# ===================== Vizualizări =====================

def plot_variance(pca):
    """ Grafic cu varianța explicată de fiecare componentă """
    explained_variance = pca.explained_variance_
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='blue', alpha=0.7)
    plt.xlabel("Componente Principale")
    plt.ylabel("Varianță Explicată")
    plt.title("Varianta PCA (Eigenvalues)")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.show()


def plot_cumulative_variance(pca):
    """ Grafic cu varianța cumulativă pentru a determina numărul optim de componente """
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o', linestyle='-')
    plt.axhline(y=0.9, color='r', linestyle='--', label="90% Varianță Explicată")
    plt.axhline(y=0.8, color='g', linestyle='--', label="80% Varianță Explicată")
    plt.xlabel("Număr Componente Principale")
    plt.ylabel("Varianță Cumulativă")
    plt.title("Varianta PCA Cumulativă")
    plt.legend()
    plt.show()


def plot_correlation_circle(corelatii):
    """ Cerc corelații pentru primele două componente principale """
    pc1 = corelatii[0, :]
    pc2 = corelatii[1, :]

    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, linestyle='dashed', color='black')
    plt.axhline(0, color='grey', linewidth=1)
    plt.axvline(0, color='grey', linewidth=1)

    plt.scatter(pc1, pc2)
    for i, (x, y) in enumerate(zip(pc1, pc2)):
        plt.text(x, y, f"Var {i + 1}", fontsize=12)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cercul Corelațiilor (PC1 vs PC2)")
    plt.show()


def plot_correlation_heatmap(corelatii):
    """ Corelogramă corelații factoriale """
    plt.figure(figsize=(8, 5))
    sns.heatmap(corelatii, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Corelogramă Corelații Factoriale")
    plt.xlabel("Variabile")
    plt.ylabel("Componente")
    plt.show()


def plot_communalities_heatmap(comunalitati):
    """ Corelogramă comunalități """
    plt.figure(figsize=(8, 5))
    sns.heatmap(comunalitati.reshape(1, -1), annot=True, cmap="coolwarm", cbar=True, linewidths=0.5)
    plt.title("Corelogramă Comunalități")
    plt.xlabel("Variabile")
    plt.ylabel("Comunalitate")
    plt.show()


def plot_pca_scores(scores):
    """ Scatter plot pentru primele două scoruri PCA """
    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, edgecolors='black')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("Scor PCA 1")
    plt.ylabel("Scor PCA 2")
    plt.title("Scoruri PCA (PC1 vs PC2)")
    plt.show()


# ===================== Aplicația PCA pe Datele Iris =====================
iris = load_iris()
X = iris.data

# Aplicăm PCA
pca_results = perform_pca(X)

# Ploturi & Analiză
plot_variance(pca_results["pca_full"])
plot_cumulative_variance(pca_results["pca_full"])
plot_correlation_circle(pca_results["corelatii"])
plot_correlation_heatmap(pca_results["corelatii"])
plot_communalities_heatmap(pca_results["comunalitati"])
plot_pca_scores(pca_results["scores_full"])
