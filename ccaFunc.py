import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2
from sklearn.datasets import load_iris

def perform_cca(X, Y):
    """
    CCA: relațiile dintre două seturi de variabile => cât de bine sunt corelate între ele
    -gaseste combinatii liniare de variabile intre X si Y care sunt maxim corelate
    -creează noi variabile latente numite "componente canonice", care sumarizează informația importantă.
    -ne arată cât de puternic sunt legate cele două seturi
    Student	Matematică	Limbaj	Știință	Salariu	Promoții	Nivel Post
    A	        85	     78	       92	70.000	    2	        3
    B	        76	     74	       80	50.000	    1	        2
    C	        90	     88	       96	100.000	    5	        5
    D	        60	     65	       70	40.000	    0	        1
    E	        82	     80	       88	75.000	    3	        4
    Output:
    Prima componentă canonică:
    din X: 0.8 * Matematică + 0.7 * Știință - 0.2 * Limbaj
    din Y: 0.9 * Salariu + 0.85 * Promoții + 0.7 * Nivel Post
    Canonical Correlation: 0.92
    Înseamnă că prima combinație de examene explică 92% din variația succesului profesional

    :param X: First set of variables.
    :param Y: Second set of variables.
    :return: Dictionary with all.py CCA results.
    """

    # ===================== 1. Compute Canonical Correlation Analysis =====================
    n_components = min(X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)

    """
    nr componente este egal cu minimul dintre nr variabile din X si cel din Y
    CCA under the hood:
    cauta a1, a2, a3, b1, b2, b3 astfel incat
    Xc = a1X1 + a2X2 + a3X3
    Yc = b1Y1 + b2Y2 + b3Y3
    sa fie cat mai corelate
    exemplu:
    X (original)   X_c (nou)	   Y (original)	    Y_c (nou)
     85 78 92	     2.1	        70.000 2 3	       1.9
     76 74 80	     1.3	        50.000 1 2	       1.2
     90 88 96	     3.0	        100.000 5 5	       2.8
     60 65 70	     0.8	        40.000 0 1	       0.6
    """

    # ===================== 2. Compute Canonical Correlations =====================
    canonical_correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    r2 = np.array(canonical_correlations) ** 2  # Squared correlations

    """
    calculeaza corelatiile dintre elementele X_Ci, Y_Ci
    corrcoef returneaza [1, 0,95]
                        [0,95, 1] si trebuie sa luam elem de pe rand 0 col 1 sau rand 1 col 0 pentru corelatia dintre ele
    le ridicam la patrat pt a le pregati pt Bartlett
    """

    # ===================== 3. Bartlett's Test for Canonical Roots =====================
    def bartlett_test(X, Y, r2):
        n = X.shape[0] #numar de observatii
        p, q = X.shape[1], Y.shape[1] #numar de variabile din X si din Y
        s = min(p, q) #numar maxim de comp canonice
        test_stats, p_values = [], []

        for i in range(s):
            lambda_product = np.prod(1 - r2[i:])
            chi_square = -(n - 1 - (p + q + 1) / 2) * np.log(lambda_product) #misto formula varutu sigur invat asta
            df = (s - i) * (s - i)
            p_values.append(1 - chi2.cdf(chi_square, df))
            test_stats.append(chi_square)

        return pd.DataFrame({"Canonical Root": range(1, s + 1), "Chi-Square": test_stats, "p-Value": p_values})

    bartlett_results = bartlett_test(X, Y, r2)

    """
    din pacate nu pot simplifica mai mult (ca la EFA) folosind o biblioteca externa
    np.prod(1 - r2[i:]) -> r2[i:] : luam doar corelatiile ramase de la i
                        -> 1 - r2[i:] : partea neexplicata a variantei
                        -> np.prod : inmultim toate aceste valori
    o sa ma rog sa nu dea asa cv
    """

    # ===================== 4. Compute Correlations Between Observed & Canonical Variables =====================
    correlations = np.corrcoef(np.hstack((X, Y, X_c, Y_c)).T)
    correlation_matrix = correlations[:X.shape[1] + Y.shape[1], -n_components:]
    correlation_df = pd.DataFrame(correlation_matrix, index=X.columns.tolist() + Y.columns.tolist(),
                                  columns=[f"Canonical Var {i+1}" for i in range(n_components)])

    # ===================== 5. Compute Explained Variance & Redundancy =====================
    X_var, Y_var = np.var(X, axis=0).sum(), np.var(Y, axis=0).sum()
    X_c_var, Y_c_var = np.var(X_c, axis=0), np.var(Y_c, axis=0)
    explained_variance = (X_c_var / X_var, Y_c_var / Y_var)
    redundancy = (explained_variance[0] * canonical_correlations, explained_variance[1] * canonical_correlations)

    # ===================== Return Dictionary =====================
    return {
        "n_components": n_components,
        "canonical_scores_X": X_c,
        "canonical_scores_Y": Y_c,
        "canonical_correlations": canonical_correlations,
        "bartlett_test": bartlett_results,
        "correlation_matrix": correlation_df,
        "explained_variance": explained_variance,
        "redundancy": redundancy
    }

# ===================== Visualization Functions =====================

def plot_correlation_circle(correlation_df):
    """Plots correlation circle between observed & canonical variables."""
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='grey', linewidth=1)
    plt.axvline(0, color='grey', linewidth=1)

    for var, (x, y) in zip(correlation_df.index, correlation_df.values):
        plt.arrow(0, 0, x, y, color='b', alpha=0.5)
        plt.text(x, y, var, fontsize=12)

    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Canonical Variable 1")
    plt.ylabel("Canonical Variable 2")
    plt.title("Correlation Circle")
    plt.show()

def plot_correlation_heatmap(correlation_df):
    """Plots a heatmap for observed vs canonical variable correlations."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlogram - Observed vs Canonical Variables")
    plt.show()

def plot_biplot(X_c, Y_c):
    """Plots instances in the canonical variable space."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_c[:, 0], Y_c[:, 0], c='blue', alpha=0.7, label="Canonical Variable 1")
    plt.scatter(X_c[:, 1], Y_c[:, 1], c='red', alpha=0.7, label="Canonical Variable 2")

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Canonical Component 1")
    plt.ylabel("Canonical Component 2")
    plt.title("Biplot - Instances in Canonical Space")
    plt.legend()
    plt.show()

# ===================== Run CCA on Sample Data =====================
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
X, Y = data.iloc[:, :2], data.iloc[:, 2:]  # Split into two variable sets

cca_results = perform_cca(X, Y)

# ===================== Print Key Results =====================
print(f"Optimal Number of Components: {cca_results['n_components']}")
print(f"Canonical Correlations: {cca_results['canonical_correlations']}")
print("\nBartlett’s Test Results:\n", cca_results["bartlett_test"])
print("\nCorrelations Between Observed and Canonical Variables:\n", cca_results["correlation_matrix"])
print(f"\nExplained Variance (X, Y): {cca_results['explained_variance']}")
print(f"Redundancy (X, Y): {cca_results['redundancy']}")

# ===================== Generate Plots =====================
plot_correlation_circle(cca_results["correlation_matrix"])
plot_correlation_heatmap(cca_results["correlation_matrix"])
plot_biplot(cca_results["canonical_scores_X"], cca_results["canonical_scores_Y"])
