import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

"""
Think of EFA as a method to describe hidden patterns in data
example: survey with 6 questions.

At first, it seems like six different variables. But if you look closely:
Q1, Q2, Q3 seem to be related to technology interest.
Q4, Q5, Q6 seem to be related to social anxiety.
EFA helps us uncover these hidden factors (latent variables):

Factor 1 (Tech Enthusiasm): Groups Q1, Q2, Q3.
Factor 2 (Social Anxiety): Groups Q4, Q5, Q6.

Why Not Just Use Correlations?
Correlation only shows if two variables are related.
EFA goes deeper and groups multiple variables into hidden factors.
"""

def perform_efa(data):
    """
    Performs Exploratory Factor Analysis (EFA) and returns results in a dictionary.
    Also provides visualizations for various factor analysis metrics.

    :param data: DataFrame with observed variables.
    :return: Dictionary with all.py EFA results.
    """

    # ===================== 1. Bartlett & KMO Tests =====================
    ##Analiza factorabilității - Bartlett
    bartlett_test, p_value = calculate_bartlett_sphericity(data)
    #tests if variables are correlated enough for FA
    #if p-val < 0.05 => EFA is suitable

    ##Analiza factorabilității - KMO
    kmo_all, kmo_model = calculate_kmo(data)
    #tests if sampling is adequate, same as bartlett
    #if KMO > 0.6 => EFA is reliable

    # ===================== 2. Determine Number of Factors =====================
    fa = FactorAnalyzer(n_factors=data.shape[1], rotation=None)
    #incepem analiza cu toate variabilele
    fa.fit(data) #calculeaza toate draciile
    ev, v = fa.get_eigenvalues()
    #ev = eigenvalues (how much variance each factor explains)
    #v = explained variance proportions (% of total variance explained by each factor)
    num_factors = sum(ev > 1)  # Select factors where Eigenvalue > 1

    # ===================== 3. Factor Analysis (With & Without Rotation) =====================
    fa_no_rot = FactorAnalyzer(n_factors=num_factors, rotation=None).fit(data)
    fa_rot = FactorAnalyzer(n_factors=num_factors, rotation="varimax").fit(data)
    #varimax le face perpendiculare, oblimin e pt factori care ar fi corelati intre ei si ar putea fi gen /_
    #nerotit => compararea rezultatelor pentru a decide dacă rotația îmbunătățește interpretarea factorilor

    """
    Factorii extrași fără rotație sunt greu de interpretat.
    Unele variabile au încărcături mari pe mai mulți factori.
    Nu este clar ce variabilă aparține cărui factor.
    
    Factorii sunt reorientați pentru a clarifica structura.
    Fiecare variabilă tinde să se coreleze puternic cu un singur factor.
    Interpretarea devine mult mai ușoară.
    """

    # ===================== 4. Factor Variance =====================
    #Calcul varianță factori (cu/fără rotație)
    variance_no_rot = fa_no_rot.get_factor_variance()
    variance_rot = fa_rot.get_factor_variance()
    """
    get_factor_variance intoarce 3 liste:
    Eigenvalues	= Cantitatea de variație explicată de fiecare factor (valori proprii)
    Proportion of Variance = Ponderea variației explicate de fiecare factor (în procente)
    Cumulative Variance	= Variația totală explicată de toți factorii împreună (cumsum)
    """

    # ===================== 5. Factor Loadings (Correlations) =====================
    ##Calcul corelații factoriale (cu/fără rotație)
    factor_corr_no_rot = np.corrcoef(fa_no_rot.loadings_.T)
    factor_corr_rot = np.corrcoef(fa_rot.loadings_.T)
    """
    loadings_ = coeficienți care arată cât de mult fiecare variabilă este influențată de fiecare factor
    Variabilă	                            Factor 1	Factor 2
    Q1 (Îmi place matematica)	            0.80	    0.10
    Q2 (Sunt bun la programare)	            0.75	    0.05
    Q3 (Îmi place să citesc știință)	    0.85	    0.15
    Q4 (Sunt timid în public)	            0.10	    0.85
    Q5 (Nu îmi place să vorbesc în grupuri)	0.05	    0.80
    .T => transpunem matricea asta
    corrcoef (exemplu output) => 
    Factor Correlations:
    [[ 1.00  0.45]
    [ 0.45  1.00]]
    Factorii sunt corelați între ei (r = 0.45).
    Dacă acest număr este mare (>0.7), probabil avem un singur factor și nu doi.
    """

    # ===================== 6. Communalities & Specific Variance =====================
    ##Calcul comunalități și varianță specifică
    communalities = fa_rot.get_communalities()
    specific_variance = 1 - communalities

    communalities_df = pd.DataFrame({
        "Variable": data.columns,
        "Communalities": communalities,
        "Specific Variance": specific_variance
    })

    """
    Communalities = Câtă variație a fiecărei variabile este explicată de factorii extrași.
    aproape de 1 → variabila este foarte bine explicată de factori.
    aproape de 0 → variabila nu este bine explicată și poate fi eliminată.
    
    Variabilă	                            Factor 1	Factor 2	    Communality             Specific Variance (1 - Communality)
    Q1 (Îmi place matematica)	            0.80	    0.10	    0.80² + 0.10² = 0.65                0.36
    Q2 (Sunt bun la programare)	            0.75	    0.05	    0.75² + 0.05² = 0.56                0.44
    Q3 (Îmi place să citesc știință)	    0.85	    0.15	    0.85² + 0.15² = 0.76                0.24
    Q4 (Sunt timid în public)	            0.10	    0.85	    0.10² + 0.85² = 0.74                0.26
    Q5 (Nu îmi place să vorbesc în grupuri)	0.05	    0.80	    0.05² + 0.80² = 0.64                0.35
    
    Specific Variance = Câtă variație rămâne neexplicată (specifică fiecărei variabile, zgomot de date)
    Specific Variance mare → variabila are mult zgomot, poate nu este relevantă.
    Specific Variance mic → variabila este bine explicată de factori.
    """

    # ===================== 7. Factor Scores =====================
    ##Calcul scoruri (cu/fără rotație)
    factor_scores = fa_rot.transform(data)
    """
    coordonatele fiecărei observații (instanțe) în noul spațiu al factorilor

    Date inițiale (6 variabile pentru fiecare persoană):
    Persoană	Q1 (Math)	Q2 (Coding)	        Q3 (Reading)	Q4 (Public Speaking)	Q5 (Social Skills)	Q6 (Group Work)
    P1	            5	            4	            5	                2	                    3	              4
    P2	            3	            3	            4	                4	                    5	              5
    
    După transformare (fa.transform()):
    Persoană	Factor 1 (Inteligență Tehnică)	Factor 2 (Abilități Sociale)
    P1	                2.5	                                -1.2
    P2	                1.0	                                2.2
    
    Formula generală: Factor Scores = 𝑋 x 𝑊
    X = matricea de date standardizate
    W = matricea factorilor (loadings_)
    """

    # ===================== Return Dictionary =====================
    return {
        "bartlett_test": bartlett_test,
        "bartlett_p_value": p_value,
        "kmo_model": kmo_model,
        "num_factors": num_factors,
        "factor_variance_no_rotation": variance_no_rot,
        "factor_variance_rotation": variance_rot,
        "factor_loadings_no_rotation": fa_no_rot.loadings_,
        "factor_loadings_rotation": fa_rot.loadings_,
        "factor_correlation_no_rotation": factor_corr_no_rot,
        "factor_correlation_rotation": factor_corr_rot,
        "communalities": communalities_df,
        "factor_scores": factor_scores
    }

# ===================== Visualization Functions =====================

##Trasare corelogramă corelații factoriale (cu/fără rotație)
def plot_factor_correlation(correlation_matrix, title):
    """Plots a heatmap of the factor correlation matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

##Trasare cercul corelațiilor (cu/fără rotație)
def plot_correlation_circle(loadings, data_columns, title):
    """Plots the correlation circle using factor loadings."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)

    for i, var in enumerate(data_columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='b', alpha=0.5)
        plt.text(loadings[i, 0], loadings[i, 1], var, fontsize=12)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.title(title)
    plt.show()

##Trasare corelogramă comunalități și varianță specifică
def plot_communalities(communalities_df):
    """Plots a heatmap for communalities and specific variance."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(communalities_df.set_index("Variable").T, annot=True, cmap="coolwarm")
    plt.title("Communalities & Specific Variance")
    plt.show()

##Trasare plot scoruri
def plot_factor_scores(scores, title):
    """Plots the factor scores in a scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.title(title)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.show()

# ===================== Run EFA on Sample Data =====================
np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 6), columns=["A", "B", "C", "D", "E", "F"])
efa_results = perform_efa(data)

# ===================== Print Key Results =====================
print(f"Bartlett Test: {efa_results['bartlett_test']:.2f}, p-value: {efa_results['bartlett_p_value']:.5f}")
print(f"KMO: {efa_results['kmo_model']:.3f}")
print(f"Optimal Number of Factors: {efa_results['num_factors']}")
print("\nCommunalities & Specific Variance:\n", efa_results["communalities"])

# ===================== Generate Plots =====================
plot_factor_correlation(efa_results["factor_correlation_rotation"], "Factor Correlation Matrix (Rotated)")
plot_correlation_circle(efa_results["factor_loadings_rotation"], data.columns, "Correlation Circle (Rotated)")
plot_communalities(efa_results["communalities"])
plot_factor_scores(efa_results["factor_scores"], "Factor Scores Plot")
