import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris


def perform_discriminant_analysis(X, y, model_type="LDA"):
    """
    Aplică Linear Discriminant Analysis (LDA) sau Quadratic Discriminant Analysis (QDA) pe setul de date.
    LDA separa datele cu o dreapta (hiperplan in MD), QDA separa datele cu o curba (suprafete de separare curbate in MD)
    LDA reduce dimensiunea N -> C-1 dimens
    QDA separa clasele in spatiul original
    Returnează modelul antrenat, scorurile, predicțiile și evaluarea.

    :param X: Variabilele independente (features)
    :param y: Variabila dependentă (etichetele claselor)
    :param model_type: "LDA" pentru model liniar, "QDA" pentru model bayesian
    :return: Dicționar cu modelul, scorurile, predicțiile și evaluarea
    """

    # 1. Împărțim datele în set de antrenare și testare
    # test_size: procent pt testare
    # stratify: se asigură că distribuția claselor din y este proporțională în ambele subseturi.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    """
        Param iesire:
        X_train – datele de antrenament (features)
        X_test – datele de testare (features)
        y_train – etichetele claselor pentru antrenament
        y_test – etichetele claselor pentru testare
    """

    # 2. Alegem modelul (LDA sau QDA)
    if model_type == "LDA":
        model = LDA()
    else:
        model = QDA()

    # 3. Antrenăm modelul
    model.fit(X_train, y_train)

    """
    În timpul antrenării, LDA:
        Calculează media fiecărei clase.
        Calculează variația intra-clasă și inter-clasă.
        Determină direcțiile optime pentru separarea claselor.
    """

    # 4. Calculăm scorurile discriminante pentru setul de antrenare
    ## Calcul scoruri discriminante model liniar/bayesian
    if model_type == "LDA":
        scores = model.transform(X_train)  # QDA nu are transform
    else:
        scores = None  # QDA nu creează un nou spațiu de proiecție

    """
        model.transform(X_train) → Proiectează fiecare observație din X_train pe noile axe discriminante găsite de LDA.
        Rezultatul (scores) → O matrice unde fiecare rând reprezintă un scor discriminant pentru fiecare instanță din X_train.

        Ce reprezintă aceste scoruri?

        Sunt coordonatele fiecărei instanțe în spațiul LDA.
        Permit vizualizarea separării claselor.
        Pot fi folosite ca features reduse dimensional pentru modele de clasificare.

        Numărul maxim de dimensiuni (componente LDA) este min(nr clase - 1, nr var initiale)
        scores.shape[1] ne arată câte dimensiuni au fost păstrate
        """

    # 5. Facem predicții pe setul de testare
    ## Predicția în setul de testare model liniar/bayesian
    y_pred = model.predict(X_test)

    """
        model.predict(X_test) Folosește modelul antrenat pentru a determina clasa fiecărei observații din X_test
        y_pred Este un vector care conține clasele prezise de LDA pentru fiecare rând din X_test
    """

    # 6. Evaluăm performanța modelului
    ## Evaluare model pe setul de testare (matricea de confuzie + indicatori de acuratețe) SAU
    ## Evaluare model bayesian (matricea de confuzie + indicatori de acuratețe)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    """
        confusion_matrix(y_test, y_pred) → Matricea de confuzie
        Arată câte predicții au fost corecte și câte au fost greșite, comparând y_pred cu y_test.
        Matrice CxC, unde fiecare rând reprezintă clasele reale, iar fiecare coloană reprezintă clasele prezise.

        accuracy_score(y_test, y_pred) → Returnează procentajul instanțelor clasificate corect (intre 0 si 1)

        classification_report → Raport detaliat de clasificare
        Include 3 metrici importante pentru fiecare clasă:
        - Precizie (Precision) → Cât de multe predicții pentru o clasă sunt corecte?
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1_Score = 2 * (precision * recall) / (precision + recall)
        - Recall (Sensibilitate) → Cât de multe instanțe reale dintr-o clasă au fost corect identificate?
        - F1-Score → Media armonică dintre Precision și Recall, mai utilă când avem clase dezechilibrate.
        F1-Score apr de 1 => model apr perfect
    """

    # 7. Facem predicții pe un set de aplicare
    ## Predicția în setul de aplicare model liniar/bayesian
    X_new = X_test[:5]  # Folosim primele 5 instanțe ca exemplu de aplicare
    y_pred_application = model.predict(X_new)

    """
    Aici folosim modelul antrenat (LDA sau QDA) pentru a face predicții pe un set de aplicare (nu avem unul nou asa ca luam X_test[:5])
    predict aplica modelul antrenat si returneaza etichetele 
    """

    return {
        "model": model,
        "scores": scores,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "X_new": X_new,
        "y_pred_application": y_pred_application
    }


# ===================== Vizualizări =====================

## Trasare plot instanțe în axe discriminante
def plot_lda_scatter(scores, y_train):
    """ Scatter plot pentru instanțe în spațiul discriminant """
    if scores is None:
        print("QDA nu generează un spațiu discriminant.")
        return

    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(y_train)

    for cls in unique_classes:
        plt.scatter(scores[y_train == cls, 0],
                    scores[y_train == cls, 1] if scores.shape[1] > 1 else np.zeros_like(scores[y_train == cls, 0]),
                    label=f"Class {cls}")

    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel("Discriminant Function 1")
    plt.ylabel("Discriminant Function 2" if scores.shape[1] > 1 else "Zero Line")
    plt.title("Plot Instanțe în Axe Discriminante")
    plt.legend()
    plt.show()


## Trasare plot distribuții în axele discriminante
def plot_lda_distribution(scores, y_train):
    """ Distribuția scorurilor pe axele discriminante """
    if scores is None:
        print("QDA nu generează scoruri discriminante.")
        return

    plt.figure(figsize=(8, 6))
    for cls in np.unique(y_train):
        sns.kdeplot(scores[y_train == cls, 0], label=f"Class {cls}", fill=True, alpha=0.5)

    plt.xlabel("Discriminant Function 1")
    plt.ylabel("Density")
    plt.title("Distribuția Scorurilor Discriminante")
    plt.legend()
    plt.show()


## Matricea de confuzie
def plot_confusion_matrix(conf_matrix, title="Matricea de Confuzie"):
    """ Corelogramă pentru matricea de confuzie """
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


# ===================== Aplicarea Analizei Discriminante pe Iris =====================

# 📌 Încărcăm setul de date Iris
iris = load_iris()
X, y = iris.data, iris.target

# 🚀 Aplicăm LDA
lda_results = perform_discriminant_analysis(X, y, model_type="LDA")
qda_results = perform_discriminant_analysis(X, y, model_type="QDA")

# 📊 Vizualizări pentru LDA
plot_lda_scatter(lda_results["scores"], y[:len(lda_results["scores"])])
plot_lda_distribution(lda_results["scores"], y[:len(lda_results["scores"])])
plot_confusion_matrix(lda_results["confusion_matrix"], title="Matricea de Confuzie - LDA")

# 📊 Vizualizări pentru QDA
plot_confusion_matrix(qda_results["confusion_matrix"], title="Matricea de Confuzie - QDA")

# 📄 Afișăm raportul de clasificare pentru LDA și QDA
print("Evaluare LDA:")
print(lda_results["classification_report"])
print("\nEvaluare QDA:")
print(qda_results["classification_report"])

# 📌 Afișăm predicțiile pe setul de aplicare
print("\nPredicția pe setul de aplicare - LDA:", lda_results["y_pred_application"])
print("Predicția pe setul de aplicare - QDA:", qda_results["y_pred_application"])
