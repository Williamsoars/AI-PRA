import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import seaborn as sns
import numpy as np

def avaliar_classificador(y_test, y_pred, y_proba=None, nome_modelo="Modelo"):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n🔎 Avaliação — {nome_modelo}")
    print(f"Acurácia:     {acc:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    if roc_auc:
        print(f"ROC AUC:      {roc_auc:.4f}")

    return acc, f1, roc_auc

def plotar_matriz_confusao(y_test, y_pred, nome_modelo="Modelo"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.show()

def plotar_roc(y_test, y_proba, nome_modelo="Modelo"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{nome_modelo} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdadeiros Positivos")
    plt.title(f"Curva ROC - {nome_modelo}")
    plt.legend()
    plt.grid(True)
    plt.show()
