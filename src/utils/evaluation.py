import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import logging

def evaluate_model(model, X_test, y_test, model_name="Modelo"):
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        logging.info(f"\n🔎 Avaliação — {model_name}")
        logging.info(f"Acurácia: {acc:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        if roc_auc:
            logging.info(f"ROC AUC: {roc_auc:.4f}")

        # Plotar gráficos
        plot_confusion_matrix(y_test, y_pred, model_name)
        if roc_auc:
            plot_roc_curve(y_test, y_proba, model_name)

        return acc, f1, roc_auc
    except Exception as e:
        logging.error(f"Erro na avaliação do modelo: {str(e)}")
        raise

def plot_confusion_matrix(y_test, y_pred, model_name="Modelo"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.savefig(f"reports/{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_test, y_proba, model_name="Modelo"):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdadeiros Positivos")
    plt.title(f"Curva ROC - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/{model_name}_roc_curve.png")
    plt.close()
