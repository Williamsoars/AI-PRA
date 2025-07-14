import pandas as pd

def mostrar_erros(caminho_csv: str, y_test, y_pred, indices_test, texto_col="text", label_col="label", max_exemplos=10):
    """
    Mostra exemplos de tweets mal classificados.
    """
    df = pd.read_csv(caminho_csv)
    erros = []

    for i, idx in enumerate(indices_test):
        verdadeiro = y_test[i]
        previsto = y_pred[i]
        if verdadeiro != previsto:
            texto = df.loc[idx, texto_col]
            erros.append((texto, verdadeiro, previsto))
        if len(erros) >= max_exemplos:
            break

    print(f"\n🔍 {len(erros)} exemplos de erros de classificação:\n")
    for i, (texto, real, pred) in enumerate(erros, 1):
        print(f"[{i}] Real: {real} | Previsto: {pred}")
        print(f"     Texto: {texto}\n")
