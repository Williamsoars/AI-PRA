import os

def gerar_relatorio_markdown(resultados, caminho_out="docs/relatorio.md"):
    """
    Gera um relatório markdown com as métricas dos modelos.
    `resultados`: lista de tuplas (nome_modelo, acc, f1, roc_auc)
    """
    os.makedirs(os.path.dirname(caminho_out), exist_ok=True)

    with open(caminho_out, "w", encoding="utf-8") as f:
        f.write("# Relatório de Análise de Sentimentos em Tweets\n\n")
        f.write("## Tabela de Resultados\n\n")
        f.write("| Modelo | Acurácia | F1 Score | ROC AUC |\n")
        f.write("|--------|----------|----------|---------|\n")
        for nome, acc, f1, auc in resultados:
            f.write(f"| {nome} | {acc:.4f} | {f1:.4f} | {auc:.4f} |\n")

        f.write("\n## Observações\n")
        f.write("- Os resultados mostram que o modelo XYZ tem melhor desempenho.\n")
        f.write("- Pode-se investigar erros relacionados a ironia ou ambiguidade.\n")
        f.write("- Sugestões: testar BERTimbau, mais dados, análise emocional, etc.\n")

    print(f"[✓] Relatório salvo em: {caminho_out}")
