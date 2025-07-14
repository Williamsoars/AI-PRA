from pathlib import Path
import logging

def generate_report(results, output_path):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Relatório de Análise de Sentimentos em Tweets\n\n")
            f.write("## Tabela de Resultados\n\n")
            f.write("| Modelo | Acurácia | F1 Score | ROC AUC |\n")
            f.write("|--------|----------|----------|---------|\n")
            for name, acc, f1, auc in results:
                f.write(f"| {name} | {acc:.4f} | {f1:.4f} | {auc:.4f if auc else 'N/A'} |\n")

            f.write("\n## Gráficos\n")
            for name, _, _, _ in results:
                f.write(f"![Matriz de Confusão]({name}_confusion_matrix.png)\n")
                f.write(f"![Curva ROC]({name}_roc_curve.png)\n\n")

            f.write("\n## Observações\n")
            f.write("- Os resultados mostram o desempenho comparativo dos modelos.\n")
            f.write("- Considere ajustar hiperparâmetros para melhorar os resultados.\n")

        logging.info(f"Relatório salvo em: {output_path}")
    except Exception as e:
        logging.error(f"Erro ao gerar relatório: {str(e)}")
        raise
