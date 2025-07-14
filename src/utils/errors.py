import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple

def show_errors(
    input_path: Path,
    y_test: List,
    y_pred: List,
    test_indices: List[int],
    text_column: str = "text",
    label_column: str = "label",
    max_examples: int = 10
) -> List[Tuple[str, int, int]]:
    """
    Display examples of misclassified texts with their true and predicted labels.
    
    Returns:
        List of tuples (text, true_label, predicted_label)
    """
    try:
        df = pd.read_csv(input_path)
        errors = []
        
        for i, idx in enumerate(test_indices):
            true_label = y_test[i]
            predicted_label = y_pred[i]
            
            if true_label != predicted_label:
                text = df.loc[idx, text_column]
                errors.append((text, true_label, predicted_label))
                
            if len(errors) >= max_examples:
                break
        
        if errors:
            logging.info(f"\n🔍 Found {len(errors)} misclassification examples:\n")
            for i, (text, true, pred) in enumerate(errors, 1):
                logging.info(f"[{i}] True: {true} | Predicted: {pred}")
                logging.info(f"     Text: {text[:200]}...\n")
        
        return errors
        
    except Exception as e:
        logging.error(f"Error analyzing classification errors: {str(e)}")
        raise
