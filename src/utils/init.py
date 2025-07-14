"""
Utility functions for evaluation, error analysis and reporting.
"""

from .evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve
)

from .errors import show_errors
from .report import generate_report
