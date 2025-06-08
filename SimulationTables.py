import pandas as pd
import numpy as np
from typing import Dict, Any

def generate_summary_tables(summary_df: pd.DataFrame, pointwise_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns a dictionary of tables summarizing:
    - Coverage by lambda and threshold
    - CI Length by lambda and threshold
    - Support Recovery by lambda and threshold
    - Pointwise variance fidelity
    """
    coverage_table = summary_df.pivot_table(
        index=["lambda_val", "threshold_val"],
        columns="method",
        values="mean_coverage"
    )

    ci_length_table = summary_df.pivot_table(
        index=["lambda_val", "threshold_val"],
        columns="method",
        values="mean_ci_length"
    )

    support_recovery_table = summary_df.pivot_table(
        index=["lambda_val", "threshold_val"],
        columns="method",
        values=["jaccard", "support_size", "fdr", "fnr", "precision", "recall"]
    )

    variance_fidelity_table = pointwise_df.pivot_table(
        index=["beta_index", "lambda_val", "threshold_val"],
        columns="method",
        values=["mc_variance", "boot_variance_mean", "mse_boot_variance"]
    )

    return {
        "coverage": coverage_table,
        "ci_length": ci_length_table,
        "support_recovery": support_recovery_table,
        "variance_fidelity": variance_fidelity_table
    }


