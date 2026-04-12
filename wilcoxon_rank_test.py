from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

# =========================
# User configuration
# =========================
FILEPATH_1 = r"C:\UPB\master-thesis\05_April_Statistical results\Gpt-oss\Results_Hybrid_frames_local_q1-300.xlsx"
FILEPATH_2 = r"C:\UPB\master-thesis\05_April_Statistical results\Gpt-oss\Results_Hybrid_frames_local_q1-300_with_Rerank.xlsx"
COLUMN_NAME = "retriever_f1"
ALPHA = 0.05
ALTERNATIVE = "less"  # options: "two-sided", "greater", "less"


def load_column_from_excel(file_path: str, column_name: str) -> pd.Series:
    """Load one column from an Excel file and coerce it to numeric."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return pd.to_numeric(df[column_name], errors="coerce")


def interpret_result(
    p_value: float,
    alpha: float,
    median_diff: float,
    alternative: str,
) -> str:
    """Return a human-readable interpretation of the Wilcoxon test result."""
    if p_value < alpha:
        if alternative == "two-sided":
            if median_diff > 0:
                direction = "first file tends to have larger values"
            elif median_diff < 0:
                direction = "second file tends to have larger values"
            else:
                direction = "the median paired difference is zero"

            return (
                f"p = {p_value:.6g} < alpha = {alpha}. Reject H0. "
                f"There is a statistically significant paired difference; "
                f"{direction}."
            )

        if alternative == "greater":
            return (
                f"p = {p_value:.6g} < alpha = {alpha}. Reject H0. "
                "Evidence supports that values in the first file are greater than "
                "in the second file."
            )

        # alternative == "less"
        return (
            f"p = {p_value:.6g} < alpha = {alpha}. Reject H0. "
            "Evidence supports that values in the first file are less than "
            "in the second file."
        )

    return (
        f"p = {p_value:.6g} >= alpha = {alpha}. Fail to reject H0. "
        "No statistically significant paired difference was detected."
    )


def run_wilcoxon_on_differences(diffs: pd.Series, alternative: str):
    """
    Run Wilcoxon signed-rank test on paired differences.

    Uses a compatibility fallback because newer SciPy versions use `method=`,
    while older versions use `mode=`.
    """
    try:
        return wilcoxon(
            diffs,
            zero_method="wilcox",
            alternative=alternative,
            method="auto",
        )
    except TypeError:
        # Fallback for older SciPy versions
        return wilcoxon(
            diffs,
            zero_method="wilcox",
            alternative=alternative,
            mode="auto",
        )


def main() -> None:
    x = load_column_from_excel(FILEPATH_1, COLUMN_NAME)
    y = load_column_from_excel(FILEPATH_2, COLUMN_NAME)

    # Keep only rows where both paired values are valid numbers
    paired = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()

    if paired.empty:
        raise ValueError(
            "No valid paired numeric rows remain after dropping "
            "missing/non-numeric values."
        )

    if len(paired) < 2:
        raise ValueError(
            "At least 2 paired observations are required for "
            "Wilcoxon signed-rank test."
        )

    diffs = paired["x"] - paired["y"]

    # zero_method='wilcox' removes zero differences from the ranking step.
    nonzero_diffs = diffs[diffs != 0]

    if nonzero_diffs.empty:
        raise ValueError(
            "All paired differences are zero. "
            "The Wilcoxon signed-rank test is not informative in this case."
        )

    stat, p_value = run_wilcoxon_on_differences(diffs, ALTERNATIVE)

    print("Wilcoxon Signed-Rank Test")
    print("-" * 30)
    print(f"File 1: {FILEPATH_1}")
    print(f"File 2: {FILEPATH_2}")
    print(f"Column: {COLUMN_NAME}")
    print(f"Paired sample size after dropping NaNs: {len(paired)}")
    print(f"Non-zero paired differences used by test: {len(nonzero_diffs)}")
    print(f"Alternative hypothesis: {ALTERNATIVE}")
    print(f"Test statistic (W): {stat:.6g}")
    print(f"p-value: {p_value:.6g}")
    print(f"Median paired difference (file1 - file2): {diffs.median():.6g}")
    print()

    print("Interpretation:")
    print(interpret_result(p_value, ALPHA, diffs.median(), ALTERNATIVE))


if __name__ == "__main__":
    main()