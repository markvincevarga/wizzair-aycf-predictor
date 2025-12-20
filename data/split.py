import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    target_col: str = "occurs",
    test_size: float = 0.2,
    date_col: str = "availability_start",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train-test split for realistic forecasting evaluation.

    Sorts by date_col and splits chronologically, with earlier data for training
    and later data for testing.

    Args:
        df: Input dataframe with features and target.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing (default 0.2).
        date_col: Column to sort by for time-based splitting.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    X = df_sorted.drop(columns=[target_col])
    y = df_sorted[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
