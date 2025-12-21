"""Label encoding helpers for categorical features."""

from __future__ import annotations

from typing import Optional, Iterable

import pandas as pd


def add_label_encoded_columns(
    df: pd.DataFrame,
    columns: list[str],
    suffix: str = "_encoded",
    known_categories: Optional[dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Add label-encoded columns for specified categorical columns.

    Encoding is deterministic: unique values are sorted alphabetically
    and assigned consecutive integers starting from 0. This ensures
    identical encoding when regenerated from the same data.

    Args:
        df: Input DataFrame containing the columns to encode.
        columns: List of column names to encode.
        suffix: Suffix to append to encoded column names (default: "_encoded").
        known_categories: Optional dict mapping column names to list of all possible values.
                          If provided, encoding will be based on these values (sorted)
                          rather than just the values present in df.

    Returns:
        The same DataFrame with new encoded columns added (e.g.,
        'departure_from_encoded', 'departure_to_encoded').

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If any specified column is missing from df.

    Example:
        >>> df = pd.DataFrame({
        ...     "departure_from": ["Budapest", "London", "Budapest"],
        ...     "departure_to": ["Vienna", "Prague", "Vienna"],
        ... })
        >>> df = add_label_encoded_columns(df, ["departure_from", "departure_to"])
        >>> df["departure_from_encoded"].tolist()
        [0, 1, 0]  # Budapest=0, London=1 (alphabetical)
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if df.empty:
        for col in columns:
            df[f"{col}{suffix}"] = pd.Series(dtype="Int64")
        return df

    for col in columns:
        # Get unique values and sort alphabetically for deterministic encoding
        if known_categories and col in known_categories:
            unique_values = known_categories[col]
        else:
            unique_values = df[col].dropna().unique()

        sorted_values = sorted(unique_values, key=str)

        # Create mapping: value -> integer label
        value_to_label = {val: idx for idx, val in enumerate(sorted_values)}

        # Apply mapping to create encoded column
        encoded_col = f"{col}{suffix}"
        df[encoded_col] = df[col].map(value_to_label).astype("Int64")

    return df

