"""
Data Processing Utilities
Author: [Your Name]
Date: 2024

Reusable functions for common data processing tasks in research workflows.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict
import warnings


def clean_column_names(df: pd.DataFrame, 
                       lowercase: bool = True,
                       replace_spaces: str = '_') -> pd.DataFrame:
    """
    Standardize column names for easier manipulation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    lowercase : bool
        Convert to lowercase (default: True)
    replace_spaces : str
        Character to replace spaces with (default: '_')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names
        
    Example:
    --------
    >>> df = pd.DataFrame({'First Name': [1, 2], 'Last Name': [3, 4]})
    >>> clean_df = clean_column_names(df)
    >>> print(clean_df.columns.tolist())
    ['first_name', 'last_name']
    """
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', replace_spaces)
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
    
    if lowercase:
        df.columns = df.columns.str.lower()
    
    return df


def identify_missing_patterns(df: pd.DataFrame, 
                              threshold: float = 0.0) -> pd.DataFrame:
    """
    Identify patterns of missing data in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Minimum percentage of missing values to report (0-100)
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing data patterns
        
    Example:
    --------
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    >>> missing_summary = identify_missing_patterns(df)
    >>> print(missing_summary)
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * missing_count / len(df)
    
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_count.values,
        'missing_percent': missing_percent.values,
        'dtype': df.dtypes.values
    })
    
    missing_df = missing_df[missing_df['missing_percent'] >= threshold]
    missing_df = missing_df.sort_values('missing_percent', ascending=False)
    missing_df = missing_df.reset_index(drop=True)
    
    return missing_df


def winsorize_data(data: Union[pd.Series, np.ndarray],
                   lower_percentile: float = 0.01,
                   upper_percentile: float = 0.99) -> Union[pd.Series, np.ndarray]:
    """
    Winsorize data to handle extreme outliers.
    
    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Input data
    lower_percentile : float
        Lower percentile to cap at (0-1)
    upper_percentile : float
        Upper percentile to cap at (0-1)
        
    Returns:
    --------
    Same type as input
        Winsorized data
        
    Example:
    --------
    >>> data = pd.Series([1, 2, 3, 100, 200])
    >>> winsorized = winsorize_data(data, lower_percentile=0.05, upper_percentile=0.95)
    """
    if isinstance(data, pd.Series):
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        lower_bound = np.percentile(data, lower_percentile * 100)
        upper_bound = np.percentile(data, upper_percentile * 100)
        return np.clip(data, lower_bound, upper_bound)


def create_lag_variables(df: pd.DataFrame,
                        columns: List[str],
                        lags: List[int],
                        group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Create lagged variables for time series or panel data analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str
        Columns to create lags for
    lags : list of int
        Number of periods to lag (can be negative for leads)
    group_by : str, optional
        Column to group by (for panel data)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original and lagged variables
        
    Example:
    --------
    >>> df = pd.DataFrame({'id': [1,1,1], 'time': [1,2,3], 'x': [10,20,30]})
    >>> df_lagged = create_lag_variables(df, ['x'], [1, 2], group_by='id')
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            lag_name = f'{col}_lag{lag}' if lag > 0 else f'{col}_lead{abs(lag)}'
            
            if group_by:
                df[lag_name] = df.groupby(group_by)[col].shift(lag)
            else:
                df[lag_name] = df[col].shift(lag)
    
    return df


def standardize_numeric_columns(df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                method: str = 'zscore') -> pd.DataFrame:
    """
    Standardize numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str, optional
        Columns to standardize (default: all numeric columns)
    method : str
        Standardization method: 'zscore' or 'minmax'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized columns
        
    Example:
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    >>> df_std = standardize_numeric_columns(df, method='zscore')
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame")
            continue
            
        if method == 'zscore':
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            raise ValueError("Method must be 'zscore' or 'minmax'")
    
    return df


def create_interaction_terms(df: pd.DataFrame,
                            var1: str,
                            var2: str,
                            name: Optional[str] = None) -> pd.DataFrame:
    """
    Create interaction terms between two variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    var1 : str
        First variable name
    var2 : str
        Second variable name
    name : str, optional
        Name for interaction term (default: var1_x_var2)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction term added
        
    Example:
    --------
    >>> df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
    >>> df_interact = create_interaction_terms(df, 'x1', 'x2')
    """
    df = df.copy()
    
    if name is None:
        name = f'{var1}_x_{var2}'
    
    df[name] = df[var1] * df[var2]
    
    return df


def create_dummy_variables(df: pd.DataFrame,
                          columns: List[str],
                          drop_first: bool = True,
                          prefix_sep: str = '_') -> pd.DataFrame:
    """
    Create dummy variables from categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str
        Categorical columns to convert
    drop_first : bool
        Whether to drop first category (avoid multicollinearity)
    prefix_sep : str
        Separator between column name and category
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dummy variables
        
    Example:
    --------
    >>> df = pd.DataFrame({'group': ['A', 'B', 'C']})
    >>> df_dummies = create_dummy_variables(df, ['group'])
    """
    df = df.copy()
    
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, 
                                prefix_sep=prefix_sep, 
                                drop_first=drop_first)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    
    return df


def aggregate_by_group(df: pd.DataFrame,
                      group_cols: List[str],
                      agg_dict: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
    """
    Aggregate data by groups with multiple aggregation functions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_cols : list of str
        Columns to group by
    agg_dict : dict
        Dictionary mapping column names to aggregation functions
        
    Returns:
    --------
    pd.DataFrame
        Aggregated dataframe
        
    Example:
    --------
    >>> df = pd.DataFrame({'group': ['A','A','B','B'], 
    ...                    'value': [1, 2, 3, 4]})
    >>> agg_df = aggregate_by_group(df, ['group'], {'value': ['mean', 'std']})
    """
    aggregated = df.groupby(group_cols).agg(agg_dict)
    aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) 
                         else col for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()
    
    return aggregated


def detect_outliers_iqr(data: Union[pd.Series, np.ndarray],
                       multiplier: float = 1.5) -> np.ndarray:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Input data
    multiplier : float
        IQR multiplier (default: 1.5 for standard outliers, 3.0 for extreme)
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating outliers
        
    Example:
    --------
    >>> data = pd.Series([1, 2, 3, 4, 100])
    >>> outliers = detect_outliers_iqr(data)
    >>> print(data[outliers])
    """
    if isinstance(data, pd.Series):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        data_array = data.values
    else:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        data_array = data
    
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (data_array < lower_bound) | (data_array > upper_bound)
    
    return outliers


# Example usage
if __name__ == "__main__":
    print("Data Processing Utilities - Example Usage")
    print("=" * 60)
    
    # Create sample data
    df = pd.DataFrame({
        'First Name': ['John', 'Jane', 'Bob'],
        'Last Name': ['Doe', 'Smith', 'Johnson'],
        'Age': [25, 30, None],
        'Income': [50000, 75000, 60000],
        'Category': ['A', 'B', 'A']
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Clean column names
    df_clean = clean_column_names(df)
    print("\nAfter cleaning column names:")
    print(df_clean.columns.tolist())
    
    # Check missing data
    print("\nMissing data summary:")
    print(identify_missing_patterns(df_clean))
    
    # Standardize numeric column
    df_std = standardize_numeric_columns(df_clean, ['income'], method='zscore')
    print("\nAfter standardizing income:")
    print(df_std[['income']])
    
    print("\n" + "=" * 60)
