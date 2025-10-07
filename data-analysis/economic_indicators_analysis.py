"""
Economic Indicators Analysis
Author: [Your Name]
Date: 2024

This script demonstrates data analysis skills using Python pandas.
It analyzes economic indicators including GDP, unemployment, and inflation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_clean_data(filepath):
    """
    Load economic data and perform initial cleaning.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing economic data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with economic indicators
    """
    # In a real scenario, this would load actual data
    # For demonstration, we create sample data
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='Q')
    
    data = pd.DataFrame({
        'date': dates,
        'gdp_growth': np.random.normal(2.5, 1.5, len(dates)),
        'unemployment_rate': np.random.normal(5.5, 1.2, len(dates)),
        'inflation_rate': np.random.normal(2.0, 0.8, len(dates)),
        'interest_rate': np.random.normal(2.5, 1.0, len(dates))
    })
    
    # Ensure realistic bounds
    data['unemployment_rate'] = data['unemployment_rate'].clip(lower=3.0, upper=10.0)
    data['inflation_rate'] = data['inflation_rate'].clip(lower=0.0, upper=5.0)
    data['interest_rate'] = data['interest_rate'].clip(lower=0.0, upper=6.0)
    
    return data


def calculate_summary_statistics(data):
    """
    Calculate summary statistics for economic indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing economic indicators
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    summary = data[numeric_cols].describe()
    
    # Add additional statistics
    summary.loc['median'] = data[numeric_cols].median()
    summary.loc['skewness'] = data[numeric_cols].skew()
    summary.loc['kurtosis'] = data[numeric_cols].kurtosis()
    
    return summary


def analyze_correlations(data):
    """
    Analyze correlations between economic indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing economic indicators
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    return correlation_matrix


def create_time_series_plot(data):
    """
    Create time series visualizations of economic indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing economic indicators with date column
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Economic Indicators Over Time', fontsize=16, fontweight='bold')
    
    indicators = ['gdp_growth', 'unemployment_rate', 'inflation_rate', 'interest_rate']
    titles = ['GDP Growth Rate (%)', 'Unemployment Rate (%)', 
              'Inflation Rate (%)', 'Interest Rate (%)']
    
    for ax, indicator, title in zip(axes.flat, indicators, titles):
        ax.plot(data['date'], data[indicator], linewidth=2, color='steelblue')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(correlation_matrix):
    """
    Create a heatmap of correlations between indicators.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    ax.set_title('Correlation Matrix of Economic Indicators', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def calculate_rolling_statistics(data, window=4):
    """
    Calculate rolling statistics for trend analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing economic indicators
    window : int
        Rolling window size (default: 4 quarters = 1 year)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling statistics
    """
    rolling_data = data.copy()
    
    indicators = ['gdp_growth', 'unemployment_rate', 'inflation_rate', 'interest_rate']
    
    for indicator in indicators:
        rolling_data[f'{indicator}_ma'] = data[indicator].rolling(window=window).mean()
        rolling_data[f'{indicator}_std'] = data[indicator].rolling(window=window).std()
    
    return rolling_data


def main():
    """
    Main execution function for the analysis.
    """
    print("=" * 60)
    print("Economic Indicators Analysis")
    print("=" * 60)
    print()
    
    # Load and clean data
    print("Loading and cleaning data...")
    data = load_and_clean_data('economic_data.csv')
    print(f"Data loaded: {len(data)} observations")
    print()
    
    # Calculate summary statistics
    print("Summary Statistics:")
    print("-" * 60)
    summary_stats = calculate_summary_statistics(data)
    print(summary_stats.round(3))
    print()
    
    # Analyze correlations
    print("Correlation Analysis:")
    print("-" * 60)
    correlations = analyze_correlations(data)
    print(correlations.round(3))
    print()
    
    # Calculate rolling statistics
    print("Calculating rolling statistics (1-year moving average)...")
    rolling_data = calculate_rolling_statistics(data, window=4)
    print("Rolling statistics calculated successfully.")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    ts_fig = create_time_series_plot(data)
    # ts_fig.savefig('economic_indicators_timeseries.png', dpi=300, bbox_inches='tight')
    
    corr_fig = create_correlation_heatmap(correlations)
    # corr_fig.savefig('economic_indicators_correlation.png', dpi=300, bbox_inches='tight')
    
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
