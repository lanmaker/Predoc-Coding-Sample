"""
Regression Analysis Examples
Author: [Your Name]
Date: 2024

This script demonstrates various regression techniques commonly used in 
econometric research, including OLS, robust standard errors, and fixed effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns


def generate_sample_data(n=1000, n_groups=50):
    """
    Generate sample panel data for demonstration.
    
    Parameters:
    -----------
    n : int
        Total number of observations
    n_groups : int
        Number of groups/entities
        
    Returns:
    --------
    pd.DataFrame
        Sample panel data
    """
    np.random.seed(42)
    
    # Generate group IDs
    group_id = np.random.choice(range(n_groups), size=n)
    
    # Generate time periods
    time_period = np.random.choice(range(1, 21), size=n)
    
    # Generate group-specific fixed effects
    group_effects = np.random.normal(5, 2, n_groups)
    
    # Independent variables
    x1 = np.random.normal(10, 3, n)
    x2 = np.random.normal(20, 5, n)
    x3 = np.random.binomial(1, 0.5, n)  # Binary variable
    
    # Generate dependent variable with group effects
    epsilon = np.random.normal(0, 2, n)
    y = (3 + group_effects[group_id] + 
         2.5 * x1 + 1.5 * x2 + 3.0 * x3 + epsilon)
    
    data = pd.DataFrame({
        'group_id': group_id,
        'time': time_period,
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    
    return data


def run_ols_regression(data):
    """
    Run basic OLS regression.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for regression
        
    Returns:
    --------
    statsmodels RegressionResults
        OLS regression results
    """
    # Add constant term
    X = sm.add_constant(data[['x1', 'x2', 'x3']])
    y = data['y']
    
    # Fit OLS model
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results


def run_ols_with_robust_se(data):
    """
    Run OLS regression with robust (heteroskedasticity-consistent) standard errors.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for regression
        
    Returns:
    --------
    statsmodels RegressionResults
        OLS regression results with robust SE
    """
    X = sm.add_constant(data[['x1', 'x2', 'x3']])
    y = data['y']
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC3')  # HC3 robust standard errors
    
    return results


def run_fixed_effects_regression(data):
    """
    Run fixed effects regression (within estimator).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel data with group_id
        
    Returns:
    --------
    statsmodels RegressionResults
        Fixed effects regression results
    """
    # Demean the data within groups (fixed effects transformation)
    data_demeaned = data.copy()
    
    for var in ['y', 'x1', 'x2', 'x3']:
        group_means = data.groupby('group_id')[var].transform('mean')
        data_demeaned[f'{var}_dm'] = data[var] - group_means
    
    # Run OLS on demeaned data
    X = data_demeaned[['x1_dm', 'x2_dm', 'x3_dm']]
    y = data_demeaned['y_dm']
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results


def run_clustered_se_regression(data):
    """
    Run regression with cluster-robust standard errors.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for regression with clustering variable
        
    Returns:
    --------
    statsmodels RegressionResults
        Regression results with clustered SE
    """
    X = sm.add_constant(data[['x1', 'x2', 'x3']])
    y = data['y']
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='cluster', 
                       cov_kwds={'groups': data['group_id']})
    
    return results


def check_multicollinearity(data):
    """
    Check for multicollinearity using Variance Inflation Factors (VIF).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing independent variables
        
    Returns:
    --------
    pd.DataFrame
        VIF values for each variable
    """
    X = data[['x1', 'x2', 'x3']]
    X = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    return vif_data


def plot_residual_diagnostics(results, data):
    """
    Create diagnostic plots for regression residuals.
    
    Parameters:
    -----------
    results : statsmodels RegressionResults
        Regression results object
    data : pd.DataFrame
        Original data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fitted vs Residuals
    axes[0, 0].scatter(results.fittedvalues, results.resid, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    sm.qqplot(results.resid, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scale-Location plot
    standardized_resid = results.resid / np.std(results.resid)
    axes[1, 0].scatter(results.fittedvalues, np.sqrt(np.abs(standardized_resid)), 
                      alpha=0.5)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1, 1].hist(results.resid, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(results_dict):
    """
    Create a comparison table of multiple regression models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results as values
        
    Returns:
    --------
    statsmodels Summary
        Comparison table
    """
    comparison = summary_col(
        list(results_dict.values()),
        model_names=list(results_dict.keys()),
        stars=True,
        float_format='%.4f',
        info_dict={
            'N': lambda x: f"{int(x.nobs)}",
            'R-squared': lambda x: f"{x.rsquared:.4f}",
            'Adj. R-squared': lambda x: f"{x.rsquared_adj:.4f}"
        }
    )
    
    return comparison


def main():
    """
    Main function to run regression analysis examples.
    """
    print("=" * 70)
    print("Regression Analysis Examples")
    print("=" * 70)
    print()
    
    # Generate sample data
    print("Generating sample panel data...")
    data = generate_sample_data(n=1000, n_groups=50)
    print(f"Data shape: {data.shape}")
    print(f"Number of groups: {data['group_id'].nunique()}")
    print()
    
    # Check for multicollinearity
    print("Checking for multicollinearity (VIF):")
    print("-" * 70)
    vif = check_multicollinearity(data)
    print(vif)
    print("\nNote: VIF < 5 generally indicates no multicollinearity concerns")
    print()
    
    # Run different regression specifications
    print("Running regression models...")
    print()
    
    print("1. Basic OLS Regression:")
    print("-" * 70)
    ols_results = run_ols_regression(data)
    print(ols_results.summary())
    print()
    
    print("2. OLS with Robust Standard Errors (HC3):")
    print("-" * 70)
    robust_results = run_ols_with_robust_se(data)
    print(robust_results.summary())
    print()
    
    print("3. Fixed Effects Regression:")
    print("-" * 70)
    fe_results = run_fixed_effects_regression(data)
    print(fe_results.summary())
    print()
    
    print("4. Clustered Standard Errors (by group):")
    print("-" * 70)
    cluster_results = run_clustered_se_regression(data)
    print(cluster_results.summary())
    print()
    
    # Model comparison
    print("Model Comparison:")
    print("-" * 70)
    models = {
        'OLS': ols_results,
        'Robust SE': robust_results,
        'Clustered SE': cluster_results
    }
    comparison = compare_models(models)
    print(comparison)
    print()
    
    # Diagnostic plots
    print("Creating diagnostic plots...")
    fig = plot_residual_diagnostics(ols_results, data)
    # fig.savefig('regression_diagnostics.png', dpi=300, bbox_inches='tight')
    
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
