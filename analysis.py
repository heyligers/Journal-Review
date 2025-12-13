import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def compute_correlations(x, y, methods):
    """
    Performs statistical correlation analyses.
    """
    results = {}
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        for m in methods:
            # TRANSLATED ERROR
            results[m] = {'error': 'Insufficient data'}
        return results
    
    for m in methods:
        try:
            if m == 'Pearson':
                stat, p = pearsonr(x_clean, y_clean)
            elif m == 'Spearman':
                stat, p = spearmanr(x_clean, y_clean)
            elif m == 'Kendall':
                stat, p = kendalltau(x_clean, y_clean)
            results[m] = {'stat': stat, 'p-value': p}
        except Exception as e:
            results[m] = {'error': str(e)}
    
    return results

def calculate_dynamic_correlation(combined_df):
    """
    Calculates the time series of the correlation between complexity and volatility.
    """
    # We need at least 3 derivatives per day for a meaningful correlation
    daily_corrs = []
    
    # Group by date
    for date, group in combined_df.groupby('Date'):
        valid_data = group.dropna(subset=['Complexity', 'Volatility'])
        
        if len(valid_data) >= 4: # At least 4 points for stable correlation
            try:
                corr = valid_data['Complexity'].corr(valid_data['Volatility'])
                if not np.isnan(corr):
                    daily_corrs.append({'Date': date, 'Correlation': corr})
            except:
                pass
                
    if not daily_corrs:
        return pd.DataFrame()
        
    return pd.DataFrame(daily_corrs).sort_values('Date').set_index('Date')

def rolling_correlation(df, window, method='pearson'):
    """(Legacy Helper - not primarily used here, kept for compatibility)"""
    return pd.Series(dtype=float)
