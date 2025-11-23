import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def compute_correlations(x, y, methods):
    """
    Führt statische Korrelationsanalysen durch.
    """
    results = {}
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        for m in methods:
            results[m] = {'error': 'Zu wenige Daten'}
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
    Berechnet die Zeitreihe der Korrelation zwischen Komplexität und Volatilität.
    Zeigt, ob die Hypothese in bestimmten Marktphasen stärker gilt.
    """
    # Wir brauchen pro Tag mindestens 3 Derivate für eine sinnvolle Korrelation
    daily_corrs = []
    
    # Gruppieren nach Datum
    # combined_df hat Spalten: Date, Derivative, Complexity, Volatility
    for date, group in combined_df.groupby('Date'):
        # Filterung ungültiger Werte
        valid_data = group.dropna(subset=['Complexity', 'Volatility'])
        
        if len(valid_data) >= 4: # Mindestens 4 Punkte für stabile Korrelation
            # Pearson Korrelation für diesen Tag berechnen
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
    """(Legacy Helper - wird hier nicht primär genutzt, bleibt für Kompatibilität)"""
    return pd.Series(dtype=float)