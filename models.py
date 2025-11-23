import numpy as np
from scipy.optimize import curve_fit

# Potenzmodell (HAUPTHYPOTHESE - Fermat-konform)
def power_model(x, a, alpha):
    """Power Law: σ = a * C^α"""
    return a * np.power(x, alpha)

# Exponentielles Modell (Vergleich)
def exp_model(x, a, b):
    """Exponentielles Modell: σ = a * e^(b*C)"""
    return a * np.exp(b * x)

def calculate_metrics(y_true, y_pred, model_params, formula):
    """
    Berechnet R2, RMSE, AIC und BIC.
    
    AIC/BIC Formel für Least Squares (mit n = Anzahl Beobachtungen, k = Anzahl Parameter):
    AIC = n * ln(RSS/n) + 2*k
    BIC = n * ln(RSS/n) + k * ln(n)
    """
    # Residuen und Sum of Squares
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2) # RSS
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    n = len(y_true)
    k = len(model_params) # Anzahl der Parameter (hier meist 2: a, b/alpha)
    
    # 1. R²
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    # 2. RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # 3. AIC & BIC (Nur berechnen wenn n > k, sonst Inf)
    if n > k and ss_res > 0:
        # Log-Likelihood Approximation für normalverteilte Fehler
        log_likelihood_term = n * np.log(ss_res / n)
        aic = log_likelihood_term + 2 * k
        bic = log_likelihood_term + k * np.log(n)
    else:
        aic = np.inf
        bic = np.inf
    
    return {
        'params': model_params,
        'R2': r2,
        'RMSE': rmse,
        'AIC': aic,
        'BIC': bic,
        'formula': formula
    }

def fit_models(x, y):
    """
    Passt Modelle an und gibt erweiterte Statistiken zurück.
    """
    results = {}
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Filter für gültige Daten
    mask = (x > 0) & (y > 0) & (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3: # Mindestens 3 Punkte für 2-Parameter Modell + Freiheitsgrade
        return {k: {'error': 'Zu wenige Datenpunkte'} for k in ['Potenzmodell', 'Linear', 'Exponentiell']}

    x_scale = np.max(x_clean)
    y_scale = np.max(y_clean)
    x_norm = x_clean / x_scale
    y_norm = y_clean / y_scale

    # --- POTENZMODELL ---
    try:
        log_x = np.log(x_norm)
        log_y = np.log(y_norm)
        coeffs_log = np.polyfit(log_x, log_y, 1)
        alpha_init = coeffs_log[0]
        a_norm_init = np.exp(coeffs_log[1])
        
        popt_norm, _ = curve_fit(
            power_model, x_norm, y_norm, 
            p0=[a_norm_init, alpha_init],
            bounds=([0, -np.inf], [np.inf, np.inf]),
            maxfev=10000
        )
        
        alpha_final = popt_norm[1]
        a_final = popt_norm[0] * y_scale / (x_scale ** alpha_final)
        
        y_pred_clean = power_model(x_clean, a_final, alpha_final)
        res = calculate_metrics(
            y_clean, y_pred_clean, [a_final, alpha_final],
            f'σ = {a_final:.4f} * C^{alpha_final:.4f}'
        )
        
        y_pred_full = power_model(x, a_final, alpha_final)
        res['y_pred'] = y_pred_full
        res['alpha'] = alpha_final
        results['Potenzmodell'] = res

    except Exception as e:
        results['Potenzmodell'] = {'error': str(e)}

    # --- LINEARES MODELL ---
    try:
        coeffs = np.polyfit(x_clean, y_clean, 1)
        y_pred_clean = np.polyval(coeffs, x_clean)
        
        res = calculate_metrics(
            y_clean, y_pred_clean, coeffs,
            f'σ = {coeffs[0]:.4f} * C + {coeffs[1]:.4f}'
        )
        
        y_pred_full = np.polyval(coeffs, x)
        res['y_pred'] = y_pred_full
        results['Linear'] = res
    except Exception as e:
        results['Linear'] = {'error': str(e)}

    # --- EXPONENTIELLES MODELL ---
    try:
        coeffs_exp = np.polyfit(x_norm, np.log(y_norm), 1)
        b_norm_init = coeffs_exp[0]
        a_norm_init = np.exp(coeffs_exp[1])
        
        popt_norm, _ = curve_fit(
            exp_model, x_norm, y_norm,
            p0=[a_norm_init, b_norm_init],
            bounds=([0, -np.inf], [np.inf, np.inf]),
            maxfev=10000
        )
        
        a_final = popt_norm[0] * y_scale
        b_final = popt_norm[1] / x_scale
        
        y_pred_clean = exp_model(x_clean, a_final, b_final)
        res = calculate_metrics(
            y_clean, y_pred_clean, [a_final, b_final],
            f'σ = {a_final:.4f} * e^({b_final:.4f}*C)'
        )
        
        y_pred_full = exp_model(x, a_final, b_final)
        res['y_pred'] = y_pred_full
        results['Exponentiell'] = res
    except Exception as e:
        results['Exponentiell'] = {'error': str(e)}

    return results