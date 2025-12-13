import numpy as np
from scipy.optimize import curve_fit

# Power Model (MAIN HYPOTHESIS - Fermat-compliant)
def power_model(x, a, alpha):
    """Power Law: σ = a * C^α"""
    return a * np.power(x, alpha)

# Exponential Model (Comparison)
def exp_model(x, a, b):
    """Exponential Model: σ = a * e^(b*C)"""
    return a * np.exp(b * x)

def calculate_metrics(y_true, y_pred, model_params, formula):
    """
    Calculates R2, RMSE, AIC and BIC.
    """
    # Residuals and Sum of Squares
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2) # RSS
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    n = len(y_true)
    k = len(model_params) 
    
    # 1. R²
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    # 2. RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # 3. AIC & BIC (Only calculate if n > k)
    if n > k and ss_res > 0:
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
    Fits models and returns extended statistics.
    """
    results = {}
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Filter for valid data
    mask = (x > 0) & (y > 0) & (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3: 
        return {k: {'error': 'Insufficient data points'} for k in ['Power Law', 'Linear', 'Exponential']}

    x_scale = np.max(x_clean)
    y_scale = np.max(y_clean)
    x_norm = x_clean / x_scale
    y_norm = y_clean / y_scale

    # --- POWER LAW MODEL ---
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
        results['Power Law'] = res

    except Exception as e:
        results['Power Law'] = {'error': str(e)}

    # --- LINEAR MODEL ---
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

    # --- EXPONENTIAL MODEL ---
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
        results['Exponential'] = res
    except Exception as e:
        results['Exponential'] = {'error': str(e)}

    return results
