import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Berechnet den theoretischen Preis einer Call-Option nach Black-Scholes-Merton.
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def simulate_rolling_option(history_df, T_days=30, strike_pct=1.0, r_default=0.04):
    """
    Simuliert eine 'Rolling Constant Maturity Call Option' Strategie.
    """
    df = history_df.copy()
    
    # Volatility Calculation
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol_30d'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252)
    df['Hist_Vol_30d'] = df['Hist_Vol_30d'].fillna(0.15)
    
    T_years = T_days / 365.0
    
    if 'RiskFreeRate' not in df.columns:
        df['RiskFreeRate'] = r_default

    S = df['Close'].values
    r = df['RiskFreeRate'].values
    sigma = df['Hist_Vol_30d'].values
    
    # Rolling ATM Strike
    K = S * strike_pct 
    
    option_prices = []
    
    for i in range(len(S)):
        vol_input = max(sigma[i], 0.01)
        price = black_scholes_call_price(S[i], K[i], T_years, r[i], vol_input)
        option_prices.append(price)
        
    df['Option_Price_Theoretical'] = option_prices
    
    # Returns calculation
    df['Returns'] = df['Option_Price_Theoretical'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], 0)
    
    # Reconstruct Price Index
    start_val = df['Close'].iloc[0]
    df['Simulated_Close'] = start_val * (1 + df['Returns']).cumprod()
    
    return pd.DataFrame({'Close': df['Simulated_Close'], 'Returns': df['Returns']}, index=df.index).dropna()

def simulate_barrier_option(history_df, barrier_pct=0.95, T_days=30, r_default=0.04):
    """
    Simuliert eine 'Rolling Down-and-Out Call Option'.
    
    Logik:
    Wir halten eine Option. Wenn der Preis (Low des Tages) die Barriere berührt,
    ist die Option wertlos (Knock-Out). Dies erzeugt extreme 'Gap-Risiken'.
    """
    df = history_df.copy()
    
    # Volatility Calculation
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol_30d'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252)
    df['Hist_Vol_30d'] = df['Hist_Vol_30d'].fillna(0.15)
    
    if 'RiskFreeRate' not in df.columns: df['RiskFreeRate'] = r_default
    
    # WICHTIG: Wir brauchen das Tagestief (Low), um Knock-Outs zu prüfen.
    # Wenn 'Low' nicht da ist, nutzen wir 'Close' (weniger präzise).
    if 'Low' not in df.columns:
        df['Low'] = df['Close']

    S = df['Close'].values
    Low = df['Low'].values 
    r = df['RiskFreeRate'].values
    sigma = df['Hist_Vol_30d'].values
    
    T_years = T_days / 365.0
    
    prices = []
    
    for i in range(len(S)):
        # Barriere liegt bei X% des aktuellen Preises (wir simulieren, dass wir
        # jeden Tag eine "frische" Barrier Option bewerten, um das Risikoprofil zu zeigen)
        # In einer echten Haltedauer wäre die Barriere fix. Hier 'rollieren' wir den Barrier-Level mit.
        
        # Um das Knock-Out Risiko zu simulieren:
        # Wir betrachten die Option von GESTERN (i-1).
        # Hat das HEUTIGE Low die Barriere von gestern verletzt?
        
        vol = max(sigma[i], 0.01)
        
        # 1. Basis: Vanilla Preis
        vanilla_price = black_scholes_call_price(S[i], S[i], T_years, r[i], vol)
        
        # 2. Barrier Check
        if i > 0:
            barrier_level = S[i-1] * barrier_pct
            if Low[i] <= barrier_level:
                # KNOCK OUT EVENT
                # Der Preis der Option springt auf fast 0
                prices.append(0.001)
                continue

        # 3. Wenn kein Knock-Out: Preis ist ähnlich Vanilla, aber leicht diskontiert
        # (Vereinfachung für Simulation des Risikoprofils)
        prices.append(vanilla_price)

    df['Barrier_Price'] = prices
    
    # Returns berechnen
    # Bei einem Knock-Out (Preis springt von X auf 0.001) entsteht ein Return von fast -100%
    df['Returns'] = df['Barrier_Price'].pct_change().fillna(0)
    
    # Filterung extremer Artefakte (außer Knock-Outs)
    df['Returns'] = df['Returns'].clip(-1.0, 5.0) 
    
    start_val = df['Close'].iloc[0]
    df['Simulated_Close'] = start_val * (1 + df['Returns']).cumprod()
    
    return pd.DataFrame({'Close': df['Simulated_Close'], 'Returns': df['Returns']}, index=df.index).dropna()

# --- NEU: BLACK-76 MODEL FÜR FUTURES OPTIONEN ---
def black_76_call_price(F, K, T, r, sigma):
    """
    Berechnet den Preis einer Option auf Futures nach dem Black-76 Modell.
    Unterschied zu Black-Scholes: Der Drift ist 0, da Futures bereits Forward-Preise sind.
    
    Formula: C = e^(-rT) * [F * N(d1) - K * N(d2)]
    """
    if F <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0

    # Hinweis: Im d1 fehlt das 'r', da der Future-Preis driftlos ist (martingale under risk-neutral measure)
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Der Diskontfaktor e^(-rT) wird auf den GANZEN Term angewendet
    disc = np.exp(-r * T)
    call_price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    return call_price

def simulate_futures_option(history_df, T_days=30, strike_pct=1.0, r_default=0.04):
    """
    Simuliert eine Option auf Gold-Futures (OG).
    Underlying: GC=F (Gold Futures)
    Modell: Black-76
    """
    df = history_df.copy()
    
    # Volatility Calculation (basierend auf Futures Returns)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Vol_30d'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252)
    df['Hist_Vol_30d'] = df['Hist_Vol_30d'].fillna(0.15)
    
    T_years = T_days / 365.0
    
    if 'RiskFreeRate' not in df.columns:
        df['RiskFreeRate'] = r_default

    F = df['Close'].values # Hier ist Close der Futures Preis
    r = df['RiskFreeRate'].values
    sigma = df['Hist_Vol_30d'].values
    
    # Rolling ATM Strike
    K = F * strike_pct 
    
    option_prices = []
    
    for i in range(len(F)):
        vol_input = max(sigma[i], 0.01)
        # WICHTIG: Aufruf von Black-76 statt Black-Scholes
        price = black_76_call_price(F[i], K[i], T_years, r[i], vol_input)
        option_prices.append(price)
        
    df['Option_Price_Theoretical'] = option_prices
    
    # Returns calculation
    df['Returns'] = df['Option_Price_Theoretical'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], 0)
    
    # Reconstruct Price Index
    start_val = df['Close'].iloc[0]
    df['Simulated_Close'] = start_val * (1 + df['Returns']).cumprod()
    
    return pd.DataFrame({'Close': df['Simulated_Close'], 'Returns': df['Returns']}, index=df.index).dropna()
