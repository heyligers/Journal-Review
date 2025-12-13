import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import warnings
# UPDATE: Import der neuen Black-76 Simulationsfunktion
from financial_models import simulate_rolling_option, simulate_barrier_option, simulate_futures_option

# Suppress FutureWarnings from yfinance
warnings.filterwarnings('ignore', category=FutureWarning)

def flatten_yfinance_columns(df):
    """Cleans MultiIndex columns from yfinance DataFrames."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- LOADING LOGIC ---

@st.cache_data
def load_multiple_gold_derivatives(start_date, end_date, selected_derivatives):
    """
    Lädt Daten für reale Derivate (inkl. GLDI) und simuliert BSM/Barrier/Black-76 Optionen.
    """
    # Map internal names to Yahoo Tickers
    all_derivatives = {
        'Gold_Spot': 'GC=F',
        'Gold_Futures': 'GC=F',
        'Gold_ETF_GLD': 'GLD',
        'Gold_ETF_IAU': 'IAU',
        'Sprott_Physical_Trust': 'PHYS',
        'Gold_Futures_ETF_DGL': 'DGL',
        'Inverse_Gold_ETF_GLL': 'GLL',
        'Gold_Leveraged_ETF_2x': 'UGL',
        # REALES PRODUKT: Credit Suisse Gold Shares Covered Call ETN
        'Gold_Covered_Call_GLDI': 'GLDI' 
    }
    
    # Filter only selected real derivatives
    real_derivatives = {k: v for k, v in all_derivatives.items() if k in selected_derivatives}
    
    derivative_data = {}
    base_gold_data = None
    gold_spot_data = None
    risk_free_data = None
    
    # 1. Zinsdaten laden (^IRX = 13 Week Treasury Bill)
    try:
        irx = yf.download("^IRX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        risk_free_data = flatten_yfinance_columns(irx)
    except:
        st.warning("Could not fetch Risk Free Rate (^IRX). Using constant fallback.")

    # 2. Reale Daten laden
    for name, ticker in real_derivatives.items():
        try:
            # Wir laden auch 'Low' für Barrier Checks
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            data = flatten_yfinance_columns(data)
            
            if not data.empty and len(data) > 10 and 'Close' in data.columns:
                if data['Close'].notna().sum() > 0:
                    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    derivative_data[name] = data.dropna()
                    
                    if name == 'Gold_Spot':
                        gold_spot_data = data.copy()
                    if name == 'Gold_Futures':
                        base_gold_data = data.copy()
                else:
                    st.warning(f"{name}: Invalid data (Null/NaN values)")
            else:
                st.warning(f"{name}: No data found for {name}")
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
            
    # Fallback für Simulationen (brauchen Basisdaten GC=F)
    # UPDATE: 'Gold_Futures_Option' zur Liste hinzugefügt
    simulated_keys = ['Gold_Options', 'Gold_Barrier_Option', 'Gold_Futures_Option']
    needs_simulation = any(k in selected_derivatives for k in simulated_keys)
    
    if needs_simulation and (base_gold_data is None or base_gold_data.empty):
        try:
            base_gold_data = yf.download('GC=F', start=start_date, end=end_date, progress=False, auto_adjust=False)
            base_gold_data = flatten_yfinance_columns(base_gold_data)
            base_gold_data['Returns'] = np.log(base_gold_data['Close'] / base_gold_data['Close'].shift(1))
            base_gold_data = base_gold_data.dropna()
        except:
            st.error("Could not load base data (Gold Futures) required for simulations.")

    # 3. Simulationen durchführen (Wissenschaftliche Modelle)
    if base_gold_data is not None and not base_gold_data.empty:
        
        # A) Black-Scholes Vanilla Options (Spot Options)
        if 'Gold_Options' in selected_derivatives:
            try:
                derivative_data['Gold_Options'] = simulate_rolling_option(base_gold_data, T_days=30, strike_pct=1.0)
            except Exception as e:
                st.warning(f"Simulation Error (BSM Options): {e}")

        # B) Options on Futures (Black-76) - NEU
        if 'Gold_Futures_Option' in selected_derivatives:
            try:
                derivative_data['Gold_Futures_Option'] = simulate_futures_option(
                    base_gold_data, T_days=30, strike_pct=1.0
                )
            except Exception as e:
                st.warning(f"Simulation Error (Black-76 Options): {e}")
                
        # C) Barrier Options (Down-and-Out)
        if 'Gold_Barrier_Option' in selected_derivatives:
            try:
                derivative_data['Gold_Barrier_Option'] = simulate_barrier_option(
                    base_gold_data, barrier_pct=0.95, T_days=30
                )
            except Exception as e:
                st.warning(f"Simulation Error (Barrier Option): {e}")

    return derivative_data, gold_spot_data
