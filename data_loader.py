import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import warnings

# Suppress FutureWarnings from yfinance
warnings.filterwarnings('ignore', category=FutureWarning)

def flatten_yfinance_columns(df):
    """Cleans MultiIndex columns from yfinance DataFrames."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- SIMULATION LOGIC ---

def simulate_gold_options_data(base_data, volatility_multiplier=1.5):
    """Simulates Gold Options data (Call Option, ATM)."""
    data = base_data.copy()
    if 'Returns' in data.columns:
        data['Returns'] = data['Returns'] * volatility_multiplier
    
    if 'Close' in data.columns:
        option_percentage = 0.08
        noise = np.random.normal(0, 0.1, len(data))
        data['Close'] = data['Close'] * option_percentage * (1 + noise)
    return data

def simulate_vanilla_swap_data(base_data):
    """
    Simulates a Standard Swap (Plain Vanilla).
    Characteristics: High leverage, linear response.
    """
    data = base_data.copy()
    leverage = 2.5
    noise_level = 0.005
    
    if 'Returns' in data.columns:
        noise = np.random.normal(0, noise_level, len(data))
        data['Returns'] = data['Returns'] * leverage + noise
        data['Close'] = base_data['Close'] * (1 + data['Returns'].cumsum())
        
    return data

def simulate_customized_swap_data(base_data):
    """
    Simulates an Exotic Swap (Customized/Structured).
    Characteristics: Non-linear, Tail-Risks ("Jumps").
    """
    data = base_data.copy()
    leverage = 3.0
    
    if 'Returns' in data.columns:
        basic_returns = data['Returns'] * leverage
        
        # Jump Diffusion (1% probability of event)
        jump_prob = 0.01 
        jump_size = np.random.normal(-0.05, 0.10, len(data)) 
        jumps = np.random.choice([0, 1], size=len(data), p=[1-jump_prob, jump_prob])
        
        data['Returns'] = basic_returns + (jumps * jump_size)
        data['Close'] = base_data['Close'] * (1 + data['Returns'].cumsum())
        
    return data

# --- LOADING LOGIC ---

@st.cache_data
def load_multiple_gold_derivatives(start_date, end_date, selected_derivatives):
    """
    Loads data for selected derivatives and runs simulations.
    UPDATED: Removed individual st.success messages for a cleaner UI.
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
    }
    
    real_derivatives = {k: v for k, v in all_derivatives.items() if k in selected_derivatives}
    
    derivative_data = {}
    base_gold_data = None
    gold_spot_data = None
    
    # 1. Load Real Data
    for name, ticker in real_derivatives.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            data = flatten_yfinance_columns(data)
            
            if not data.empty and len(data) > 10 and 'Close' in data.columns:
                if data['Close'].notna().sum() > 0 and (data['Close'] > 0).sum() > 0:
                    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                    derivative_data[name] = data.dropna()
                    
                    if name == 'Gold_Spot':
                        gold_spot_data = data.copy()
                    if name == 'Gold_Futures':
                        base_gold_data = data.copy()
                    
                    # CLEANUP: No st.success here anymore to avoid clutter
                else:
                    st.warning(f"⚠️ {name}: Invalid data (Null/NaN values)")
            else:
                st.warning(f"⚠️ {name}: No data found for timeframe")
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            
    # Fallback for simulations if Futures were not selected but needed
    simulated_keys = ['Gold_Options', 'Gold_Swap_Vanilla', 'Gold_Swap_Customized']
    needs_simulation = any(k in selected_derivatives for k in simulated_keys)
    
    if needs_simulation and (base_gold_data is None or base_gold_data.empty):
        try:
            base_gold_data = yf.download('GC=F', start=start_date, end=end_date, progress=False, auto_adjust=False)
            base_gold_data = flatten_yfinance_columns(base_gold_data)
            base_gold_data['Returns'] = np.log(base_gold_data['Close'] / base_gold_data['Close'].shift(1))
            base_gold_data = base_gold_data.dropna()
        except:
            st.error("❌ Could not load base data (Gold Futures) required for simulations.")

    # 2. Run Simulations
    if base_gold_data is not None and not base_gold_data.empty:
        if 'Gold_Options' in selected_derivatives:
            try:
                derivative_data['Gold_Options'] = simulate_gold_options_data(base_gold_data)
            except Exception as e:
                st.warning(f"⚠️ Simulation Error (Options): {e}")
                
        if 'Gold_Swap_Vanilla' in selected_derivatives:
            try:
                derivative_data['Gold_Swap_Vanilla'] = simulate_vanilla_swap_data(base_gold_data)
            except Exception as e:
                st.warning(f"⚠️ Simulation Error (Vanilla Swap): {e}")
                
        if 'Gold_Swap_Customized' in selected_derivatives:
            try:
                derivative_data['Gold_Swap_Customized'] = simulate_customized_swap_data(base_gold_data)
            except Exception as e:
                st.warning(f"⚠️ Simulation Error (Custom Swap): {e}")

    return derivative_data, gold_spot_data