import numpy as np
import pandas as pd
import streamlit as st

def calc_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Berechnet die annualisierte Realized Volatility über ein Rolling Window.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_all_volatilities(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Berechnet Realized Volatility für einen Datensatz.
    """
    data = data.copy()
    data['Volatility'] = calc_realized_volatility(data['Returns'], window)
    return data
