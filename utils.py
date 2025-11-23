import pandas as pd
import numpy as np

def merge_and_clean_data(gold_data: pd.DataFrame, complexity_data: pd.DataFrame) -> pd.DataFrame:
    """
    Führt Preisdaten und Komplexitätsdaten zusammen.
    """
    gold_df = gold_data.reset_index()

    # MultiIndex-Spalten auf einfache Spaltennamen reduzieren
    if isinstance(gold_df.columns, pd.MultiIndex):
        gold_df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in gold_df.columns]

    if isinstance(complexity_data.columns, pd.MultiIndex):
        complexity_data.columns = complexity_data.columns.get_level_values(-1)

    # Datumsspalte für gold_df sicherstellen
    if 'Date' not in gold_df.columns:
        # Beispielalternativen prüfen
        if 'date' in gold_df.columns:
            gold_df.rename(columns={'date': 'Date'}, inplace=True)
        elif 'index' in gold_df.columns:
            gold_df.rename(columns={'index': 'Date'}, inplace=True)
        else:
            # Versuch, den Index als Datum zu nutzen, falls er DatetimeIndex ist
            if isinstance(gold_df.index, pd.DatetimeIndex):
                gold_df['Date'] = gold_df.index
            else:
                pass # Wird später Fehler werfen, wenn 'Date' immer noch fehlt

    # Datumsspalte für complexity_data sicherstellen
    if 'Date' not in complexity_data.columns:
        if complexity_data.index.name == 'Date':
            complexity_data = complexity_data.reset_index()
        else:
            pass

    # Sicherstellen, dass beide 'Date' haben, sonst Abbruch vermeiden oder Fehler werfen
    if 'Date' in gold_df.columns and 'Date' in complexity_data.columns:
        df = pd.merge(gold_df, complexity_data, how='inner', on='Date')
        df = df.dropna(subset=['Volatility', 'Complexity'])
        return df
    else:
        # Fallback, falls Merge nicht möglich (sollte in normaler Pipeline nicht passieren)
        return pd.DataFrame()

def flatten_yfinance_columns(df):
    """
    Bereinigt MultiIndex-Spalten von yfinance DataFrames.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def safe_style_format(df, format_dict=None, default_format='{:.4f}'):
    """
    Wendet Styling sicher nur auf numerische Spalten an.
    """
    if format_dict is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        format_dict = {col: default_format for col in numeric_cols}
    else:
        # Filter keys that are actually in columns
        format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
    
    return df.style.format(format_dict)
