import pandas as pd
import numpy as np

class DerivativeComplexityClassifier:
    """
    Klassifiziert Gold-Derivate basierend auf quantitativen Komplexitätsmetriken.
    Update: Ermöglicht dynamische Gewichtungsanpassung ("What-If").
    """
    
    def __init__(self):
        self.complexity_scores = {
            # --- 1. PHYSISCHE / SPOT PRODUKTE ---
            'Gold_Spot': {
                'structure_layers': 0, 'leverage_factor': 1, 'pricing_complexity': 1, 
                'liquidity_score': 5, 'counterparty_risk': 0
            },
            'Gold_ETF_GLD': {
                'structure_layers': 2, 'leverage_factor': 1, 'pricing_complexity': 2, 
                'liquidity_score': 5, 'counterparty_risk': 2
            },
            'Gold_ETF_IAU': {
                'structure_layers': 2, 'leverage_factor': 1, 'pricing_complexity': 2, 
                'liquidity_score': 4, 'counterparty_risk': 2
            },
            'Sprott_Physical_Trust': {
                'structure_layers': 2, 'leverage_factor': 1, 'pricing_complexity': 2,
                'liquidity_score': 4, 'counterparty_risk': 1,
            },

            # --- 2. FUTURES BASIERTE PRODUKTE ---
            'Gold_Futures': {
                'structure_layers': 1, 'leverage_factor': 10, 'pricing_complexity': 1, 
                'liquidity_score': 5, 'counterparty_risk': 1
            },
            'Gold_Futures_ETF_DGL': {
                'structure_layers': 3, 'leverage_factor': 1, 'pricing_complexity': 3,
                'liquidity_score': 3, 'counterparty_risk': 3,
            },

            # --- 3. GEHEBELTE & INVERSE PRODUKTE ---
            'Gold_Leveraged_ETF_2x': {
                'structure_layers': 3, 'leverage_factor': 2, 'pricing_complexity': 3,
                'liquidity_score': 4, 'counterparty_risk': 2,
            },
            'Inverse_Gold_ETF_GLL': {
                'structure_layers': 4, 'leverage_factor': 2, 'pricing_complexity': 5,
                'liquidity_score': 3, 'counterparty_risk': 3,
            },

            # --- 4. OPTIONEN & SWAPS (SIMULIERT) ---
            'Gold_Options': {
                'structure_layers': 2, 'leverage_factor': 15, 'pricing_complexity': 4,
                'liquidity_score': 4, 'counterparty_risk': 1,
            },
            'Gold_Swap_Vanilla': {
                'structure_layers': 3, 'leverage_factor': 20, 'pricing_complexity': 3,
                'liquidity_score': 3, 'counterparty_risk': 4,
            },
            'Gold_Swap_Customized': {
                'structure_layers': 5, 'leverage_factor': 25, 'pricing_complexity': 5,
                'liquidity_score': 1, 'counterparty_risk': 5,
            },
        }
        
        # Standard-Gewichtungen
        self.weights = {
            'structure_layers': 0.25,
            'leverage_factor': 0.20,
            'pricing_complexity': 0.25,
            'liquidity_score': -0.15,
            'counterparty_risk': 0.15,
        }
    
    def update_weights(self, new_weights):
        """
        Aktualisiert die Gewichtungen basierend auf Nutzereingaben (What-If Szenario).
        Args:
            new_weights (dict): Dictionary mit neuen Gewichtungswerten.
        """
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value

    def calculate_complexity_score(self, derivative_name):
        """Berechnet einen gewichteten Komplexitätsscore."""
        if derivative_name not in self.complexity_scores:
            return None
        
        metrics = self.complexity_scores[derivative_name]
        score = 0
        for metric, value in metrics.items():
            score += value * self.weights[metric]
        
        return np.clip(score, 0, 10)
    
    def get_all_complexity_scores(self):
        """Gibt alle Komplexitätsscores als DataFrame zurück."""
        scores = {}
        for derivative in self.complexity_scores.keys():
            scores[derivative] = {
                'Complexity_Score': self.calculate_complexity_score(derivative),
                **self.complexity_scores[derivative]
            }
        
        df = pd.DataFrame(scores).T
        df = df.sort_values('Complexity_Score')
        return df
    
    def assign_complexity_to_timeseries(self, derivative_data_dict):
        """Weist jedem Datenpunkt seinen Komplexitätsscore zu."""
        combined_data = []
        
        for derivative_name, data_df in derivative_data_dict.items():
            complexity = self.calculate_complexity_score(derivative_name)
            if complexity is not None and not data_df.empty:
                temp_df = data_df.copy()
                temp_df['Derivative'] = derivative_name
                temp_df['Complexity'] = complexity
                temp_df = temp_df.reset_index()
                combined_data.append(temp_df)
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            return result
        else:
            return pd.DataFrame()