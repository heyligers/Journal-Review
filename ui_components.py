import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_r2_explanation, METRIC_CONFIG
from fermat_connection import add_fermat_section
from utils import safe_style_format

def render_tab_footer():
    """
    Footer with 'Back to top' button.
    """
    st.markdown("---")
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px;">
            <span style="color: grey; font-size: 0.8em;">Gold Derivatives Analysis Dashboard v2.0</span>
            <a href="#top" target="_self" style="
                text-decoration: none;
                background-color: #ffffff;
                color: #31333F;
                padding: 8px 16px;
                border-radius: 5px;
                border: 1px solid #d6d9ef;
                font-weight: 600;
                font-size: 0.9em;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                transition: all 0.2s;
            ">
                Back to top
            </a>
        </div>
        <br>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """Sidebar configuration."""
    st.sidebar.markdown("---")
    
    st.sidebar.header("Configuration")
    
    # 1. Time Period
    st.sidebar.subheader("Time Period")
    min_date = pd.to_datetime('2000-01-01').date()
    max_date = pd.to_datetime('today').date()
    
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2010-01-01').date(), min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today').date(), min_value=min_date, max_value=max_date)
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date!")
        st.stop()
    
    # 2. Derivatives
    st.sidebar.subheader("Derivative Selection")
    selected_derivatives = []
    
    if st.sidebar.checkbox("Gold Spot Price (Benchmark)", value=True): selected_derivatives.append('Gold_Spot')
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("**Physical & Linear**")
    if st.sidebar.checkbox("Gold Futures (GC=F)", value=True): selected_derivatives.append('Gold_Futures')
    if st.sidebar.checkbox("Gold ETF (GLD)", value=True): selected_derivatives.append('Gold_ETF_GLD')
    if st.sidebar.checkbox("Sprott Physical Trust (PHYS)", value=True): selected_derivatives.append('Sprott_Physical_Trust')
    
    st.sidebar.markdown("**Structured ETFs**")
    if st.sidebar.checkbox("Futures Strategy ETF (DGL)", value=True): selected_derivatives.append('Gold_Futures_ETF_DGL')
    if st.sidebar.checkbox("2x Leveraged ETF (UGL)", value=True): selected_derivatives.append('Gold_Leveraged_ETF_2x')
    if st.sidebar.checkbox("Inverse -2x ETF (GLL)", value=False): selected_derivatives.append('Inverse_Gold_ETF_GLL')
    
    st.sidebar.markdown("**OTC & Exotics (Simulated)**")
    if st.sidebar.checkbox("Gold Options (Call ATM)", value=True): selected_derivatives.append('Gold_Options')
    if st.sidebar.checkbox("Vanilla Swap (Standard)", value=True): selected_derivatives.append('Gold_Swap_Vanilla')
    if st.sidebar.checkbox("Customized Swap (Exotic)", value=True): selected_derivatives.append('Gold_Swap_Customized')
    
    # 3. What-If Weighting
    st.sidebar.markdown("---")
    st.sidebar.subheader("Weighting Scenario")
    
    custom_weights = {}
    with st.sidebar.expander("Adjust Weights (What-If)", expanded=False):
        st.info("Define 'Complexity':")
        
        defaults = {
            'structure_layers': 0.25,
            'pricing_complexity': 0.25,
            'leverage_factor': 0.20,
            'counterparty_risk': 0.15,
            'liquidity_score': -0.15
        }
        
        for key, default_val in defaults.items():
            if key in METRIC_CONFIG:
                label = METRIC_CONFIG[key]['label']
                val = st.slider(
                    f"{label}", 
                    min_value=-0.5, 
                    max_value=0.5, 
                    value=default_val, 
                    step=0.05,
                    format="%.2f"
                )
                custom_weights[key] = val
            
    # 4. Calculation
    st.sidebar.markdown("---")
    st.sidebar.subheader("Calculation")
    rolling_window = st.sidebar.slider("Rolling Window (Days)", 10, 252, 30)
    
    st.sidebar.subheader("Correlations")
    correlations = {
        'Pearson': st.sidebar.checkbox("Pearson", value=True),
        'Spearman': st.sidebar.checkbox("Spearman", value=True),
        'Kendall': st.sidebar.checkbox("Kendall", value=False)
    }
    
    if len(selected_derivatives) == 0:
        st.error("Please select at least one derivative.")
        st.stop()
        
    return {
        'start_date': start_date, 'end_date': end_date,
        'selected_derivatives': selected_derivatives, 'rolling_window': rolling_window,
        'correlations': correlations,
        'custom_weights': custom_weights
    }

def display_header():
    st.title("Hypothesis Test: Power Law of Volatility")
    st.markdown("""
    **Research Question:** Does the volatility of gold financial products increase disproportionately with structural complexity?
    
    We analyze pure gold derivatives ranging from physical trusts to exotic OTC swaps.
    """)

def display_methodology_section(selected_derivatives, rolling_window, derivative_data):
    with st.expander("Details on Data Sources & Simulations", expanded=False):
        st.markdown("### Classification")
        st.info("""
        **Linear / Physical:** GLD, PHYS (1:1 tracking)
        **Structured:** DGL (Futures Roll), UGL (2x Leverage), GLL (Inverse -2x)
        **Simulated (OTC):** Options (Gamma), Swaps (Counterparty & Barrier Risks)
        """)
        st.markdown("### Simulation Logic")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Vanilla Swap**")
            st.latex(r"R_{vanilla} = R_{base} \times 2.5 + \epsilon")
        with c2:
            st.markdown("**Customized Swap**")
            st.latex(r"R_{custom} = R_{base} \times 3.0 + Jumps(\lambda=0.01)")

def display_r2_explanation(y_true, y_pred, model_name, ss_res, ss_tot, r2_calculated):
    with st.expander("What does R² mean?", expanded=False):
        st.markdown("### R² - Coefficient of Determination")
        st.latex(r'R^2 = 1 - \frac{SS_{res}}{SS_{tot}}')
        col1, col2, col3 = st.columns(3)
        col1.metric("SS_res (Model Error)", f"{ss_res:.6f}")
        col2.metric("SS_tot (Total Variation)", f"{ss_tot:.6f}")
        col3.metric("R² Calculated", f"{r2_calculated:.4f}")
        plot_r2_explanation(y_true, y_pred)

def display_detailed_model_equations(results):
    with st.expander("View Detailed Model Equations"):
        for model_name, model_data in results.items():
            # Map names for display
            display_name = model_name
            if model_name == 'Potenzmodell': display_name = 'Power Law'
            if model_name == 'Exponentiell': display_name = 'Exponential'

            if 'error' not in model_data:
                st.markdown(f"### {display_name}")
                if model_name == 'Exponentiell' and 'params' in model_data:
                    a, b = model_data['params']
                    st.latex(f"\\sigma(C) = {a:.6f} \cdot e^{{{b:.6f} \cdot C}}")
                elif model_name == 'Linear' and 'params' in model_data:
                    m, c = model_data['params']
                    st.latex(f"\\sigma(C) = {m:.6f} \cdot C + {c:.6f}")
                elif model_name == 'Potenzmodell' and 'params' in model_data:
                    a, b = model_data['params']
                    st.latex(f"\\sigma(C) = {a:.6f} \cdot C^{{{b:.6f}}}")
                st.markdown(f"- R² = {model_data['R2']:.4f}")
                st.markdown(f"- RMSE = {model_data['RMSE']:.6f}")
                st.markdown("---")

def display_correlation_interpretation(pearson_corr, pearson_p):
    col1, col2 = st.columns(2)
    with col1:
        if pearson_p < 0.05:
            st.success(f"**Significant Correlation!**\n\nr = {pearson_corr:.3f} (p < 0.05)")
        else:
            st.warning(f"**No Significant Correlation**\n\np = {pearson_p:.3f}")
    with col2:
        if abs(pearson_corr) < 0.3: effect = "Weak"; color = "🔵"
        elif abs(pearson_corr) < 0.7: effect = "Moderate"; color = "🟡"
        else: effect = "Strong"; color = "🟢"
        st.info(f"{color} **Effect Size: {effect}**")

def display_hypothesis_conclusion(best_model, best_r2, pearson_p, pearson_corr, results):
    st.header("Hypothesis Test Result")
    power_best = (best_model == 'Potenzmodell')
    high_r2 = (best_r2 > 0.7)
    sig_corr = (pearson_p < 0.05)
    pos_corr = (pearson_corr > 0.5)
    criteria_met = sum([power_best, high_r2, sig_corr, pos_corr])
    
    st.subheader("Evaluation Criteria")
    c1, c2 = st.columns(2)
    c1.metric("Power Law is Best Fit", "Yes" if power_best else "No")
    c1.metric("R² > 0.7", f"Yes ({best_r2:.3f})" if high_r2 else f"No ({best_r2:.3f})")
    c2.metric("Significant Correlation", "Yes" if sig_corr else "No")
    c2.metric("Strong Pos. Correlation (>0.5)", "Yes" if pos_corr else "No")
    
    st.subheader("Conclusion")
    if criteria_met >= 3:
        st.success(f"### HYPOTHESIS CONFIRMED ({criteria_met}/4 Criteria)\nData strongly supports the **Power Law Hypothesis**.")
    elif criteria_met >= 2:
        st.warning("### PARTIALLY CONFIRMED")
    else:
        st.error("### HYPOTHESIS REJECTED")

def display_downloads(combined_df, derivative_data, start_date, end_date, best_model, best_r2, pearson_corr):
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    with col1:
        csv = combined_df.to_csv(index=False)
        st.download_button("Download Raw Data (CSV)", data=csv, file_name="gold_derivatives_analysis.csv", mime="text/csv")
    with col2:
        summary = f"Analysis {start_date} to {end_date}\nR²: {best_r2:.4f}"
        st.download_button("Download Summary (TXT)", data=summary, file_name="summary.txt", mime="text/plain")