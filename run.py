import streamlit as st
import pandas as pd
import numpy as np

# Module imports
from data_loader import load_multiple_gold_derivatives
from derivative_complexity import DerivativeComplexityClassifier
from volatility import calculate_all_volatilities
from models import fit_models
from analysis import compute_correlations, calculate_dynamic_correlation
from utils import safe_style_format
from fermat_connection import create_fermat_animation
# Importiere Simulation Logic direkt, um den Code anzeigen zu können
from financial_models import black_scholes_call_price, simulate_barrier_option

# Visualization imports
from visualization import (plot_scatter, plot_volatility_analysis_interactive,
                          plot_complexity_distribution, plot_model_fits,
                          create_summary_dashboard, plot_complexity_methodology, 
                          plot_complexity_breakdown_chart, plot_residuals_interactive,
                          plot_dynamic_correlation, display_model_metrics_comparison,
                          display_scoring_rubric)

# UI Components import
from ui_components import (setup_sidebar, display_header, display_methodology_section,
                          display_r2_explanation, display_hypothesis_conclusion,
                          display_detailed_model_equations, display_correlation_interpretation,
                          display_downloads, render_tab_footer, apply_presentation_style,
                          display_code_snippet, display_limitations_section) 

st.set_page_config(page_title="Gold Derivatives Analysis", page_icon=None, layout="wide", initial_sidebar_state="expanded")

def main():
    # AKTIVIERT DEN CLEAN MODE FÜR PRÄSENTATIONEN
    apply_presentation_style()
    
    # HTML Anchor for "Back to top"
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    display_header()
    config = setup_sidebar()
    
    start_date = config['start_date']
    end_date = config['end_date']
    selected_derivatives = config['selected_derivatives']
    rolling_window = config['rolling_window']
    custom_weights = config['custom_weights']
    
    # --- CALCULATION ---
    with st.spinner("Loading data and performing calculations..."):
        derivative_data, gold_spot_data = load_multiple_gold_derivatives(start_date, end_date, selected_derivatives)
        
        if not derivative_data:
            st.error("No data loaded.")
            st.stop()

        for name, data in derivative_data.items():
            derivative_data[name] = calculate_all_volatilities(data, rolling_window)
        
        # FIX: Capping for Barrier Option to keep graph readable (Presentation Fix)
        if 'Gold_Barrier_Option' in derivative_data:
            if 'Volatility' in derivative_data['Gold_Barrier_Option'].columns:
                 derivative_data['Gold_Barrier_Option']['Volatility'] = derivative_data['Gold_Barrier_Option']['Volatility'].clip(upper=3.0)

        if gold_spot_data is not None:
            gold_spot_data = calculate_all_volatilities(gold_spot_data, rolling_window)

        classifier = DerivativeComplexityClassifier()
        if custom_weights:
            classifier.update_weights(custom_weights)
            
        complexity_scores_df = classifier.get_all_complexity_scores()
        
        combined_df = classifier.assign_complexity_to_timeseries(derivative_data)
        if combined_df.empty: st.stop()
             
        combined_df = combined_df.dropna(subset=['Volatility', 'Complexity'])
        avg_vol_by_complexity = combined_df.groupby('Derivative').agg({'Complexity': 'first', 'Volatility': 'mean'}).sort_values('Complexity')

        # Modeling & Statistics
        model_results = {}
        pearson_p, pearson_corr = 1, 0
        best_model, best_r2 = "N/A", 0
        
        df_modeling = avg_vol_by_complexity[avg_vol_by_complexity['Complexity'] > 0]
        
        if len(df_modeling) >= 3:
            x_model = df_modeling['Complexity'].values
            y_model = df_modeling['Volatility'].values
            
            # 1. Fit Models
            model_results = fit_models(x_model, y_model)
            
            # Best model selection
            best_model = max(model_results.items(), key=lambda k: k[1].get('R2', -np.inf) if 'error' not in k[1] else -np.inf)[0]
            best_r2 = model_results[best_model].get('R2', 0)
            
            # 2. Calculate Correlations (on SAME data subset)
            active_methods = [m for m, active in config['correlations'].items() if active]
            if active_methods:
                corr_results = compute_correlations(x_model, y_model, active_methods)
                pearson_stat = corr_results.get('Pearson', {})
                pearson_p = pearson_stat.get('p-value', 1)
                pearson_corr = pearson_stat.get('stat', 0)
        else:
            st.warning("Insufficient data for modeling (need at least 3 derivatives with Complexity > 0).")
    
    # --- UI STRUCTURE ---
    st.markdown("### Analysis Sections")
    # UPDATE: Neuer Tab "Critique & Limitations"
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Fermat Connection", 
        "Data & Methodology", 
        "Complexity", 
        "Analysis & Trends", 
        "Modeling & Hypothesis",
        "Critique & Limitations"
    ])

    with tab1:
        create_fermat_animation()
        # SHOW CODE: Fermat Logic
        display_code_snippet(create_fermat_animation, "Show Code: Fermat Animation")
        render_tab_footer()

    with tab2:
        st.header("Data Basis")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"Analysis Period: **{start_date}** to **{end_date}**")
            st.success(f"Loaded: {len(derivative_data)} Derivatives")
        
        with col2:
            with st.expander("Show Loaded Instruments Details", expanded=False):
                data_summary = [{'Derivative': name.replace('_', ' '), 'Records': len(data)} for name, data in derivative_data.items()]
                # FIX: width="stretch"
                st.dataframe(pd.DataFrame(data_summary), width="stretch", hide_index=True)
        
        display_methodology_section(selected_derivatives, rolling_window, derivative_data)
        
        # SHOW CODE: Mathematical Models
        st.markdown("#### Implementation Details")
        display_code_snippet(black_scholes_call_price, "Show Code: Black-Scholes Formula (Python)")
        display_code_snippet(simulate_barrier_option, "Show Code: Barrier Option Simulation Logic")
        
        st.markdown("---")
        display_downloads(combined_df, derivative_data, start_date, end_date, best_model, best_r2, pearson_corr)
        render_tab_footer()

    with tab3:
        st.header("Structural Complexity Analysis")
        c1, c2 = st.columns([1, 2])
            
        with c1: 
            # 1. Daten filtern & kopieren
            display_scores_df = complexity_scores_df[complexity_scores_df.index.isin(selected_derivatives)].copy()
                
            # 2. Index bereinigen (Unterstriche weg)
            display_scores_df.index = display_scores_df.index.str.replace('_', ' ')
            display_scores_df.index.name = "Derivative"
                
            # 3. Spalte für die Anzeige umbenennen (Complexity_Score -> Complexity Score)
            df_view = display_scores_df[['Complexity_Score']].rename(columns={'Complexity_Score': 'Complexity Score'})
                
            # 4. Anzeigen
            st.dataframe(
                safe_style_format(df_view).background_gradient(cmap='RdYlGn_r'), 
                width="stretch"
            )
                
        with c2: 
            plot_complexity_distribution(complexity_scores_df[complexity_scores_df.index.isin(selected_derivatives)])
            # SHOW CODE: Plotting
            display_code_snippet(plot_complexity_distribution, "Show Code: Complexity Plot")

        st.markdown("---")
        with st.expander("View Calculation Details"):
            plot_complexity_methodology(classifier)
            st.markdown("---")
            display_scoring_rubric()
            st.markdown("---")
            plot_complexity_breakdown_chart(classifier)
            # SHOW CODE: Classifier Logic
            st.markdown("#### Classifier Logic")
            display_code_snippet(classifier.calculate_complexity_score, "Show Code: Score Calculation Method")
                
        render_tab_footer()

    with tab4:
        st.header("Volatility & Trends")
        plot_volatility_analysis_interactive(derivative_data, gold_spot_data, rolling_window)
        # SHOW CODE: Volatility Plot
        display_code_snippet(plot_volatility_analysis_interactive, "Show Code: Interactive Volatility Plot")
        
        st.markdown("---")
        st.subheader("Temporal Stability of Hypothesis")
        st.markdown("Correlation between Complexity and Volatility over time.")
        dynamic_corr_df = calculate_dynamic_correlation(combined_df)
        plot_dynamic_correlation(dynamic_corr_df)
        # SHOW CODE: Dynamic Correlation
        display_code_snippet(calculate_dynamic_correlation, "Show Code: Rolling Correlation Calculation")
        
        st.markdown("---")
        st.subheader("Cluster Analysis")
        st.caption("Visualizing the groupings of different product types based on their risk/complexity profile.")
        plot_scatter(combined_df)
        # SHOW CODE: Scatter Plot
        display_code_snippet(plot_scatter, "Show Code: Scatter Visualization")
        
        render_tab_footer()

    with tab5:
        st.header("Statistical Modeling")
        if len(df_modeling) < 3:
            st.warning("Insufficient data points for statistical significance.")
        else:
            create_summary_dashboard({'best_model': best_model, 'best_r2': best_r2, 'pearson_corr': pearson_corr, 'pearson_p': pearson_p})
            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Model Fit")
                plot_model_fits(df_modeling.reset_index(), model_results) 
                # SHOW CODE: Model Fit Plot
                display_code_snippet(plot_model_fits, "Show Code: Model Fit Visualization")
                
            with c2:
                display_model_metrics_comparison(model_results)
                st.markdown("### Correlation")
                display_correlation_interpretation(pearson_corr, pearson_p)
                # SHOW CODE: Correlation Math
                display_code_snippet(compute_correlations, "Show Code: Pearson/Spearman Logic")
                
            st.markdown("---")
            
            # SHOW CODE: SciPy Fitting Logic
            st.subheader("Scientific Curve Fitting")
            display_code_snippet(fit_models, "Show Code: SciPy Curve Fitting Logic")
            
            # Detailed Equations with Explanations
            display_detailed_model_equations(model_results)
            
            st.subheader("Residual Analysis")
            plot_residuals_interactive(df_modeling.reset_index(), model_results, best_model)
            # SHOW CODE: Residuals
            display_code_snippet(plot_residuals_interactive, "Show Code: Residuals Plotting")
            
            st.markdown("---")
            display_hypothesis_conclusion(best_model, best_r2, pearson_p, pearson_corr, model_results)
        render_tab_footer()

    with tab6:
        display_limitations_section()
        render_tab_footer()

if __name__ == "__main__":
    main()
