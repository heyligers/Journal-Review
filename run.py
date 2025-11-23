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

# Visualization imports
from visualization import (plot_scatter, plot_volatility_analysis_interactive,
                          plot_complexity_distribution, plot_model_fits,
                          create_summary_dashboard, plot_complexity_methodology, 
                          plot_complexity_breakdown_chart, plot_residuals_interactive,
                          plot_dynamic_correlation, display_model_metrics_comparison)

# UI Components import
from ui_components import (setup_sidebar, display_header, display_methodology_section,
                          display_r2_explanation, display_hypothesis_conclusion,
                          display_detailed_model_equations, display_correlation_interpretation,
                          display_downloads, render_tab_footer)

st.set_page_config(page_title="Gold Derivatives Analysis", page_icon=None, layout="wide", initial_sidebar_state="expanded")

def main():
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

        # Modeling
        model_results = {}
        pearson_p, pearson_corr = 1, 0
        best_model, best_r2 = "N/A", 0
        
        if len(avg_vol_by_complexity) >= 3:
            x_model = avg_vol_by_complexity['Complexity'].values
            y_model = avg_vol_by_complexity['Volatility'].values
            
            model_results = fit_models(x_model, y_model)
            # Best model based on R2
            best_model = max(model_results.items(), key=lambda k: k[1].get('R2', -np.inf) if 'error' not in k[1] else -np.inf)[0]
            best_r2 = model_results[best_model].get('R2', 0)
            
            active_methods = [m for m, active in config['correlations'].items() if active]
            if active_methods:
                corr_results = compute_correlations(x_model, y_model, active_methods)
                pearson_stat = corr_results.get('Pearson', {})
                pearson_p = pearson_stat.get('p-value', 1)
                pearson_corr = pearson_stat.get('stat', 0)
    
    # --- UI STRUCTURE ---
    st.markdown("### Analysis Sections")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fermat Connection", "Data & Methodology", "Complexity", "Analysis & Trends", "Modeling & Hypothesis"])

    with tab1:
        create_fermat_animation()
        render_tab_footer()

    with tab2:
        st.header("Data Basis")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"Analysis Period: **{start_date}** to **{end_date}**")
            st.success(f"Loaded: {len(derivative_data)} Derivatives")
        
        with col2:
            with st.expander("Show Loaded Instruments Details", expanded=False):
                # CLEAN NAMES IN DATA TABLE
                data_summary = [{'Derivative': name.replace('_', ' '), 'Records': len(data)} for name, data in derivative_data.items()]
                st.dataframe(pd.DataFrame(data_summary), use_container_width=True, hide_index=True)
        
        display_methodology_section(selected_derivatives, rolling_window, derivative_data)
        st.markdown("---")
        display_downloads(combined_df, derivative_data, start_date, end_date, best_model, best_r2, pearson_corr)
        render_tab_footer()

    with tab3:
        st.header("Structural Complexity Analysis")
        c1, c2 = st.columns([1, 2])
        with c1: st.dataframe(safe_style_format(complexity_scores_df[complexity_scores_df.index.isin(selected_derivatives)][['Complexity_Score']]).background_gradient(cmap='RdYlGn_r'), use_container_width=True)
        with c2: plot_complexity_distribution(complexity_scores_df[complexity_scores_df.index.isin(selected_derivatives)])
        st.markdown("---")
        with st.expander("View Calculation Details"):
            plot_complexity_methodology(classifier)
            plot_complexity_breakdown_chart(classifier)
        render_tab_footer()

    with tab4:
        st.header("Volatility & Trends")
        plot_volatility_analysis_interactive(derivative_data, gold_spot_data, rolling_window)
        st.markdown("---")
        st.subheader("Temporal Stability of Hypothesis")
        st.markdown("How strongly does complexity correlate with volatility over time? (Insight: Correlation often increases during crises)")
        dynamic_corr_df = calculate_dynamic_correlation(combined_df)
        plot_dynamic_correlation(dynamic_corr_df)
        st.markdown("---")
        st.subheader("Cluster Analysis")
        plot_scatter(combined_df)
        render_tab_footer()

    with tab5:
        st.header("Statistical Modeling")
        if len(avg_vol_by_complexity) < 3:
            st.warning("Insufficient data points.")
        else:
            create_summary_dashboard({'best_model': best_model, 'best_r2': best_r2, 'pearson_corr': pearson_corr, 'pearson_p': pearson_p, 'hypothesis_conclusion': 'Confirmed' if best_r2 > 0.7 and pearson_p < 0.05 else 'Partial'})
            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Model Fit")
                plot_model_fits(avg_vol_by_complexity.reset_index(), model_results)
            with c2:
                display_model_metrics_comparison(model_results)
                st.markdown("### Correlation")
                display_correlation_interpretation(pearson_corr, pearson_p)
            st.markdown("---")
            display_detailed_model_equations(model_results)
            plot_residuals_interactive(avg_vol_by_complexity.reset_index(), model_results, best_model)
            st.markdown("---")
            display_hypothesis_conclusion(best_model, best_r2, pearson_p, pearson_corr, model_results)
        render_tab_footer()

if __name__ == "__main__":
    main()