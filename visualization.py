import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for static plots (fallback)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- CONFIGURATION (English) ---
METRIC_CONFIG = {
    'structure_layers': {
        'label': 'Structural Layers',
        'desc': 'Legal structure complexity (Physical vs. Swap vs. OTC).',
        'rationale': 'High (25%): Determines transparency & legal certainty.'
    },
    'pricing_complexity': {
        'label': 'Pricing Model',
        'desc': 'Valuation difficulty (Market price vs. Model).',
        'rationale': 'High (25%): Valuation uncertainty drives systemic risk.'
    },
    'leverage_factor': {
        'label': 'Leverage Factor',
        'desc': 'Sensitivity multiplier relative to the underlying.',
        'rationale': 'Medium (20%): Amplifies gains & losses.'
    },
    'counterparty_risk': {
        'label': 'Counterparty Risk',
        'desc': 'Default risk of the issuer or swap partner.',
        'rationale': 'Low (15%): Relevant for synthetic/OTC products.'
    },
    'liquidity_score': {
        'label': 'Liquidity (Inverse)',
        'desc': 'Tradability (Ease of exit during crisis).',
        'rationale': 'Negative (-15%): High liquidity reduces risk.'
    }
}

# --- PLOTLY CHARTS (Interactive) ---

def plot_dynamic_correlation(corr_df):
    """
    Visualizes the temporal evolution of correlation (Hypothesis Stability).
    """
    if corr_df.empty:
        st.warning("Insufficient data points for dynamic correlation.")
        return

    fig = go.Figure()
    
    # Correlation Line
    fig.add_trace(go.Scatter(
        x=corr_df.index, 
        y=corr_df['Correlation'],
        mode='lines',
        name='Correlation (Complexity vs. Volatility)',
        line=dict(color='#2E86C1', width=2)
    ))
    
    # Moving Average Trend
    corr_smooth = corr_df['Correlation'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(
        x=corr_df.index,
        y=corr_smooth,
        mode='lines',
        name='Trend (30-Day Avg)',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    # Reference Lines
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Hypothesis Stability: Correlation over Time',
        xaxis_title='Date',
        yaxis_title='Correlation (Pearson)',
        yaxis=dict(range=[-0.5, 1.1]), 
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    avg_corr = corr_df['Correlation'].mean()
    st.info(f"Insight: The average correlation is {avg_corr:.2f}. Note the spikes during market stress (e.g., 2008, 2020) - complexity often dominates risk during crises.")

def plot_volatility_analysis_interactive(derivative_data_dict, gold_spot_data, window):
    """
    Combined Visualization: Overall Comparison and Faceted Plot.
    """
    st.markdown("### A) Overall Comparison (Interactive)")
    st.info("Tip: Double-click a name in the legend to isolate the line.")

    # 1. Determine Data Range
    all_dates = []
    if gold_spot_data is not None and not gold_spot_data.empty:
        all_dates.extend(gold_spot_data.index)
    
    for df in derivative_data_dict.values():
        if not df.empty:
            all_dates.extend(df.index)
            
    if not all_dates:
        st.warning("No data available to plot.")
        return

    min_date = pd.to_datetime(min(all_dates))
    max_date = pd.to_datetime(max(all_dates))

    # 2. Main Plot
    fig_main = go.Figure()

    if gold_spot_data is not None:
        if 'Volatility' in gold_spot_data.columns:
            vol_data = gold_spot_data['Volatility']
        elif 'Returns' in gold_spot_data.columns:
            vol_data = gold_spot_data['Returns'].rolling(window=window).std() * np.sqrt(252)
        else:
            vol_data = None

        if vol_data is not None:
            fig_main.add_trace(go.Scatter(
                x=gold_spot_data.index, 
                y=vol_data,
                name='Gold Spot (Benchmark)',
                line=dict(color='gold', width=4),
                mode='lines',
                zorder=10
            ))

    colors = px.colors.qualitative.Plotly
    combined_data_list = []
    
    for i, (name, data) in enumerate(derivative_data_dict.items()):
        if name != 'Gold_Spot' and not data.empty:
            if 'Volatility' in data.columns:
                y_vals = data['Volatility']
            elif 'Returns' in data.columns:
                y_vals = data['Returns'].rolling(window=window).std() * np.sqrt(252)
            else:
                continue

            # CLEAN NAME
            clean_name = name.replace('_', ' ')
            
            color = colors[i % len(colors)]
            fig_main.add_trace(go.Scatter(
                x=data.index, 
                y=y_vals,
                name=clean_name,
                line=dict(width=1.5, color=color),
                opacity=0.8,
                visible='legendonly' if i > 5 else True
            ))
            
            combined_data_list.append(pd.DataFrame({
                'Date': data.index, 
                'Volatility': y_vals, 
                'Derivative': clean_name
            }))
    
    # --- EVENT ANNOTATIONS ---
    market_events = [
        ("2001-09-11", "9/11 Attacks"),
        ("2003-03-20", "Iraq War Start"),
        ("2004-11-18", "Launch of GLD ETF"), 
        ("2008-09-15", "Lehman Brothers Collapse"),
        ("2010-05-02", "Euro Crisis"),
        ("2011-09-06", "Gold All-Time High 2011"),
        ("2013-04-12", "2013 Gold Crash"),
        ("2020-03-16", "COVID-19 Crash"),
        ("2022-02-24", "Ukraine Invasion"),
        ("2023-03-10", "US Banking Crisis")
    ]
    
    for date_str, label in market_events:
        event_date = pd.to_datetime(date_str)
        if min_date <= event_date <= max_date:
            fig_main.add_vline(x=event_date, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig_main.add_annotation(
                x=event_date, 
                y=1.0, 
                yref="paper", 
                text=label, 
                showarrow=False, 
                xanchor="right", 
                textangle=-90, 
                font=dict(size=10, color="gray")
            )

    fig_main.update_layout(
        title='Comparison: Volatility of Derivatives vs. Spot (with Events)',
        xaxis_title='Date', yaxis_title='Volatility (Annualized)',
        template='plotly_white', hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[min_date, max_date]) 
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # 3. Faceted Plot
    st.markdown("### B) Individual Trends (Separated Grid)")
    if combined_data_list:
        df_long = pd.concat(combined_data_list, ignore_index=True)
        fig_facet = px.line(
            df_long, x='Date', y='Volatility', color='Derivative',
            facet_col='Derivative', facet_col_wrap=3,
            height=300 * (len(combined_data_list) // 3 + 1),
            title="Individual Volatility Trajectories"
        )
        
        fig_facet.update_xaxes(showticklabels=True, range=[min_date, max_date])
        fig_facet.update_layout(template='plotly_white', showlegend=False)
        fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig_facet, use_container_width=True)
    else:
        st.warning("Insufficient data for faceted plot.")

def plot_residuals_interactive(df: pd.DataFrame, results: dict, best_model_name: str):
    """
    Displays residuals (errors) of the best model.
    """
    if best_model_name not in results or 'error' in results[best_model_name]:
        return

    st.subheader("Residual Analysis (Error Check)")
    st.markdown(f"Shows the deviation of actual data from the **{best_model_name}**. Ideally, points scatter randomly around zero.")

    model_data = results[best_model_name]
    
    x_vals = df['Complexity'].values
    y_true = df['Volatility'].values
    
    names = [n.replace('_', ' ') for n in df['Derivative'].values]
    
    if 'params' not in model_data:
        return

    if best_model_name == 'Potenzmodell': 
        a, alpha = model_data['params']
        y_pred = a * np.power(x_vals, alpha)
    elif best_model_name == 'Linear':
        m, c = model_data['params']
        y_pred = m * x_vals + c
    elif best_model_name == 'Exponentiell':
        a, b = model_data['params']
        y_pred = a * np.exp(b * x_vals)
    else:
        return

    residuals = y_true - y_pred
    
    res_df = pd.DataFrame({
        'Complexity': x_vals,
        'Residual': residuals,
        'Derivative': names,
        'Predicted': y_pred,
        'Actual': y_true
    })

    fig = go.Figure()

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Perfect Fit")

    fig.add_trace(go.Scatter(
        x=res_df['Complexity'],
        y=res_df['Residual'],
        mode='markers',
        text=res_df['Derivative'],
        marker=dict(
            size=12,
            color=res_df['Residual'],
            colorscale='RdBu_r', 
            showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate="<b>%{text}</b><br>Complexity: %{x:.1f}<br>Error: %{y:.4f}<br>Predicted: %{customdata[0]:.4f}<br>Actual: %{customdata[1]:.4f}<extra></extra>",
        customdata=res_df[['Predicted', 'Actual']]
    ))

    fig.update_layout(
        title=f'Residuals of {best_model_name}',
        xaxis_title='Complexity (Score)',
        yaxis_title='Residual (Actual - Model)',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    
    max_err_idx = np.argmax(np.abs(residuals))
    outlier = names[max_err_idx]
    st.info(f"Interpretation: The largest outlier is **{outlier}**. The model {'underestimated' if residuals[max_err_idx] > 0 else 'overestimated'} volatility by {abs(residuals[max_err_idx]):.3f}.")

def plot_scatter(df: pd.DataFrame):
    """Scatterplot Complexity vs Volatility."""
    df_clean = df.copy()
    df_clean['Derivative'] = df_clean['Derivative'].str.replace('_', ' ')
    
    fig = px.scatter(
        df_clean, x='Complexity', y='Volatility', color='Derivative',
        hover_data=['Date'], title='Complexity vs. Volatility: Cluster Analysis',
        trendline="ols"
    )
    fig.update_layout(template='plotly_white', height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_complexity_distribution(complexity_df: pd.DataFrame):
    """Distribution of Scores."""
    complexity_df = complexity_df.sort_values('Complexity_Score')
    complexity_df.index = complexity_df.index.str.replace('_', ' ')
    
    fig = px.bar(
        complexity_df, x='Complexity_Score', y=complexity_df.index, orientation='h',
        color='Complexity_Score', color_continuous_scale='Viridis',
        title='Complexity Ranking'
    )
    fig.update_layout(yaxis_title="Derivative", xaxis_title="Score", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def plot_model_fits(df: pd.DataFrame, results: dict):
    """Shows fitted models."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Complexity'], y=df['Volatility'], mode='markers',
        name='Observations (Mean)', marker=dict(size=10, color='grey')
    ))
    
    x_sorted = np.sort(df['Complexity'].unique())
    colors = {'Exponentiell': 'red', 'Linear': 'blue', 'Potenzmodell': 'green'}
    
    for model_name, model_data in results.items():
        display_name = model_name
        if model_name == 'Potenzmodell': display_name = 'Power Law'
        if model_name == 'Exponentiell': display_name = 'Exponential'
        
        if 'error' not in model_data and 'params' in model_data:
            if model_name == 'Potenzmodell':
                a, alpha = model_data['params']
                y_fit = a * np.power(x_sorted, alpha)
            elif model_name == 'Linear':
                m, c = model_data['params']
                y_fit = m * x_sorted + c
            elif model_name == 'Exponentiell':
                a, b = model_data['params']
                y_fit = a * np.exp(b * x_sorted)
            else:
                continue
                
            fig.add_trace(go.Scatter(
                x=x_sorted, y=y_fit, mode='lines',
                name=f"{display_name} (R²={model_data['R2']:.3f})",
                line=dict(color=colors.get(model_name, 'black'), width=2)
            ))

    fig.update_layout(
        title='Model Fit: σ(C) Hypothesis Test',
        xaxis_title='Complexity (C)', yaxis_title='Volatility (σ)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_complexity_breakdown_chart(classifier):
    """
    Stacked Bar Chart for Complexity Components.
    UPDATED: With highlighted total score for better readability.
    """
    st.subheader("Contribution of Components to Total Complexity")
    derivatives = list(classifier.complexity_scores.keys())
    data = []
    
    for derivative in derivatives:
        metrics = classifier.complexity_scores[derivative]
        row = {'Derivat': derivative.replace('_', ' ')}
        for comp, weight in classifier.weights.items():
            if comp in METRIC_CONFIG:
                value = metrics[comp]
                contribution = value * weight
                label = METRIC_CONFIG[comp]['label']
                row[label] = contribution
        row['Gesamt'] = classifier.calculate_complexity_score(derivative)
        data.append(row)
    
    df = pd.DataFrame(data).sort_values('Gesamt')
    fig = go.Figure()
    labels = [conf['label'] for key, conf in METRIC_CONFIG.items()]
    
    for label in labels:
        if label in df.columns:
            fig.add_trace(go.Bar(
                name=label, y=df['Derivat'], x=df[label], orientation='h',
                hovertemplate=f"<b>{label}</b><br>Contribution: %{{x:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        barmode='relative', title='Breakdown: Drivers of the Complexity Score',
        xaxis_title='Contribution to Score', yaxis_title='',
        height=600, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(r=50) # Increased right margin
    )
    
    # UPDATE: Highlighting the total score
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Gesamt'], y=row['Derivat'], 
            text=f"<b>{row['Gesamt']:.1f}</b>",
            xanchor='left', 
            showarrow=False, 
            xshift=10,
            bgcolor="rgba(255, 255, 255, 0.9)", # White background box
            bordercolor="#333", # Border
            borderwidth=1,
            borderpad=4
        )
    st.plotly_chart(fig, use_container_width=True)

def display_model_metrics_comparison(results):
    """
    Scientific Comparison Table (AIC/BIC).
    """
    st.subheader("Statistical Model Comparison")
    
    metrics_data = []
    for name, res in results.items():
        display_name = name
        if name == 'Potenzmodell': display_name = 'Power Law'
        if name == 'Exponentiell': display_name = 'Exponential'

        if 'error' not in res:
            metrics_data.append({
                'Model': display_name,
                'R2': res.get('R2', 0),
                'RMSE': res.get('RMSE', 0),
                'AIC': res.get('AIC', np.inf),
                'BIC': res.get('BIC', np.inf)
            })
            
    df_metrics = pd.DataFrame(metrics_data).sort_values('AIC')
    
    st.dataframe(
        df_metrics,
        column_config={
            "Model": st.column_config.TextColumn("Model Type", width="medium"),
            "R2": st.column_config.ProgressColumn("R² (Fit)", format="%.3f", min_value=0, max_value=1, help="Explanatory power (higher is better)"),
            "RMSE": st.column_config.NumberColumn("RMSE (Error)", format="%.4f", help="Root Mean Square Error (lower is better)"),
            "AIC": st.column_config.NumberColumn("AIC (Efficiency)", format="%.1f", help="Akaike Information Criterion (lower is better)"),
            "BIC": st.column_config.NumberColumn("BIC (Robustness)", format="%.1f", help="Bayesian Information Criterion (lower is better)"),
        },
        use_container_width=True,
        hide_index=True
    )
    st.caption("*AIC/BIC: Information criteria. Lower values indicate a better balance between fit and model complexity.*")

def plot_complexity_methodology(classifier):
    """Visualization of Methodology."""
    st.subheader("Methodology & Weighting")
    st.markdown("The Complexity Score (0-10) consists of five weighted factors.")
    
    table_data = []
    for key, weight in classifier.weights.items():
        if key in METRIC_CONFIG:
            info = METRIC_CONFIG[key]
            table_data.append({
                "Factor": info['label'],
                "Weight": weight * 100,
                "Description": info['desc'],
                "Rationale": info['rationale']
            })
            
    st.dataframe(
        pd.DataFrame(table_data),
        column_config={
            "Factor": st.column_config.TextColumn("Factor", width="medium"),
            "Weight": st.column_config.ProgressColumn("Impact", format="%d%%", min_value=-20, max_value=30),
            "Description": st.column_config.TextColumn("Measurement", width="medium"),
            "Rationale": st.column_config.TextColumn("Reasoning", width="medium"),
        },
        use_container_width=True, hide_index=True
    )
    st.latex(r'\text{Total Score} = \sum (\text{Metric}_i \times \text{Weight}_i)')

def create_summary_dashboard(results_dict):
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    
    model_name = results_dict.get('best_model', 'N/A')
    if model_name == 'Potenzmodell': model_name = 'Power Law'
    if model_name == 'Exponentiell': model_name = 'Exponential'
    
    status_en = "Confirmed" if "Bestätigt" in results_dict.get('hypothesis_conclusion', '') or "Confirmed" in results_dict.get('hypothesis_conclusion', '') else "Partial"

    with col1:
        st.metric("Best Model Fit", model_name, delta=f"R² = {results_dict.get('best_r2', 0):.3f}")
    with col2:
        st.metric("Pearson Correlation", f"{results_dict.get('pearson_corr', 0):.3f}", delta=f"p = {results_dict.get('pearson_p', 1):.4f}")
    with col3:
        st.metric("Hypothesis", status_en)

def plot_r2_explanation(y_true, y_pred):
    """R² Visualization (Matplotlib)."""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    x_vals = np.arange(len(y_true))
    ax1.scatter(x_vals, y_true, color='blue', s=50, alpha=0.6, label='Observed')
    ax1.axhline(y=y_mean, color='red', linestyle='--', linewidth=2, label='Mean')
    for x, y in zip(x_vals, y_true):
        ax1.plot([x, x], [y_mean, y], 'r-', alpha=0.3)
    ax1.set_title('SS_tot (Total Variation)\nDeviation from Mean', fontsize=10, fontweight='bold')
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(x_vals, y_true, color='blue', s=50, alpha=0.6)
    ax2.plot(x_vals, y_pred, 'g-', linewidth=2, label='Model')
    for x, y, yp in zip(x_vals, y_true, y_pred):
        ax2.plot([x, x], [y, yp], 'orange', alpha=0.5)
    ax2.set_title('SS_res (Residual Variation)\nModel Error', fontsize=10, fontweight='bold')
    ax2.legend()

    ax3 = axes[2]
    ax3.axis('off')
    text = f"R² = 1 - (SS_res / SS_tot)\nR² = 1 - ({ss_res:.2f} / {ss_tot:.2f})\nR² = {r2:.4f}"
    ax3.text(0.5, 0.6, text, ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax_ins = ax3.inset_axes([0.3, 0.1, 0.4, 0.3])
    x_pos = [0, 1]
    ax_ins.bar(x_pos, [ss_res, ss_tot], color=['orange', 'red'], alpha=0.6)
    ax_ins.set_xticks(x_pos)
    ax_ins.set_xticklabels(['SS_res', 'SS_tot'])
    ax_ins.set_title('Comparison Error vs. Total')
    ax_ins.tick_params(axis='both', which='major', labelsize=8)

    st.pyplot(fig)