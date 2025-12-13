import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from plotly.subplots import make_subplots

# Set style for static plots (fallback)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- CONFIGURATION & SCIENTIFIC RATIONALE ---
METRIC_CONFIG = {
    'structure_layers': {
        'label': 'Structure Layers',
        'weight': 0.25,
        'desc': 'Legal distance from the physical asset (Trust > Swap > Futures > Spot).',
        'rationale': (
            "<b>Why 25% (High Impact)?</b><br>"
            "This metric measures <i>Structural Opacity</i>. Every legal layer between the investor "
            "and the gold bar introduces friction and legal risk. In the 2008 crisis, it was "
            "structural complexity (CDO-squared), not the underlying asset, that caused the collapse. "
            "Therefore, this is a primary driver of the Complexity Score."
        )
    },
    'pricing_complexity': {
        'label': 'Pricing Model',
        'weight': 0.25,
        'desc': 'Mathematical difficulty to determine fair value (Market vs. Black-Scholes).',
        'rationale': (
            "<b>Why 25% (High Impact)?</b><br>"
            "This represents <i>Model Risk</i>. If a product price cannot be observed on an exchange "
            "but must be calculated via simulation (e.g., Monte Carlo or Black-Scholes), it carries "
            "inherent estimation errors. If the map (model) is wrong, the territory (risk) is unknown."
        )
    },
    'leverage_factor': {
        'label': 'Leverage Factor',
        'weight': 0.20,
        'desc': 'Multiplier of returns (e.g., 2x, 3x).',
        'rationale': (
            "<b>Why 20% (Medium Impact)?</b><br>"
            "Leverage amplifies volatility linearly ($2x$). While dangerous, it is mathematically "
            "transparent and predictable. Unlike structural opacity (which hides risk), leverage "
            "simply magnifies it. Hence, slightly lower weight than Structure/Pricing."
        )
    },
    'counterparty_risk': {
        'label': 'Counterparty Risk',
        'weight': 0.15,
        'desc': 'Risk of issuer default (ETN vs. ETF).',
        'rationale': (
            "<b>Why 15% (Low-Medium Impact)?</b><br>"
            "This captures <i>Credit Risk</i>. Physical ETFs are ring-fenced assets. ETNs (Notes) "
            "are unsecured debt. While critical, most exchange-traded products are collateralized, "
            "making this a secondary risk factor compared to the product's internal mechanics."
        )
    },
    'liquidity_score': {
        'label': 'Liquidity (Inverse)',
        'weight': -0.15,
        'desc': 'Ease of trading/exit capacity.',
        'rationale': (
            "<b>Why -15% (Negative/Dampening)?</b><br>"
            "Liquidity is the <i>Antidote to Complexity</i>. If you can exit a position instantly "
            "(high liquidity), your effective risk is lower. Therefore, high liquidity "
            "<b>subtracts</b> from the complexity score (Negative Weight)."
        )
    }
}

# --- SCORING RUBRIC (TRANSPARENCY) ---
def display_scoring_rubric():
    """
    Displays the rubric used to assign base scores.
    """
    st.markdown("### 🔍 Transparency: The Scoring Rubric")
    
    st.info("""
    **The 'Bucket' Logic:**
    Base values are assigned based on three risk categories:
    * **0 - 2 (Physical / Low):** Direct ownership, simple trusts, or linear assets.
    * **3 - 6 (Synthetic / Med):** Contract-based replication (Swaps, Futures, Notes).
    * **7 - 10 (Exotic / High):** Non-linear math, path-dependency, or extreme leverage.
    """)
    
    st.markdown("#### Definition of Base Values (Pre-Weighting)")
    
    # FIX: Alle Scores als String ("0", "2"), damit "10+" keinen Typ-Fehler auslöst
    data = [
        # Structure
        ("Structure Layers", "0", "No Layer", "Direct Ownership (Spot)"),
        ("Structure Layers", "2", "Trust / Fund", "Legal Entity holds asset (GLD, PHYS)"),
        ("Structure Layers", "4", "Debt Instrument", "Unsecured Bank Note (ETN)"),
        ("Structure Layers", "5", "OTC Contract", "Bilateral derivative contract (Barrier)"),
        
        # Pricing
        ("Pricing Model", "1", "Market Observable", "Price is live on exchange (Futures)"),
        ("Pricing Model", "2", "NAV Derived", "Assets minus Fees (ETF)"),
        ("Pricing Model", "4", "Analytical (BSM)", "Black-Scholes Formula (Vanilla Opt)"),
        ("Pricing Model", "6", "Numerical / Sim", "Monte Carlo / Path Dependent (Exotic)"),
        
        # Leverage
        ("Leverage", "1", "1:1", "No Leverage"),
        ("Leverage", "2", "2:1", "Daily resets (Lev ETF)"),
        ("Leverage", "10+", "High", "Implicit Margin Leverage (Futures/Opt)"),
        
        # Counterparty
        ("Counterparty Risk", "0", "None", "Physical Possession"),
        ("Counterparty Risk", "2", "Low", "Ring-fenced assets (ETF)"),
        ("Counterparty Risk", "5", "High", "Issuer Default Risk (ETN)"),
        
        # Liquidity
        ("Liquidity (Inverse)", "5", "Very High", "Deep Market, Instant Exit (Spot/Futures)"),
        ("Liquidity (Inverse)", "3", "Medium", "Market Maker Dependent (ETN)"),
        ("Liquidity (Inverse)", "1", "Very Low", "OTC / Illiquid (Barrier/Exotic)")
    ]
    
    df = pd.DataFrame(data, columns=["Category", "Score", "Type", "Description"])
    # Falls st.table Probleme macht, nutzen wir st.dataframe mit String-Typen
    st.dataframe(df, width="stretch", hide_index=True)

# --- GAMMA VISUALIZATION ---
def plot_gamma_explanation():
    """Visualizes the Gamma effect."""
    st.markdown("### Deep Dive: Why Black-Scholes?")
    st.info("The 'Gamma' represents the curvature of the price curve. While an ETF (linear) reacts symmetrically to moves, options benefit disproportionately from large moves due to convexity.")

    spot_start = 100
    moves = np.linspace(0.8, 1.2, 100)
    spots = spot_start * moves
    
    pnl_linear = (spots - spot_start) 

    def get_bs_price(S, K=100, T=1, r=0.04, sigma=0.2):
        if S <= 0: return 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

    price_start = get_bs_price(spot_start)
    prices_new = [get_bs_price(s) for s in spots]
    pnl_option = np.array(prices_new) - price_start

    delta_approx = 0.6
    pnl_option_scaled = pnl_option * (1 / delta_approx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=pnl_linear, mode='lines', name='Linear (ETF)', line=dict(color='gray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=spots, y=pnl_option_scaled, mode='lines', name='Convex (Option)', line=dict(color='#2E86C1', width=3)))

    fig.add_annotation(x=118, y=18, text="Linear Gain", showarrow=False, yshift=-10, font=dict(size=10, color="gray"))
    fig.add_annotation(x=118, y=pnl_option_scaled[-1], text="Gamma Boost", showarrow=True, ax=-40, ay=0, font=dict(color="#2E86C1", weight="bold"))

    fig.update_layout(title='The Gamma Phenomenon: Non-Linear Gains', xaxis_title='Gold Spot Price', yaxis_title='Profit / Loss', template='plotly_white', height=400, hovermode="x unified")
    
    # FIX: width="stretch" statt use_container_width
    st.plotly_chart(fig, width="stretch")

# --- MAIN CHARTS ---

def plot_dynamic_correlation(corr_df):
    if corr_df.empty:
        st.warning("Insufficient data points for dynamic correlation.")
        return

    with st.expander("How to read: Dynamic Correlation", expanded=True):
        st.markdown("""
        **What does this line show?**
        This chart tracks **how well** our hypothesis holds true over time. It measures the correlation between Complexity and Volatility on a rolling basis.
        * **High Positive Value (> 0.5):** **Hypothesis holds.**
        * **Drop to Zero:**  **Market Panic.**
        """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr_df.index, y=corr_df['Correlation'], mode='lines', name='Correlation', line=dict(color='#2E86C1', width=2)))
    corr_smooth = corr_df['Correlation'].rolling(window=30).mean()
    fig.add_trace(go.Scatter(x=corr_df.index, y=corr_smooth, mode='lines', name='Trend (30d)', line=dict(color='#E74C3C', width=2, dash='dash')))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(title='Hypothesis Stability: Correlation over Time', xaxis_title='Date', yaxis_title='Correlation', height=400, template='plotly_white')
    st.plotly_chart(fig, width="stretch")

def plot_volatility_analysis_interactive(derivative_data_dict, gold_spot_data, window):
    st.markdown("### A) Overall Comparison (Interactive)")
    
    with st.expander("How to read: Volatility Timeline", expanded=False):
        st.markdown("""
        **Visualizing Risk Regimes:**
        * **Gold Line (Benchmark):** This is the baseline risk of owning physical gold.
        * **Colored Lines (Derivatives):** These show the risk of financial products.
        """)
        st.info("Tip: Double-click a name in the legend to isolate that specific line.")

    all_dates = []
    if gold_spot_data is not None and not gold_spot_data.empty: all_dates.extend(gold_spot_data.index)
    for df in derivative_data_dict.values():
        if not df.empty: all_dates.extend(df.index)
            
    if not all_dates: return

    min_date, max_date = pd.to_datetime(min(all_dates)), pd.to_datetime(max(all_dates))
    fig_main = go.Figure()

    if gold_spot_data is not None:
        if 'Volatility' in gold_spot_data.columns: vol_data = gold_spot_data['Volatility']
        elif 'Returns' in gold_spot_data.columns: vol_data = gold_spot_data['Returns'].rolling(window=window).std() * np.sqrt(252)
        else: vol_data = None

        if vol_data is not None:
            fig_main.add_trace(go.Scatter(x=gold_spot_data.index, y=vol_data, name='Gold Spot', line=dict(color='gold', width=4), zorder=10))

    colors = px.colors.qualitative.Plotly
    combined_data_list = []
    
    for i, (name, data) in enumerate(derivative_data_dict.items()):
        if name != 'Gold_Spot' and not data.empty:
            if 'Volatility' in data.columns: y_vals = data['Volatility']
            elif 'Returns' in data.columns: y_vals = data['Returns'].rolling(window=window).std() * np.sqrt(252)
            else: continue

            clean_name = name.replace('_', ' ')
            fig_main.add_trace(go.Scatter(x=data.index, y=y_vals, name=clean_name, line=dict(width=1.5, color=colors[i % len(colors)]), opacity=0.8, visible='legendonly' if i > 5 else True))
            combined_data_list.append(pd.DataFrame({'Date': data.index, 'Volatility': y_vals, 'Derivative': clean_name}))
    
    market_events = [
        ("2001-09-11", "9/11 Attacks"),
        ("2008-09-15", "Lehman Collapse"),
        ("2011-09-06", "Gold ATH 2011"),
        ("2020-03-16", "COVID-19 Crash"),
        ("2022-02-24", "Ukraine Invasion"),
        ("2023-03-10", "US Banking Crisis")
    ]
    
    for date_str, label in market_events:
        event_date = pd.to_datetime(date_str)
        if min_date <= event_date <= max_date:
            fig_main.add_vline(x=event_date, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig_main.add_annotation(x=event_date, y=1.0, yref="paper", text=label, showarrow=False, xanchor="right", textangle=-90, font=dict(size=10, color="gray"))

    fig_main.update_layout(title='Comparison: Volatility of Derivatives vs. Spot (with Events)', xaxis_title='Date', yaxis_title='Volatility', template='plotly_white', height=500, xaxis=dict(range=[min_date, max_date]))
    st.plotly_chart(fig_main, width="stretch")

    st.markdown("### B) Individual Trends")
    if combined_data_list:
        df_long = pd.concat(combined_data_list, ignore_index=True)
        fig_facet = px.line(df_long, x='Date', y='Volatility', color='Derivative', facet_col='Derivative', facet_col_wrap=3, height=300 * (len(combined_data_list) // 3 + 1))
        fig_facet.update_xaxes(showticklabels=True)
        fig_facet.update_layout(template='plotly_white', showlegend=False)
        fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_facet, width="stretch")

def plot_residuals_interactive(df, results, best_model_name):
    if best_model_name not in results or 'error' in results[best_model_name]: return
    
    st.subheader("Residual Analysis")
    with st.expander("How to read: Residuals", expanded=False):
        st.markdown("""
        **Quality Control for the Model:**
        * **Ideal Scenario:** The dots should be randomly scattered around the zero-line (dashed).
        """)

    model_data = results[best_model_name]
    x_vals, y_true = df['Complexity'].values, df['Volatility'].values
    names = [n.replace('_', ' ') for n in df['Derivative'].values]
    
    if 'params' not in model_data: return

    if best_model_name == 'Power Law': a, alpha = model_data['params']; y_pred = a * np.power(x_vals, alpha)
    elif best_model_name == 'Linear': m, c = model_data['params']; y_pred = m * x_vals + c
    elif best_model_name == 'Exponential': a, b = model_data['params']; y_pred = a * np.exp(b * x_vals)
    else: return

    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(x=x_vals, y=residuals, mode='markers', text=names, marker=dict(size=12, color=residuals, colorscale='RdBu_r', showscale=True, line=dict(width=1, color='black')), hovertemplate="<b>%{text}</b><br>Res: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=f'Residuals ({best_model_name})', xaxis_title='Complexity', yaxis_title='Residual', template='plotly_white', height=500)
    st.plotly_chart(fig, width="stretch")

def plot_scatter(df):
    with st.expander("How to read: Cluster Analysis", expanded=True):
        st.markdown("""
        **The Risk-Complexity Map:**
        **Trendline:** The dashed line shows the global relationship across ALL products.
        """)

    df_clean = df.copy()
    df_clean['Derivative'] = df_clean['Derivative'].str.replace('_', ' ')
    
    # 1. Main Scatter Plot
    fig = px.scatter(
        df_clean, 
        x='Complexity', 
        y='Volatility', 
        color='Derivative', 
        title='Cluster Analysis: Testing the Hypothesis',
        opacity=0.6 
    )
    
    # 2. Manual Trendline
    mask = df_clean['Complexity'].notnull() & df_clean['Volatility'].notnull()
    
    if mask.sum() > 2:
        x_data = df_clean.loc[mask, 'Complexity']
        y_data = df_clean.loc[mask, 'Volatility']
        
        try:
            m, b = np.polyfit(x_data, y_data, 1)
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            y_range = m * x_range + b
            
            fig.add_trace(go.Scatter(
                x=x_range, 
                y=y_range, 
                mode='lines', 
                name='Global Trend',
                line=dict(color='black', width=3, dash='dash')
            ))
        except Exception as e:
            st.warning(f"Could not calculate trendline: {e}")

    fig.update_layout(template='plotly_white', height=600)
    st.plotly_chart(fig, width="stretch")

def plot_complexity_distribution(complexity_df):
    with st.expander("How to read: Complexity Ranking", expanded=False):
        st.markdown("""
        **The "C" Variable:**
        This chart shows the calculated **Complexity Score ($C$)** for each product on a scale from 0 to 10.
        """)
            
    # 1. Kopie erstellen und sortieren
    df_plot = complexity_df.sort_values('Complexity_Score').copy()
        
    # 2. Index in eine echte Spalte umwandeln ("Derivative")
    df_plot = df_plot.reset_index() 
    # Die erste Spalte ist der alte Index, wir nennen sie sauber um
    df_plot.rename(columns={df_plot.columns[0]: 'Derivative'}, inplace=True)
        
    # 3. Unterstriche im Namen ersetzen
    df_plot['Derivative'] = df_plot['Derivative'].str.replace('_', ' ')
        
    fig = px.bar(
        df_plot, 
        x='Complexity_Score', 
        y='Derivative', 
        orientation='h', 
        color='Complexity_Score', 
        color_continuous_scale='RdYlGn_r', 
        title='Complexity Ranking (The "C" Variable)',
        # HIER IST DER FIX: Wir mappen den internen Namen auf den schönen Namen
        labels={'Complexity_Score': 'Complexity Score'}
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, width="stretch")

def plot_model_fits(df, results):
    with st.expander("How to read: Curve Fitting", expanded=False):
        st.markdown("""
        **Testing the Hypothesis:**
        We try to draw a line through the data points to see which mathematical law describes the market best.
        """)

    fig = go.Figure()
    
    # 1. Real Data
    fig.add_trace(go.Scatter(
        x=df['Complexity'], 
        y=df['Volatility'], 
        mode='markers', 
        name='Data', 
        marker=dict(size=12, color='rgba(0,0,0,0.6)', line=dict(width=1, color='DarkSlateGrey'))
    ))
    
    # 2. Smooth Curves
    x_min, x_max = df['Complexity'].min(), df['Complexity'].max()
    x_smooth = np.linspace(x_min, x_max, 100)
    
    colors = {'Exponential': '#E74C3C', 'Linear': '#3498DB', 'Power Law': '#2ECC71'}
    line_styles = {'Exponential': 'dot', 'Linear': 'dash', 'Power Law': 'solid'}
    
    for name, data in results.items():
        if 'error' not in data and 'params' in data:
            if name == 'Power Law': 
                y_fit = data['params'][0] * np.power(x_smooth, data['params'][1])
            elif name == 'Linear': 
                y_fit = data['params'][0] * x_smooth + data['params'][1]
            elif name == 'Exponential': 
                y_fit = data['params'][0] * np.exp(data['params'][1] * x_smooth)
            else: 
                continue
            
            fig.add_trace(go.Scatter(
                x=x_smooth, 
                y=y_fit, 
                mode='lines', 
                name=f"{name} (R²={data['R2']:.3f})", 
                line=dict(color=colors.get(name, 'black'), width=3, dash=line_styles.get(name, 'solid'))
            ))
            
    fig.update_layout(
        title='Model Fit: Geometry of Risk (Smooth Curves)', 
        xaxis_title='Complexity (C)', 
        yaxis_title='Volatility (σ)', 
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    st.plotly_chart(fig, width="stretch")

def plot_complexity_breakdown_chart(classifier):
    st.subheader("Contribution of Components")
    st.caption("Which factors contribute most to the complexity score?")
    
    derivatives = list(classifier.complexity_scores.keys())
    data = []
    for d in derivatives:
        metrics = classifier.complexity_scores[d]
        row = {'Derivative': d.replace('_', ' ')}
        for comp, weight in classifier.weights.items():
            if comp in METRIC_CONFIG:
                row[METRIC_CONFIG[comp]['label']] = metrics[comp] * weight
        row['Total'] = classifier.calculate_complexity_score(d)
        data.append(row)
    
    df = pd.DataFrame(data).sort_values('Total')
    fig = go.Figure()
    for label in [c['label'] for c in METRIC_CONFIG.values()]:
        if label in df.columns:
            fig.add_trace(go.Bar(name=label, y=df['Derivative'], x=df[label], orientation='h'))
            
    fig.update_layout(barmode='relative', title='Score Components Breakdown', height=600, template='plotly_white')
    
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Total'], 
            y=row['Derivative'], 
            text=f"<b>{row['Total']:.1f}</b>", 
            showarrow=False, 
            xshift=20, 
            font=dict(color="black", size=12) 
        )
    st.plotly_chart(fig, width="stretch")

def display_model_metrics_comparison(results):
    st.subheader("Statistical Model Comparison")
    metrics_data = []
    for name, res in results.items():
        if 'error' not in res:
            metrics_data.append({'Model': name, 'R²': res.get('R2',0), 'RMSE': res.get('RMSE',0), 'AIC': res.get('AIC',np.inf)})
    
    df_metrics = pd.DataFrame(metrics_data).sort_values('AIC')
    
    st.dataframe(
        df_metrics, 
        column_config={
            "R²": st.column_config.ProgressColumn("R² (Fit)", min_value=0, max_value=1, format="%.3f"),
            "RMSE": st.column_config.NumberColumn("RMSE (Error)", format="%.4f"),
            "AIC": st.column_config.NumberColumn("AIC (Quality)", format="%.1f")
        },
        width="stretch", 
        hide_index=True
    )

    with st.expander("Understanding the Metrics (Calculations & Meaning)", expanded=True):
        st.markdown("Here is how we measure which model is the 'winner':")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### 1. R² (R-Squared)")
            st.caption("**Coefficient of Determination**")
            st.info("How much of the chaos is explained by the formula?")
            st.markdown("**Formula:**")
            st.latex(r"R^2 = 1 - \frac{SS_{residual}}{SS_{total}}")

        with c2:
            st.markdown("#### 2. RMSE")
            st.caption("**Root Mean Square Error**")
            st.info("How wrong is the model on average?")
            st.markdown("**Formula:**")
            st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")

        with c3:
            st.markdown("#### 3. AIC")
            st.caption("**Akaike Information Criterion**")
            st.info("Does the model cheat by being too complex?")
            st.markdown("**Formula:**")
            st.latex(r"AIC = 2k - 2\ln(\hat{L})")

def plot_complexity_methodology(classifier):
    st.subheader("Methodology: Scientific Weighting")
    st.markdown("""
    The **Complexity Score ($C$)** is not arbitrary. It is a weighted sum of five distinct risk factors derived from financial engineering theory.
    """)
    
    for key, conf in METRIC_CONFIG.items():
        current_weight = classifier.weights.get(key, conf.get('weight', 0))
        pct = current_weight * 100
        
        if current_weight > 0.2: icon = "" 
        elif current_weight > 0: icon = "" 
        else: icon = "" 
        
        with st.expander(f"{icon} {conf['label']} (Weight: {pct:.0f}%)", expanded=False):
            st.markdown(f"**Definition:** {conf['desc']}")
            st.markdown("---")
            st.markdown(conf['rationale'], unsafe_allow_html=True)
            
    st.info("The sum of these weighted factors results in the final Complexity Score (0-10).")

def create_summary_dashboard(results):
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", results.get('best_model','N/A'), delta=f"R²={results.get('best_r2',0):.3f}")
    c2.metric("Correlation", f"{results.get('pearson_corr',0):.3f}")
    c3.metric("Hypothesis", "Confirmed" if results.get('best_r2',0)>0.65 else "Partial")

def plot_r2_explanation(y_true, y_pred):
    with st.expander("How to read: R² Visualized", expanded=False):
        st.markdown("""
        **The Goodness of Fit:**
        This plot helps visualize *why* the R² score is what it is.
        
        * **Left (Data):** The actual volatility values (Chaos).
        * **Middle (Model):** The smooth curve our model predicts (Order).
        * **Right (R²):** A score of 1.0 means the 'Order' perfectly explains the 'Chaos'.
        """)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(range(len(y_true)), y_true, color='grey', alpha=0.6); axes[0].set_title("Actual Data (Chaos)")
    axes[1].plot(y_pred, color='#2E86C1', linewidth=2); axes[1].set_title("Model Prediction (Order)")
    axes[2].text(0.5, 0.5, "R² Calculation", ha='center', fontsize=12); axes[2].axis('off')
    st.pyplot(fig)
