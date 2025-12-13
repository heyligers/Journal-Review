import streamlit as st
import pandas as pd
import numpy as np
import io
import qrcode
import inspect  # <--- NEU: Für Code-Introspektion
import matplotlib.pyplot as plt
from visualization import plot_r2_explanation, METRIC_CONFIG, plot_gamma_explanation
from fermat_connection import add_fermat_section
from utils import safe_style_format

# --- PRESENTATION STYLING ---
def apply_presentation_style():
    """
    Optimizes the UI for presentations:
    - Removes excess whitespace at the top.
    - Hides the Streamlit footer/hamburger menu.
    - Increases metric font size for readability.
    """
    st.markdown("""
        <style>
        /* Reduces top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Hides standard footer and menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Increases metric font size */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
        }
        /* Style for scenario buttons */
        div.stButton > button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        </style>
    """, unsafe_allow_html=True)

def render_tab_footer():
    st.markdown("---")
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px;">
            <span style="color: grey; font-size: 0.8em;">Gold Derivatives Analysis (Scientific Edition v2.8)</span>
            <a href="#top" target="_self" style="text-decoration: none; border: 1px solid #ddd; padding: 5px 10px; border-radius: 5px;">Back to top</a>
        </div>
        <br>
    """, unsafe_allow_html=True)

# --- NEU: CODE SNIPPET VISUALIZER ---
def display_code_snippet(func_obj, title="Show Source Code"):
    """
    Zeigt den Quellcode einer Funktion oder Klasse in einem Expander an.
    """
    with st.expander(title):
        try:
            code = inspect.getsource(func_obj)
            st.code(code, language='python')
        except OSError:
            st.warning("Source code not available (OS Error).")
        except TypeError:
            st.warning(f"Could not retrieve source for {type(func_obj)}. Ensure it is a function, class, or module.")

def setup_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.header("Configuration")
    
    # --- SCENARIO MANAGER ---
    st.sidebar.subheader("Quick Scenarios")
    
    # Grid layout for buttons
    row1_1, row1_2 = st.sidebar.columns(2)
    row2_1, row2_2 = st.sidebar.columns(2)
    row3_1, row3_2 = st.sidebar.columns(2)
    
    # 1. COVID-19 (Extreme Volatility Spike)
    if row1_1.button("COVID-19"):
        st.session_state['start_date'] = pd.to_datetime('2019-06-01').date()
        st.session_state['end_date'] = pd.to_datetime('2020-12-31').date()
        st.rerun()

    # 2. 2008 Financial Crisis (Liquidity Crisis)
    if row1_2.button("GFC 2008"):
        st.session_state['start_date'] = pd.to_datetime('2008-01-01').date()
        st.session_state['end_date'] = pd.to_datetime('2009-12-31').date()
        st.rerun()

    # 3. 2013 Crash (Bear Market / Taper Tantrum)
    if row2_1.button("2013 Crash"):
        st.session_state['start_date'] = pd.to_datetime('2013-01-01').date()
        st.session_state['end_date'] = pd.to_datetime('2013-12-31').date()
        st.rerun()

    # 4. 2022 War & Inflation (Rate Hikes)
    if row2_2.button("2022 War"):
        st.session_state['start_date'] = pd.to_datetime('2022-01-01').date()
        st.session_state['end_date'] = pd.to_datetime('2023-01-01').date()
        st.rerun()

    # 5. All Time (Long Term Validation)
    if st.sidebar.button("All Time / Reset"):
        st.session_state['start_date'] = pd.to_datetime('2000-01-01').date()
        st.session_state['end_date'] = pd.to_datetime('today').date()
        st.rerun()

    display_qr_code("https://journal-review.streamlit.app/")
    
    # --- TIME PERIOD ---
    st.sidebar.subheader("Time Period")
    min_date, max_date = pd.to_datetime('2000-01-01').date(), pd.to_datetime('today').date()
    
    # Initialize Session State if not present
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = pd.to_datetime('2013-01-01').date()
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = max_date

    # Date Inputs bound to Session State
    start_date = st.sidebar.date_input("Start", min_value=min_date, max_value=max_date, key='start_date')
    end_date = st.sidebar.date_input("End", min_value=min_date, max_value=max_date, key='end_date')
    
    if start_date >= end_date: 
        st.sidebar.error("Start Date must be before End Date!")
        st.stop()
    
    # --- DERIVATIVES ---
    st.sidebar.subheader("Derivative Selection")
    selected = []
    
    if st.sidebar.checkbox("Gold Spot (Benchmark)", True): selected.append('Gold_Spot')
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Physical & Linear**")
    if st.sidebar.checkbox("Futures (GC=F)", True): selected.append('Gold_Futures')
    if st.sidebar.checkbox("ETF (GLD)", True): selected.append('Gold_ETF_GLD')
    if st.sidebar.checkbox("Trust (PHYS)", True): selected.append('Sprott_Physical_Trust')
    
    st.sidebar.markdown("**Structured (Real)**")
    if st.sidebar.checkbox("Futures ETF (DGL)", True): selected.append('Gold_Futures_ETF_DGL')
    if st.sidebar.checkbox("2x ETF (UGL)", True): selected.append('Gold_Leveraged_ETF_2x')
    if st.sidebar.checkbox("Inv. -2x (GLL)", False): selected.append('Inverse_Gold_ETF_GLL')
    if st.sidebar.checkbox("Cov. Call ETN (GLDI)", True): selected.append('Gold_Covered_Call_GLDI')
    
    st.sidebar.markdown("**Advanced (Scientific Sim)**")
    if st.sidebar.checkbox("Options (BSM)", True): selected.append('Gold_Options')
    if st.sidebar.checkbox("Opt. on Futures (Blk-76)", True): selected.append('Gold_Futures_Option')
    if st.sidebar.checkbox("Barrier Opt (Path Dep)", True): selected.append('Gold_Barrier_Option')

    # --- WEIGHTS ---
    st.sidebar.markdown("---")
    custom_weights = {}
    with st.sidebar.expander("Adjust Weights"):
        defaults = {'structure_layers': 0.25, 'pricing_complexity': 0.25, 'leverage_factor': 0.20, 'counterparty_risk': 0.15, 'liquidity_score': -0.15}
        for k, v in defaults.items():
            if k in METRIC_CONFIG: custom_weights[k] = st.slider(METRIC_CONFIG[k]['label'], -0.5, 0.5, v, 0.05)
            
    rolling_window = st.sidebar.slider("Rolling Window", 10, 252, 30)
    correlations = {'Pearson': st.sidebar.checkbox("Pearson", True), 'Spearman': st.sidebar.checkbox("Spearman", True), 'Kendall': False}
    
    if not selected: st.error("Select at least one derivative!"); st.stop()
    
    return {
        'start_date': start_date, 
        'end_date': end_date, 
        'selected_derivatives': selected, 
        'rolling_window': rolling_window, 
        'correlations': correlations, 
        'custom_weights': custom_weights
    }

def display_header():
    st.title("Hypothesis Test: Power Law of Volatility")
    st.markdown("Analysis of Gold Derivatives: From Physical to Exotics (Real & Simulated)")
    
    # --- TAKEAWAY BOX ---
    st.info(r"""
    **Presentation Goal:** We investigate whether mathematical complexity ($C$) predicts market risk ($\sigma$).
    *Hypothesis:* The more complex the product, the more 'explosive' the volatility ($\sigma \approx C^\alpha$).
    """)

def display_methodology_section(selected_derivatives, rolling_window, derivative_data):
    """
    Enhanced methodology section with detailed formulas.
    """
    st.markdown("### Methodology & Pricing Models")
    
    with st.expander("Deep Dive: Option Pricing Logic (How we calculate)", expanded=True):
        
        tab_a, tab_b, tab_c, tab_d = st.tabs(["1. Inputs & Volatility", "2. Black-Scholes (Vanilla)", "3. Black-76 (Futures)", "3. Barrier Options (Exotic)"])
        
        with tab_a:
            st.markdown("#### Dynamic Input Parameters")
            st.markdown("""
            All option prices are recalculated **daily** based on the rolling market data.
            
            * **Underlying Price ($S_t$):** Daily Close of Gold Futures (GC=F).
            * **Risk-Free Rate ($r$):** 13-Week Treasury Bill Rate (^IRX) or 4% fallback.
            * **Time to Maturity ($T$):** Constant rolling maturity of 30 days ($T = 30/365$).
            """)
            st.info("""
            **Volatility Input ($\sigma$):** We do NOT use the VIX. We calculate the **Realized Volatility** of the underlying asset over the last 30 days:
            """)
            st.latex(r"\sigma_{hist} = \sqrt{252} \cdot \text{StdDev}(\ln(\frac{S_t}{S_{t-1}}))_{t-30 \dots t}")

        with tab_b:
            st.markdown("#### Standard Vanilla Call Option")
            st.markdown("We model a **Rolling ATM (At-The-Money) Call Strategy**. Every day, we simulate buying a fresh option with Strike $K = S_{current}$.")
            
            st.markdown("**Black-Scholes-Merton Formula:**")
            st.latex(r"C(S, t) = S_t \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")
            
            st.markdown("Where $d_1$ and $d_2$ capture the probabilistic moneyness:")
            st.latex(r"d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} \quad\text{and}\quad d_2 = d_1 - \sigma\sqrt{T}")
            
            st.markdown("This model assumes a continuous geometric Brownian motion and ignores jump risks (unlike the Barrier Option).")

        with tab_c:
            st.markdown("#### Option on Futures (Black-76)")
            st.info("""
            **The Derivative Squared:**
            This represents an option where the underlying asset is NOT physical gold, but a **Gold Futures Contract**.
            Mathematically, this changes the drift assumption because Futures are already 'forward looking'.
            """)
                
            st.markdown("**Black-76 Formula (Fischer Black, 1976):**")
            st.latex(r"C(F, t) = e^{-rT} [ F_t \cdot N(d_1) - K \cdot N(d_2) ]")
                
            st.markdown("Key Difference in $d_1$ (No rate term):")
            st.latex(r"d_1 = \frac{\ln(F/K) + (\frac{\sigma^2}{2})T}{\sigma \sqrt{T}}")
                
            st.success("""
            **Scientific Relevance:**
            This model isolates the 'Pure Volatility' component better than Black-Scholes because it removes the interest rate drift from the asset path.
            """)

        with tab_d:
            st.markdown("#### Down-and-Out Barrier Option")
            st.markdown("""
            This is a **path-dependent** exotic option.
            
            * **Mechanism:** The option behaves like a Vanilla Call, UNLESS the price hits a lower barrier.
            * **Knock-Out Event:** If $Price_{Low} \le Barrier$, the option becomes worthless (or drops to a minimal rebate).
            """)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Barrier Level", "95%", "of previous Close")
            with c2:
                st.metric("Risk Profile", "Gap Risk", "Discontinuous Payoff")
            
            st.warning("""
            **Simulation Logic:**
            1. We calculate the theoretical Vanilla Price.
            2. We check if `Day_Low <= Barrier`.
            3. If **True**: Price is set to effectively 0 (Knock-Out).
            4. This creates extreme volatility spikes when the barrier is touched.
            """)

    st.markdown("---")
    # Gamma Explanation Visualization
    plot_gamma_explanation()

def display_r2_explanation(y_true, y_pred, model_name, ss_res, ss_tot, r2_calculated):
    with st.expander("R² Explanation"):
        c1, c2, c3 = st.columns(3)
        c1.metric("SS_res", f"{ss_res:.4f}"); c2.metric("SS_tot", f"{ss_tot:.4f}"); c3.metric("R²", f"{r2_calculated:.4f}")
        plot_r2_explanation(y_true, y_pred)

def display_detailed_model_equations(results):
    st.subheader("Model Equations & Deep Dive")
        
    # Erweiterte Erklärung
    with st.expander("How to interpret these parameters?", expanded=True):
        st.markdown(r"""
        **1. The Mathematical Foundation**
        We fit the data using **Non-Linear Least Squares** optimization (via `scipy.optimize.curve_fit`). 
        The goal is to find the parameters $a$ and $\alpha$ that minimize the error between the theoretical curve and actual market data.
            
        **2. Parameter $a$ (The Base Factor)**
        * **Definition:** This is the theoretical volatility of an asset with **Complexity = 1** (Pure Physical Gold).
        * **Economic Meaning:** It represents the **intrinsic systemic risk** of the gold market itself, stripped of any structural engineering.
        * **Context:** A higher $a$ means the entire asset class is volatile, regardless of how the product is structured.

        **3. Parameter $\alpha$ (The Exponent / Alpha)**
        * **Definition:** This is the **elasticity of risk**. It dictates how "fast" risk reacts to added complexity.
        * **The Critical Thresholds:**
            * $\alpha = 1.0$: **Linear Growth.** Risk increases proportionally to complexity.
            * $\alpha > 1.0$: **Convex (Fragile).** Risk grows *exponentially*. A small increase in complexity leads to a disproportionate explosion in volatility (e.g., 2008 Crisis structures).
            * $\alpha < 1.0$: **Concave (Diminishing Returns).** Risk grows, but at a slower rate. This implies that some structural layers might actually *dampen* volatility (e.g., diversification effects within an ETF).
        """)

    st.markdown("---")
    st.markdown("#### Calculated Models")

    for m, d in results.items():
        if 'error' not in d:
            # Container für jedes Modell für saubere Trennung
            with st.container():
                c1, c2 = st.columns([1, 3])
                    
                with c1:
                    st.markdown(f"**{m}**")
                    st.caption(f"R² = {d.get('R2', 0):.4f}")
                    if 'RMSE' in d: st.caption(f"RMSE = {d['RMSE']:.4f}")
                    
                with c2:
                    if 'params' in d:
                        if m == 'Power Law': 
                            # Latex Formel groß
                            st.latex(r"\sigma(C) = " + f"{d['params'][0]:.4f}" + r" \cdot C^{" + f"{d['params'][1]:.4f}" + r"}")
                                
                            # Spezifische Interpretation für das Power Law
                            alpha_val = d['params'][1]
                            if alpha_val > 1.05:
                                interpret = "**Exponential Risk:** Complexity amplifies volatility disproportionately."
                            elif alpha_val < 0.95:
                                interpret = "**Dampening Effect:** Risk grows slower than complexity (Logarithmic tendency)."
                            else:
                                interpret = "**Linear Scaling:** Risk is directly proportional to complexity."
                            st.info(interpret)
                                
                        elif m == 'Exponential': 
                            st.latex(r"\sigma(C) = " + f"{d['params'][0]:.4f}" + r" \cdot e^{" + f"{d['params'][1]:.4f}" + r" \cdot C}")
                        elif m == 'Linear': 
                            sign = "+" if d['params'][1] >= 0 else "-"
                            st.latex(r"\sigma(C) = " + f"{d['params'][0]:.4f}" + r" \cdot C " + sign + f" {abs(d['params'][1]):.4f}")
            st.divider()

def display_correlation_interpretation(pearson_corr, pearson_p):
    """
    Displays correlation with clear visual significance indicator.
    """
    # Styling for significance
    if pearson_p < 0.05:
        box_color = "#d4edda" # Green
        border_color = "#c3e6cb"
        text_color = "#155724"
        sig_text = "Significant (p < 0.05)"
        icon = ""
    else:
        box_color = "#f8d7da" # Red
        border_color = "#f5c6cb"
        text_color = "#721c24"
        sig_text = f"Not Significant (p={pearson_p:.3f})"
        icon = ""

    st.markdown(f"""
    <div style="background-color: {box_color}; padding: 10px; border-radius: 5px; border: 1px solid {border_color}; color: {text_color}; text-align: center; margin-bottom: 10px;">
        <h3 style="margin:0; color: {text_color};">{icon} r = {pearson_corr:.3f}</h3>
        <p style="margin:0; font-size: 0.9em;">{sig_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation text
    if abs(pearson_corr) > 0.7: strength = "Strong"
    elif abs(pearson_corr) > 0.3: strength = "Moderate"
    else: strength = "Weak"
    
    st.caption(f"Interpretation: {strength} correlation between Complexity and Volatility.")

def display_hypothesis_conclusion(best_model, best_r2, pearson_p, pearson_corr, results):
    st.header("Hypothesis Test Result")
        
    # Kriterien
    is_power_law_best = (best_model == 'Power Law')
    is_strong_fit = (best_r2 > 0.65)
    is_significant = (pearson_p < 0.05)
        
    # Alpha IMMER auslesen, solange Power Law berechnet wurde
    alpha_val = 0
    has_power_law_results = 'Power Law' in results and 'params' in results['Power Law']
        
    if has_power_law_results:
        alpha_val = results['Power Law']['params'][1]
        power_law_r2 = results['Power Law'].get('R2', 0)

    # --- TEIL 1: Beste Passform (Statistischer Gewinner) ---
    st.subheader("1. Structural Validity (The 'Fermat' Link)")
    c1, c2 = st.columns(2)
        
    # Zeige das statistisch beste Modell
    c1.metric("Best Fitting Model", best_model, delta="Statistical Winner")
    c2.metric("Goodness of Fit (R²)", f"{best_r2:.4f}", delta="Strong" if is_strong_fit else "Weak")
        
    if is_power_law_best and is_strong_fit:
        st.success("**Primary Hypothesis Confirmed:** Market volatility strictly follows a Power Law structure.")
    elif has_power_law_results:
        # Falls Power Law nicht der Gewinner ist, aber gut passt
        if power_law_r2 > 0.65:
            st.info(f"**Hypothesis Plausible:** While '{best_model}' fits slightly better, the Power Law is also a strong candidate (R²={power_law_r2:.4f}).")
        else:
            st.warning(f"**Hypothesis Challenged:** The Power Law shows a weaker fit (R²={power_law_r2:.4f}) compared to {best_model}.")

    st.markdown("---")
        
    # --- TEIL 2: Parameter Analyse (FOKUS AUF POWER LAW) ---
    st.subheader("2. Parameter Interpretation (Hypothesis Focus)")
        
    if not has_power_law_results:
        st.error("Power Law model could not be calculated.")
    else:
        # Wir erzwingen die Analyse des Alphas, egal wer "gewonnen" hat
        st.caption("Analyzing the **Power Law Exponent** (regardless of best fit) to understand the nature of risk scaling.")
            
        col_a, col_b = st.columns([1, 2])
        col_a.metric("Calculated Alpha (α)", f"{alpha_val:.4f}")
            
        # Interpretation
        if alpha_val > 1.05:
            col_b.error("**Explosive Regime (α > 1):** Risk grows faster than complexity. System is structurally fragile.")
        elif 0.95 <= alpha_val <= 1.05:
            col_b.warning("**Linear Regime (α ≈ 1):** Risk is proportional to complexity. Balanced market.")
        else: # alpha < 0.95
            col_b.success("**Dampened Regime (α < 1):** Risk grows slower than complexity. Structural hedging works.")
                
        st.caption(f"""
        **Scientific Insight:** We found $\\alpha \\approx {alpha_val:.2f}$. This indicates that while complexity *does* increase risk, 
        financial engineering (like ETFs) successfully mitigates some of the 'explosive' potential found in pure number theory. 
        Unlike Fermat's equations where $n>2$ creates divergence, the market maintains $\\alpha \\approx 1$ (or slightly below) for stability.
        """)

def display_downloads(combined_df, derivative_data, start_date, end_date, best_model, best_r2, pearson_corr):
    st.subheader("Export")
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", combined_df.to_csv(index=False), "analysis.csv", "text/csv")
    summary = f"Analysis {start_date}-{end_date}\nR²: {best_r2:.4f}"
    c2.download_button("Download Summary", summary, "summary.txt", "text/plain")

def display_qr_code(url):
    st.sidebar.markdown("---")
    qr = qrcode.QRCode(box_size=5); qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO(); img.save(buf, format='PNG')
    st.sidebar.image(buf, caption="Scan Me")

def display_limitations_section():
    st.header("Critical Review: Model Limitations")
        
    # KORREKTUR: r""" verwenden für korrekte LaTeX Darstellung
    st.markdown(r"""
    Every financial model is a simplification of reality. Here we discuss the known constraints 
    and blind spots of our Complexity-Volatility Hypothesis ($\sigma \approx C^\alpha$).
    """)

    # --- 1. THE INPUT PROBLEM (Subjectivity) ---
    with st.expander("1. Input Bias: The Subjectivity of 'C'", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.warning("**The Problem**")
            st.markdown("The Complexity Score ($C$) is static and heuristic.")
        with c2:
            st.markdown(r"""
            * **Static Nature:** We assume `Liquidity = 5` for Gold Spot forever. In reality, liquidity can vanish instantly during a crash (e.g., March 2020), theoretically spiking the complexity score dynamically. Our model keeps $C$ constant.
            * **Weighting Arbitrage:** The weights (25% Structure, 25% Pricing...) are derived from theory, not regression. Changing the weights changes the slope of our curve.
            """)

    # --- 2. THE OUTPUT PROBLEM (StdDev vs. Tail Risk) ---
    with st.expander("2. Measurement Bias: Standard Deviation vs. Tail Risk", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.error("**The Problem**")
            st.markdown(r"Volatility ($\sigma$) ignores 'Black Swans'.")
        with c2:
            st.markdown("""
            * **Blind Spot:** We measure risk as *Standard Deviation* (how much prices wiggle). 
            * **The Barrier Trap:** A Barrier Option often has **low volatility** (it moves with the market) until it hits the barrier and drops -100% instantly. Standard Deviation cannot capture this "Jump Risk" adequately. It underestimates the risk of exotic products.
            * **Solution:** A better metric would be **VaR (Value at Risk)** or **Expected Shortfall**.
            """)
                
    # --- 3. THE DATA PROBLEM (Simulation vs. Reality) ---
    with st.expander("3. Data Heterogeneity: Mixing Real & Simulated Data", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("**The Problem**")
            st.markdown("We compare 'dirty' real data with 'clean' math.")
        with c2:
            st.markdown("""
            * **Apples & Oranges:** * **GLD (ETF):** Real market prices containing noise, fees, and market maker spreads.
                * **Barrier Option:** Pure mathematical simulation (Black-Scholes). It follows a perfect theoretical path.
            * **Consequence:** The simulated curves are "smoother" than real market data, which creates an artificial fit in the regression.
            """)

    st.markdown("---")
    st.caption("Conclusion: While the Power Law hypothesis holds for general structural risk, the model likely underestimates the specific 'Tail Risk' of exotic derivatives due to the use of standard deviation as the sole risk metric.")
