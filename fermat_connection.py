import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

def create_fermat_animation():
    """
    Creates a visualization showing the connection between 
    Fermat's Last Theorem and the Derivative Volatility Hypothesis.
    """
    st.title("From Fermat's Last Theorem to Derivative Volatility")

    st.markdown("""
    This section demonstrates the conceptual link between the mathematical elegance 
    of Fermat's Last Theorem and our hypothesis on volatility scaling.

    **Core Analogy:** Fermat's theorem deals with power functions ($a^n$). Our hypothesis 
    states that volatility grows as a power function of complexity: $\sigma(C) = a \cdot C^\\alpha$
    """)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Historical Context",
        "Mathematical Analogy",
        "The Connection"
    ])

    with tab1:
        show_historical_context()
    with tab2:
        show_mathematical_analogy()
    with tab3:
        show_connection_explanation()


def show_historical_context():
    """
    Shows the historical context of Fermat's Last Theorem.
    """
    st.header("Fermat's Last Theorem: A Historical Overview")

    # Timeline Data
    events = [
        {"Year": 1637, "Event": "The Conjecture", "Description": "Fermat writes his famous note in the margin."},
        {"Year": 1753, "Event": "Euler (n=3)", "Description": "Leonhard Euler proves the case for n=3."},
        {"Year": 1825, "Event": "Dirichlet & Legendre", "Description": "Proof for n=5."},
        {"Year": 1984, "Event": "Frey Curve", "Description": "Gerhard Frey links Fermat to elliptic curves."},
        {"Year": 1995, "Event": "The Proof", "Description": "Andrew Wiles publishes the complete proof."}
    ]
    
    df_events = pd.DataFrame(events)
    
    # Plotly Timeline
    fig = go.Figure()
    
    # Draw Line
    fig.add_trace(go.Scatter(
        x=df_events['Year'],
        y=[1] * len(df_events),
        mode='lines+markers',
        marker=dict(size=15, color='#31333F'),
        line=dict(color='gray', width=2),
        text=df_events['Event'],
        hovertemplate='<b>%{text}</b><br>Year: %{x}<br><extra></extra>'
    ))

    # Add Annotations
    for i, row in df_events.iterrows():
        fig.add_annotation(
            x=row['Year'],
            y=1,
            text=f"<b>{row['Year']}</b><br>{row['Event']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=0,
            ay=-40 if i % 2 == 0 else 40
        )

    fig.update_layout(
        title="Historical Milestones",
        showlegend=False,
        height=300,
        yaxis=dict(visible=False, range=[0.5, 1.5]),
        xaxis=dict(title="Year"),
        template="plotly_white",
        margin=dict(t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Historical Significance
        
        Pierre de Fermat postulated in 1637 that the equation $a^n + b^n = c^n$ 
        has no integer solutions for $n > 2$.
        
        It took **358 years** for this theorem to be proven by Andrew Wiles. 
        This illustrates how deeply power law structures are embedded in mathematics.
        """)

    with col2:
        st.info("""
        **Key Concept:**
        Fermat's Theorem shows that 
        certain mathematical relationships 
        are strictly **non-linear** and 
        governed by **power laws**.
        """)


def show_mathematical_analogy():
    """
    Shows the mathematical analogy between Fermat and Derivatives.
    """
    st.header("Mathematical Analogy")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Fermat's Last Theorem")
        st.latex(r''' a^n + b^n = c^n ''')

        st.markdown("""
        **Properties:**
        - **Power Function** $a^n$
        - Non-linear for $n > 2$
        - Growth is disproportionate
        - The **Exponent n** defines the growth rate
        """)

        # Fermat Plot
        x = np.linspace(1, 5, 100)
        fig1 = go.Figure()
        colors_fermat = ['#440154', '#31688E', '#35B779', '#FDE724', '#FF6B6B']

        for i, n in enumerate([1, 2, 3, 4, 5]):
            fig1.add_trace(go.Scatter(
                x=x, y=x**n, mode='lines', name=f'n={n}',
                line=dict(width=3, color=colors_fermat[i])
            ))

        fig1.update_layout(
            title="Fermat: Power Functions a^n",
            xaxis_title="Base (a)", yaxis_title="Value (a^n)",
            height=350, margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### Derivative Volatility Hypothesis")
        st.latex(r''' \sigma(C) = a \cdot C^{\alpha} ''')

        st.markdown("""
        **Properties:**
        - **Power Function** of Complexity
        - Non-linear
        - **IDENTICAL Structure** to Fermat!
        - The **Exponent alpha** is analogous to Fermat's n
        """)

        # Derivatives Plot
        C = np.linspace(1, 5, 100)
        fig2 = go.Figure()
        colors_deriv = ['#440154', '#31688E', '#35B779', '#FDE724', '#FF6B6B']

        for i, alpha in enumerate([0.5, 1.0, 1.5, 2.0, 2.5]):
            a = 0.1
            fig2.add_trace(go.Scatter(
                x=C, y=a * np.power(C, alpha), mode='lines', name=f'α={alpha:.1f}',
                line=dict(width=3, color=colors_deriv[i])
            ))

        fig2.update_layout(
            title="Derivatives: Volatility σ(C) = a·C^α",
            xaxis_title="Complexity (C)", yaxis_title="Volatility (σ)",
            height=350, margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.success("""
    ### The Central Analogy
    Both functions are **Power Functions** ($x^n$). We are testing whether financial markets 
    follow the same mathematical logic as Fermat's number theory.
    """)


def show_connection_explanation():
    """
    Explains the conceptual link.
    """
    st.header("The Connection: From Number Theory to Finance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Number Theory")
        st.info("""
        **Fermat (1637)**
        Investigates integer solutions for **Power Equations**.
        
        **Insight:**
        Exponents lead to explosive growth.
        """)

    with col2:
        st.markdown("#### Analogy")
        st.warning("""
        **Shared Principle**
        
        **Power Laws**
        Non-Linearity
        Exponent defines structure
        """)

    with col3:
        st.markdown("#### Finance")
        st.success("""
        **Derivatives (2025)**
        Investigates volatility as a **Power Function** of complexity.
        
        **Hypothesis:**
        Complexity drives risk according to the Power Model.
        """)

    # Transformation Visualisierung
    st.markdown("---")
    st.subheader("The Mathematical Journey")

    c1, c2, c3 = st.columns(3)
    c1.metric("Inspiration", "a^n", "Fermat")
    c2.write("## -> Transfer ->")
    c3.metric("Application", "C^alpha", "Financial Market")


def add_fermat_section():
    """
    Adds the Fermat section to the main app.
    """
    st.markdown("---")
    create_fermat_animation()
    st.markdown("---")