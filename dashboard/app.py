"""
dashboard/app.py — Streamlit Dashboard for TechSight.

Interactive visualization of technology trends, forecasts,
and trend scores using Plotly charts.

Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Import mock data (works without database)
from mock.mock_data import (
    get_technologies, get_yearly_indicators,
    get_trend_scores, get_forecasts,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechSight — Technology Trend Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #b8b5ff;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(100, 100, 255, 0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }

    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Trend score badge */
    .score-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .score-emerging {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }

    .score-growing {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
    }

    .score-stable {
        background: linear-gradient(135deg, #f5af19, #f12711);
        color: white;
    }

    .score-declining {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0ff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(100, 100, 255, 0.3);
    }

    /* Info box */
    .info-box {
        background: rgba(100, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #c0c0ff;
        font-size: 0.9rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chart container */
    .chart-container {
        background: rgba(22, 33, 62, 0.5);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(100, 126, 234, 0.1);
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        color: #b8b5ff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data():
    techs = get_technologies()
    indicators = get_yearly_indicators()
    scores = get_trend_scores()
    forecasts = get_forecasts()
    return techs, indicators, scores, forecasts


techs, indicators, scores, forecasts = load_data()

# Color palette
COLORS = {
    "Solid State Batteries": "#00b09b",
    "Quantum Computing": "#667eea",
    "Generative Ai": "#f45c43",
    "Generative AI": "#f45c43",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🔭 TechSight</h1>
    <p>Cloud-Based Technology Trend Aggregation & Forecasting Dashboard</p>
    <p style="font-size: 0.8rem; color: #8892b0; margin-top: 0.3rem;">
        Analyzing Solid-State Batteries · Quantum Computing · Generative AI
        &nbsp;|&nbsp; Data: OpenAlex + USPTO &nbsp;|&nbsp; 2012–2024
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Filters")
    selected_domains = st.multiselect(
        "Technology Domains",
        options=indicators["technology_name"].unique().tolist(),
        default=indicators["technology_name"].unique().tolist(),
    )

    year_range = st.slider(
        "Year Range",
        min_value=int(indicators["year"].min()),
        max_value=int(indicators["year"].max()),
        value=(int(indicators["year"].min()), int(indicators["year"].max())),
    )

    st.markdown("---")
    st.markdown("### 📊 Data Sources")
    st.markdown("""
    - **OpenAlex** — Research papers
    - **USPTO ODP** — Patent filings
    """)

    st.markdown("---")
    st.markdown("### ⚙️ About")
    st.markdown("""
    **TechSight v1.0**

    5th Semester Mini Project

    *Built with Python, FastAPI,
    Streamlit, scikit-learn & Plotly*
    """)

# Filter data
filtered_indicators = indicators[
    (indicators["technology_name"].isin(selected_domains)) &
    (indicators["year"] >= year_range[0]) &
    (indicators["year"] <= year_range[1])
]

filtered_scores = scores[scores["technology_name"].isin(selected_domains)]

# ---------------------------------------------------------------------------
# KPI Metrics Row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

total_docs = int(techs["document_count"].sum())
total_patents = int(filtered_indicators["patent_count"].sum())
total_papers = int(filtered_indicators["paper_count"].sum())
avg_score = round(filtered_scores["score"].mean(), 1)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Documents</div>
        <div class="metric-value">{total_docs:,}</div>
        <div style="color: #8892b0; font-size: 0.75rem;">Patents + Research Papers</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Research Papers</div>
        <div class="metric-value">{total_papers:,}</div>
        <div style="color: #8892b0; font-size: 0.75rem;">From OpenAlex</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Patent Filings</div>
        <div class="metric-value">{total_patents:,}</div>
        <div style="color: #8892b0; font-size: 0.75rem;">From USPTO</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Trend Score</div>
        <div class="metric-value">{avg_score}</div>
        <div style="color: #8892b0; font-size: 0.75rem;">0–100 Scale</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trend Overview", "🏆 Trend Scores", "🔮 Forecasts", "📋 Data Explorer"
])

# ===== TAB 1: TREND OVERVIEW =====
with tab1:
    st.markdown('<div class="section-header">Publication Activity Over Time</div>',
                unsafe_allow_html=True)

    # Combined activity chart
    fig_activity = go.Figure()

    for tech_name in selected_domains:
        tech_data = filtered_indicators[
            filtered_indicators["technology_name"] == tech_name
        ].sort_values("year")

        color = COLORS.get(tech_name, "#667eea")

        fig_activity.add_trace(go.Scatter(
            x=tech_data["year"],
            y=tech_data["total_count"],
            name=tech_name,
            mode="lines+markers",
            line=dict(width=3, color=color),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{tech_name}</b><br>"
                "Year: %{x}<br>"
                "Total: %{y:,}<br>"
                "<extra></extra>"
            ),
        ))

    fig_activity.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b8b5ff"),
        xaxis=dict(
            title="Year",
            gridcolor="rgba(100,100,255,0.1)",
            dtick=1,
        ),
        yaxis=dict(
            title="Total Activity (Patents + Papers)",
            gridcolor="rgba(100,100,255,0.1)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_activity, use_container_width=True)

    # Patents vs Papers comparison
    st.markdown('<div class="section-header">Patents vs Research Papers</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_patents = go.Figure()
        for tech_name in selected_domains:
            tech_data = filtered_indicators[
                filtered_indicators["technology_name"] == tech_name
            ].sort_values("year")
            color = COLORS.get(tech_name, "#667eea")

            fig_patents.add_trace(go.Bar(
                x=tech_data["year"],
                y=tech_data["patent_count"],
                name=tech_name,
                marker_color=color,
                opacity=0.85,
            ))

        fig_patents.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Patent Filings by Year",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#b8b5ff"),
            barmode="group",
            xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
            yaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_patents, use_container_width=True)

    with col_right:
        fig_papers = go.Figure()
        for tech_name in selected_domains:
            tech_data = filtered_indicators[
                filtered_indicators["technology_name"] == tech_name
            ].sort_values("year")
            color = COLORS.get(tech_name, "#667eea")

            fig_papers.add_trace(go.Bar(
                x=tech_data["year"],
                y=tech_data["paper_count"],
                name=tech_name,
                marker_color=color,
                opacity=0.85,
            ))

        fig_papers.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Research Papers by Year",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#b8b5ff"),
            barmode="group",
            xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
            yaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
            showlegend=False,
        )
        st.plotly_chart(fig_papers, use_container_width=True)

    # Growth rates heatmap
    st.markdown('<div class="section-header">Year-over-Year Growth Rate (%)</div>',
                unsafe_allow_html=True)

    growth_data = []
    for tech_name in selected_domains:
        tech_data = filtered_indicators[
            filtered_indicators["technology_name"] == tech_name
        ].sort_values("year")
        counts = tech_data["total_count"].values.astype(float)
        years = tech_data["year"].values
        for i in range(1, len(counts)):
            prev = max(counts[i - 1], 1)
            growth = ((counts[i] - prev) / prev) * 100
            growth_data.append({
                "Technology": tech_name,
                "Year": years[i],
                "Growth (%)": round(growth, 1),
            })

    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        pivot = growth_df.pivot(index="Technology", columns="Year", values="Growth (%)")

        fig_heatmap = px.imshow(
            pivot.values,
            x=[str(y) for y in pivot.columns],
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(color="Growth %"),
        )
        fig_heatmap.update_layout(
            template=PLOTLY_TEMPLATE,
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#b8b5ff"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# ===== TAB 2: TREND SCORES =====
with tab2:
    st.markdown('<div class="section-header">Technology Trend Scores</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        The Trend Score (0–100) measures how strongly a technology is trending based on five
        weighted components: Growth Rate (35%), Consistency (25%), Acceleration (20%),
        Activity Volume (10%), and Citation Signal (10%).
    </div>
    """, unsafe_allow_html=True)

    # Score cards
    score_cols = st.columns(len(filtered_scores))
    for i, (_, row) in enumerate(filtered_scores.iterrows()):
        label_class = f"score-{row['label'].lower()}"
        with score_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 2rem;">
                <div class="metric-label">{row['technology_name']}</div>
                <div class="metric-value">{row['score']}</div>
                <span class="score-badge {label_class}">{row['label']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart for components
    st.markdown('<div class="section-header">Score Component Breakdown</div>',
                unsafe_allow_html=True)

    fig_radar = go.Figure()
    categories = ["Growth Rate", "Consistency", "Acceleration",
                   "Activity Volume", "Citation Signal"]

    for _, row in filtered_scores.iterrows():
        values = [
            row["growth_rate"], row["consistency"], row["acceleration"],
            row["activity_volume"], row["citation_signal"],
        ]
        values.append(values[0])  # close the polygon

        color = COLORS.get(row["technology_name"], "#667eea")
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row["technology_name"],
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color else f"{color}22",
        ))

    fig_radar.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b8b5ff"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(100,100,255,0.15)",
            ),
            angularaxis=dict(
                gridcolor="rgba(100,100,255,0.15)",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Component comparison bar chart
    st.markdown('<div class="section-header">Component Comparison</div>',
                unsafe_allow_html=True)

    comp_data = []
    for _, row in filtered_scores.iterrows():
        for comp in ["growth_rate", "consistency", "acceleration",
                      "activity_volume", "citation_signal"]:
            comp_data.append({
                "Technology": row["technology_name"],
                "Component": comp.replace("_", " ").title(),
                "Score": row[comp],
            })

    comp_df = pd.DataFrame(comp_data)
    fig_comp = px.bar(
        comp_df, x="Component", y="Score", color="Technology",
        barmode="group",
        color_discrete_map=COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig_comp.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b8b5ff"),
        xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
        yaxis=dict(gridcolor="rgba(100,100,255,0.1)", range=[0, 100]),
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ===== TAB 3: FORECASTS =====
with tab3:
    st.markdown('<div class="section-header">Activity Forecasts (2025–2027)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Forecasts are generated using Moving Average and Linear Regression models.
        Shaded regions show 80% confidence intervals. Dashed lines indicate predictions.
    </div>
    """, unsafe_allow_html=True)

    forecast_tech = st.selectbox(
        "Select Technology",
        options=selected_domains,
        key="forecast_tech",
    )

    for series_type in ["paper_count", "patent_count"]:
        series_label = "Research Papers" if series_type == "paper_count" else "Patent Filings"

        st.markdown(f'<div class="section-header">{series_label} Forecast</div>',
                    unsafe_allow_html=True)

        fig_forecast = go.Figure()
        color = COLORS.get(forecast_tech, "#667eea")

        # Historical data
        hist_data = filtered_indicators[
            filtered_indicators["technology_name"] == forecast_tech
        ].sort_values("year")

        fig_forecast.add_trace(go.Scatter(
            x=hist_data["year"],
            y=hist_data[series_type],
            name="Historical",
            mode="lines+markers",
            line=dict(width=3, color=color),
            marker=dict(size=8),
        ))

        # Forecast data
        tech_forecasts = forecasts[
            (forecasts["technology_name"] == forecast_tech) &
            (forecasts["series_type"] == series_type)
        ].sort_values("year")

        if not tech_forecasts.empty:
            # Connect historical to forecast
            last_hist_year = hist_data["year"].iloc[-1]
            last_hist_val = hist_data[series_type].iloc[-1]

            forecast_years = [last_hist_year] + tech_forecasts["year"].tolist()
            forecast_vals = [last_hist_val] + tech_forecasts["predicted_value"].tolist()
            lower_bounds = [last_hist_val] + tech_forecasts["lower_bound"].tolist()
            upper_bounds = [last_hist_val] + tech_forecasts["upper_bound"].tolist()

            # Upper bound
            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=upper_bounds,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Lower bound with fill
            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=lower_bounds,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"{color}22",
                showlegend=False,
                hoverinfo="skip",
            ))

            # Forecast line
            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=forecast_vals,
                name="Forecast",
                mode="lines+markers",
                line=dict(width=3, dash="dash", color=color),
                marker=dict(size=8, symbol="diamond"),
            ))

        fig_forecast.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#b8b5ff"),
            xaxis=dict(
                title="Year",
                gridcolor="rgba(100,100,255,0.1)",
                dtick=1,
            ),
            yaxis=dict(
                title=f"{series_label} Count",
                gridcolor="rgba(100,100,255,0.1)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )

        # Add vertical line at forecast boundary
        fig_forecast.add_vline(
            x=2024.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
            annotation_text="Forecast →", annotation_position="top right",
            annotation_font_color="#8892b0",
        )

        st.plotly_chart(fig_forecast, use_container_width=True)


# ===== TAB 4: DATA EXPLORER =====
with tab4:
    st.markdown('<div class="section-header">Explore Raw Data</div>',
                unsafe_allow_html=True)

    data_type = st.radio(
        "Select Data",
        ["Yearly Indicators", "Technologies", "Trend Scores", "Forecasts"],
        horizontal=True,
    )

    if data_type == "Yearly Indicators":
        st.dataframe(
            filtered_indicators,
            use_container_width=True,
            height=500,
        )
    elif data_type == "Technologies":
        st.dataframe(techs, use_container_width=True)
    elif data_type == "Trend Scores":
        st.dataframe(
            filtered_scores,
            use_container_width=True,
        )
    elif data_type == "Forecasts":
        st.dataframe(
            forecasts[forecasts["technology_name"].isin(selected_domains)],
            use_container_width=True,
            height=500,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5a5a8a; font-size: 0.8rem; padding: 1rem 0;">
    <strong>TechSight v1.0</strong> — Cloud-Based Technology Trend Aggregation & Forecasting Dashboard<br>
    5th Semester Mini Project | Data Sources: OpenAlex + USPTO Open Data Portal<br>
    Built with Python · FastAPI · Streamlit · scikit-learn · Plotly<br>
    ⚠️ This project uses mock data for demonstration purposes.
</div>
""", unsafe_allow_html=True)
