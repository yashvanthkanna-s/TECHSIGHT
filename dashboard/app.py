"""
dashboard/app.py — TechSight Intelligence Dashboard.

Clean, minimalist visualization platform for technology trend aggregation,
forecasting models, and multi-factor trend scores.

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

# Load core analytics and indicator data
from mock.mock_data import (
    get_technologies, get_yearly_indicators,
    get_trend_scores, get_forecasts,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechSight | Technology Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clean, Minimalist Enterprise CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Top Navigation / Brand Banner */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }

    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        font-weight: 400;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #34d399;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
    }

    /* Minimalist KPI Cards */
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        transition: border-color 0.15s ease;
    }

    .metric-card:hover {
        border-color: #334155;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }

    .metric-caption {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin: 1.5rem 0 0.85rem 0;
    }

    /* Score Badges */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }

    .score-emerging {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }

    .score-growing {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .score-stable {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .score-declining {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Minimal Callout Box */
    .callout-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 3px solid #38bdf8;
        padding: 0.9rem 1.25rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }

    /* Tab Minimalist Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 0.4rem 1.1rem;
        color: #94a3b8;
        font-size: 0.86rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(255, 255, 255, 0.03);
    }

    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
    }

    /* Clean dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load Cached Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data():
    techs = get_technologies()
    indicators = get_yearly_indicators()
    scores = get_trend_scores()
    forecasts = get_forecasts()
    return techs, indicators, scores, forecasts


techs, indicators, scores, forecasts = load_data()

# Clean, professional palette
COLORS = {
    "Solid State Batteries": "#10b981",  # emerald
    "Quantum Computing": "#38bdf8",      # sky
    "Generative Ai": "#818cf8",          # indigo
    "Generative AI": "#818cf8",          # indigo
}

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Header Component
# ---------------------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div>
        <h1 class="brand-title">TechSight</h1>
        <div class="brand-subtitle">Technology Trend Aggregation & Predictive Analytics System</div>
    </div>
    <div>
        <span class="status-pill">
            <span class="status-dot"></span>
            Pipeline Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Filter Parameters")
    selected_domains = st.multiselect(
        "Technology Domains",
        options=indicators["technology_name"].unique().tolist(),
        default=indicators["technology_name"].unique().tolist(),
    )

    year_range = st.slider(
        "Observation Window",
        min_value=int(indicators["year"].min()),
        max_value=int(indicators["year"].max()),
        value=(int(indicators["year"].min()), int(indicators["year"].max())),
    )

    st.markdown("---")
    st.markdown("#### Primary Data Sources")
    st.markdown("""
    - **OpenAlex** (Research Publications)
    - **USPTO** (Patent Filings Archive)
    """)

    st.markdown("---")
    st.markdown("#### Model Architecture")
    st.caption("""
    - NLP: TF-IDF & KMeans Clustering
    - Scoring: 5-Factor Composite Metric
    - Forecasting: Linear Regression & Moving Average
    """)

# Filter records
filtered_indicators = indicators[
    (indicators["technology_name"].isin(selected_domains)) &
    (indicators["year"] >= year_range[0]) &
    (indicators["year"] <= year_range[1])
]

filtered_scores = scores[scores["technology_name"].isin(selected_domains)]

# ---------------------------------------------------------------------------
# KPI Summary Row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_docs = int(techs["document_count"].sum())
total_patents = int(filtered_indicators["patent_count"].sum())
total_papers = int(filtered_indicators["paper_count"].sum())
avg_score = round(filtered_scores["score"].mean(), 1) if not filtered_scores.empty else 0.0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Volume</div>
        <div class="metric-value">{total_docs:,}</div>
        <div class="metric-caption">Synthesized corpus entries</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Research Papers</div>
        <div class="metric-value">{total_papers:,}</div>
        <div class="metric-caption">Peer-reviewed publications</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Patent Filings</div>
        <div class="metric-value">{total_patents:,}</div>
        <div class="metric-caption">IP applications indexed</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Composite Index</div>
        <div class="metric-value">{avg_score}</div>
        <div class="metric-caption">Mean trend score (0–100)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Analytical Views (Tabs)
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Overview", "Trend Scores", "Forecast Models", "Data Explorer"
])

# ===== TAB 1: TREND OVERVIEW =====
with tab1:
    st.markdown('<div class="section-title">Aggregate Publication Velocity</div>', unsafe_allow_html=True)

    fig_activity = go.Figure()

    for tech_name in selected_domains:
        tech_data = filtered_indicators[
            filtered_indicators["technology_name"] == tech_name
        ].sort_values("year")

        color = COLORS.get(tech_name, "#38bdf8")

        fig_activity.add_trace(go.Scatter(
            x=tech_data["year"],
            y=tech_data["total_count"],
            name=tech_name,
            mode="lines+markers",
            line=dict(width=2.5, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=(
                f"<b>{tech_name}</b><br>"
                "Year: %{x}<br>"
                "Count: %{y:,}<br>"
                "<extra></extra>"
            ),
        ))

    fig_activity.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8", size=12),
        xaxis=dict(
            title="",
            gridcolor="#1e293b",
            dtick=1,
            linecolor="#1e293b",
        ),
        yaxis=dict(
            title="Total Volume (Patents + Papers)",
            gridcolor="#1e293b",
            linecolor="#1e293b",
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

    # Comparative Distribution
    st.markdown('<div class="section-title">Distribution: Research Papers vs Patents</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_papers = go.Figure()
        for tech_name in selected_domains:
            tech_data = filtered_indicators[
                filtered_indicators["technology_name"] == tech_name
            ].sort_values("year")
            color = COLORS.get(tech_name, "#38bdf8")

            fig_papers.add_trace(go.Bar(
                x=tech_data["year"],
                y=tech_data["paper_count"],
                name=tech_name,
                marker_color=color,
                opacity=0.9,
            ))

        fig_papers.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Research Papers by Year",
            title_font=dict(size=13, color="#e2e8f0"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter", color="#94a3b8"),
            barmode="group",
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Papers"),
            showlegend=False,
        )
        st.plotly_chart(fig_papers, use_container_width=True)

    with col_right:
        fig_patents = go.Figure()
        for tech_name in selected_domains:
            tech_data = filtered_indicators[
                filtered_indicators["technology_name"] == tech_name
            ].sort_values("year")
            color = COLORS.get(tech_name, "#38bdf8")

            fig_patents.add_trace(go.Bar(
                x=tech_data["year"],
                y=tech_data["patent_count"],
                name=tech_name,
                marker_color=color,
                opacity=0.9,
            ))

        fig_patents.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Patent Filings by Year",
            title_font=dict(size=13, color="#e2e8f0"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter", color="#94a3b8"),
            barmode="group",
            xaxis=dict(gridcolor="#1e293b", title=""),
            yaxis=dict(gridcolor="#1e293b", title="Patents"),
            showlegend=False,
        )
        st.plotly_chart(fig_patents, use_container_width=True)

    # YoY Heatmap
    st.markdown('<div class="section-title">Year-over-Year Growth Rate (%)</div>', unsafe_allow_html=True)

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
                "Year": str(years[i]),
                "Growth (%)": round(growth, 1),
            })

    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        pivot = growth_df.pivot(index="Technology", columns="Year", values="Growth (%)")

        fig_heatmap = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(color="Growth %"),
        )
        fig_heatmap.update_layout(
            template=PLOTLY_TEMPLATE,
            height=220,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter", color="#94a3b8"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# ===== TAB 2: TREND SCORES =====
with tab2:
    st.markdown("""
    <div class="callout-box">
        The Trend Score (0–100) is an explainable composite index evaluating 5 multi-dimensional signals:
        Growth Rate (35%), Consistency (25%), Acceleration (20%), Activity Volume (10%), and Citation Velocity (10%).
    </div>
    """, unsafe_allow_html=True)

    score_cols = st.columns(len(filtered_scores)) if len(filtered_scores) > 0 else [st.container()]
    for i, (_, row) in enumerate(filtered_scores.iterrows()):
        label_class = f"score-{row['label'].lower()}"
        with score_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 1.5rem 1rem;">
                <div class="metric-label">{row['technology_name']}</div>
                <div class="metric-value" style="font-size: 2.3rem; margin: 0.2rem 0;">{row['score']}</div>
                <span class="score-badge {label_class}">{row['label']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Factor Component Breakdown</div>', unsafe_allow_html=True)

    fig_radar = go.Figure()
    categories = ["Growth Rate", "Consistency", "Acceleration", "Activity Volume", "Citation Signal"]

    for _, row in filtered_scores.iterrows():
        values = [
            row["growth_rate"], row["consistency"], row["acceleration"],
            row["activity_volume"], row["citation_signal"],
        ]
        values.append(values[0])

        color = COLORS.get(row["technology_name"], "#38bdf8")
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row["technology_name"],
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=f"rgba(56, 189, 248, 0.08)",
        ))

    fig_radar.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8"),
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#1e293b",
            ),
            angularaxis=dict(
                gridcolor="#1e293b",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Component breakdown bars
    st.markdown('<div class="section-title">Component-Level Comparison</div>', unsafe_allow_html=True)

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
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b", title=""),
        yaxis=dict(gridcolor="#1e293b", range=[0, 100], title="Component Value"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ===== TAB 3: FORECASTS =====
with tab3:
    st.markdown("""
    <div class="callout-box">
        Forecast trajectories are modeled using Linear Regression and Moving Average estimators.
        Dashed lines denote projected values; shaded bands represent 80% confidence intervals.
    </div>
    """, unsafe_allow_html=True)

    forecast_tech = st.selectbox(
        "Focus Domain",
        options=selected_domains,
        key="forecast_tech_select",
    )

    for series_type in ["paper_count", "patent_count"]:
        series_label = "Research Papers" if series_type == "paper_count" else "Patent Filings"

        st.markdown(f'<div class="section-title">{series_label} Projection (2025–2027)</div>', unsafe_allow_html=True)

        fig_forecast = go.Figure()
        color = COLORS.get(forecast_tech, "#38bdf8")

        hist_data = filtered_indicators[
            filtered_indicators["technology_name"] == forecast_tech
        ].sort_values("year")

        fig_forecast.add_trace(go.Scatter(
            x=hist_data["year"],
            y=hist_data[series_type],
            name="Historical Observed",
            mode="lines+markers",
            line=dict(width=2.5, color=color),
            marker=dict(size=6, color=color),
        ))

        tech_forecasts = forecasts[
            (forecasts["technology_name"] == forecast_tech) &
            (forecasts["series_type"] == series_type)
        ].sort_values("year")

        if not tech_forecasts.empty:
            last_hist_year = hist_data["year"].iloc[-1]
            last_hist_val = hist_data[series_type].iloc[-1]

            forecast_years = [last_hist_year] + tech_forecasts["year"].tolist()
            forecast_vals = [last_hist_val] + tech_forecasts["predicted_value"].tolist()
            lower_bounds = [last_hist_val] + tech_forecasts["lower_bound"].tolist()
            upper_bounds = [last_hist_val] + tech_forecasts["upper_bound"].tolist()

            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=upper_bounds,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=lower_bounds,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(56, 189, 248, 0.08)",
                showlegend=False,
                hoverinfo="skip",
            ))

            fig_forecast.add_trace(go.Scatter(
                x=forecast_years,
                y=forecast_vals,
                name="Forecast Mean",
                mode="lines+markers",
                line=dict(width=2.5, dash="dash", color=color),
                marker=dict(size=6, symbol="diamond", color=color),
            ))

        fig_forecast.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(
                title="",
                gridcolor="#1e293b",
                dtick=1,
            ),
            yaxis=dict(
                title=f"{series_label} Count",
                gridcolor="#1e293b",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        )

        fig_forecast.add_vline(
            x=2024.5, line_dash="dot", line_color="#475569",
            annotation_text="Forecast Window", annotation_position="top right",
            annotation_font_color="#64748b",
            annotation_font_size=11,
        )

        st.plotly_chart(fig_forecast, use_container_width=True)


# ===== TAB 4: DATA EXPLORER =====
with tab4:
    st.markdown('<div class="section-title">Raw Dataset Explorer</div>', unsafe_allow_html=True)

    data_type = st.radio(
        "Dataset View",
        ["Yearly Indicators", "Technology Metadata", "Trend Scores", "Forecast Projections"],
        horizontal=True,
    )

    if data_type == "Yearly Indicators":
        st.dataframe(filtered_indicators, use_container_width=True, height=450)
    elif data_type == "Technology Metadata":
        st.dataframe(techs, use_container_width=True)
    elif data_type == "Trend Scores":
        st.dataframe(filtered_scores, use_container_width=True)
    elif data_type == "Forecast Projections":
        st.dataframe(forecasts[forecasts["technology_name"].isin(selected_domains)], use_container_width=True, height=450)


# ---------------------------------------------------------------------------
# Minimalist Clean Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.78rem; padding: 2rem 0 1rem 0; border-top: 1px solid #1e293b; margin-top: 3rem;">
    <strong>TechSight</strong> — Automated Technology Intelligence & Forecasting Architecture<br>
    Multi-Source Synthesis Engine: Scholarly Works & Intellectual Property Analytics
</div>
""", unsafe_allow_html=True)
