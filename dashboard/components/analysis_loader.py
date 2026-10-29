
"""
dashboard/components/analysis_loader.py — Premium Animated Technology Intelligence Loader.

Provides an enterprise-grade, animated loading experience with live radar pulses,
glowing progress indicators, and an interactive 4-stage visual pipeline tracker.
"""

import textwrap
import streamlit as st


def _safe_html(html_str):
    """Safely format HTML without leading spaces so markdown never interprets it as a code block."""
    cleaned = textwrap.dedent(html_str).strip()
    return "\n".join(line.lstrip() for line in cleaned.splitlines())


def _build_svg_spinner():
    return _safe_html("""
    <svg class="loader-spinner-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="9" stroke="#93c5fd" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="14 14" opacity="0.4"/>
        <circle cx="12" cy="12" r="9" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="24 40"/>
    </svg>
    """)


def _build_svg_check():
    return _safe_html("""
    <svg class="loader-check-svg" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="10" cy="10" r="9" fill="#10b981"/>
        <path d="M6 10.2L8.6 12.8L14 7.4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """)


def _build_svg_pending(num):
    return f'<div class="loader-pending-num">{num}</div>'


def render_animated_loader_html(tech_name, percent, current_message, stages_state):
    """
    Renders modern, animated HTML/CSS for TECHSIGHT live intelligence gathering.
    """
    clamped_pct = min(max(int(percent), 5), 100)
    query_display = tech_name.strip() if tech_name else "Technology Domain"

    # Define the 4 high-level stages
    cards_meta = [
        {
            "id": "research",
            "title": "Scientific Research Signals",
            "subtitle": "OpenAlex Global Scholarly Volume (2018–2026)",
            "num": "1",
        },
        {
            "id": "patents",
            "title": "Intellectual Property Records",
            "subtitle": "European Patent Office DocDB Filings",
            "num": "2",
        },
        {
            "id": "topics",
            "title": "Subfield & Topic Extraction",
            "subtitle": "Semantic theme & abstract discovery",
            "num": "3",
        },
        {
            "id": "forecast",
            "title": "Trajectory & Momentum Forecast",
            "subtitle": "Composite trend score & predictive horizon",
            "num": "4",
        },
    ]

    cards_html = []
    for c in cards_meta:
        cid = c["id"]
        state = stages_state.get(cid, "pending")  # "completed", "active", "pending"
        
        if state == "completed":
            badge_class = "stage-card-completed"
            icon_markup = _build_svg_check()
            status_tag = '<span class="stage-tag tag-verified">VERIFIED</span>'
        elif state == "active":
            badge_class = "stage-card-active"
            icon_markup = _build_svg_spinner()
            status_tag = '<span class="stage-tag tag-active"><span class="tag-pulse"></span>ACTIVE</span>'
        else:
            badge_class = "stage-card-pending"
            icon_markup = _build_svg_pending(c["num"])
            status_tag = '<span class="stage-tag tag-queued">QUEUED</span>'

        # Special subtitle if patents is currently active with custom year detail
        subtitle = c["subtitle"]
        if cid == "patents" and state == "active" and "Searching" in current_message:
            subtitle = current_message

        card = f"""
        <div class="loader-stage-card {badge_class}">
            <div class="stage-card-left">
                <div class="stage-icon-wrap">{icon_markup}</div>
                <div class="stage-text-wrap">
                    <div class="stage-title">{c['title']}</div>
                    <div class="stage-subtitle">{subtitle}</div>
                </div>
            </div>
            <div class="stage-card-right">
                {status_tag}
            </div>
        </div>
        """
        cards_html.append(card)

    stages_grid = "\n".join(cards_html)

    return _safe_html(f"""
    <style>
    @keyframes loaderBeaconPulse {{
        0% {{ transform: scale(0.92); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.6); }}
        70% {{ transform: scale(1.05); box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }}
        100% {{ transform: scale(0.92); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }}
    }}
    @keyframes loaderSpinRing {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    @keyframes loaderShimmerSlide {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(250%); }}
    }}
    @keyframes tagPulseGlow {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.4; transform: scale(1.3); }}
    }}

    .techsight-loader-container {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        box-shadow: 0 12px 32px -6px rgba(15, 23, 42, 0.08), 0 4px 12px -2px rgba(15, 23, 42, 0.04);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        color: #0f172a;
    }}

    .loader-header-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.125rem;
        flex-wrap: wrap;
    }}

    .loader-brand-side {{
        display: flex;
        align-items: center;
        gap: 0.875rem;
    }}

    .loader-beacon-core {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #2563eb;
        animation: loaderBeaconPulse 2s infinite ease-in-out;
        flex-shrink: 0;
    }}

    .loader-headline-label {{
        font-size: 0.6875rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 0.15rem;
    }}

    .loader-query-title {{
        font-size: 1.1875rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.015em;
        line-height: 1.25;
    }}

    .loader-query-badge {{
        color: #1d4ed8;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 0.1rem 0.55rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1rem;
    }}

    .loader-pct-pill {{
        background: #0f172a;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.8125rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        letter-spacing: 0.02em;
        font-variant-numeric: tabular-nums;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15);
    }}

    .loader-progress-track {{
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
        margin-bottom: 1rem;
    }}

    .loader-progress-bar {{
        height: 100%;
        width: {clamped_pct}%;
        background: linear-gradient(90deg, #2563eb 0%, #06b6d4 70%, #3b82f6 100%);
        border-radius: 999px;
        transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .loader-progress-shimmer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.6) 50%, transparent 100%);
        animation: loaderShimmerSlide 1.8s infinite linear;
    }}

    .loader-status-ticker {{
        display: flex;
        align-items: center;
        gap: 0.625rem;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.55rem 0.875rem;
        margin-bottom: 1.125rem;
        font-size: 0.8125rem;
        color: #334155;
        font-weight: 500;
    }}

    .ticker-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2563eb;
        animation: tagPulseGlow 1.4s infinite ease-in-out;
        flex-shrink: 0;
    }}

    .loader-stages-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.75rem;
    }}

    .loader-stage-card {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0.875rem;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        transition: all 0.25s ease;
    }}

    .stage-card-completed {{
        background: #f0fdf4;
        border-color: #bbf7d0;
    }}

    .stage-card-active {{
        background: #eff6ff;
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    }}

    .stage-card-pending {{
        background: #f8fafc;
        border-color: #e2e8f0;
        opacity: 0.75;
    }}

    .stage-card-left {{
        display: flex;
        align-items: center;
        gap: 0.625rem;
        min-width: 0;
    }}

    .stage-icon-wrap {{
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}

    .loader-spinner-svg {{
        width: 20px;
        height: 20px;
        animation: loaderSpinRing 1s linear infinite;
    }}

    .loader-check-svg {{
        width: 20px;
        height: 20px;
    }}

    .loader-pending-num {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #e2e8f0;
        color: #64748b;
        font-size: 0.6875rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .stage-text-wrap {{
        min-width: 0;
    }}

    .stage-title {{
        font-size: 0.8125rem;
        font-weight: 700;
        color: #0f172a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .stage-subtitle {{
        font-size: 0.6875rem;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .stage-card-active .stage-title {{
        color: #1d4ed8;
    }}

    .stage-card-active .stage-subtitle {{
        color: #2563eb;
        font-weight: 600;
    }}

    .stage-tag {{
        font-size: 0.625rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.45rem;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        flex-shrink: 0;
    }}

    .tag-verified {{
        background: #dcfce7;
        color: #15803d;
    }}

    .tag-active {{
        background: #dbeafe;
        color: #1d4ed8;
    }}

    .tag-queued {{
        background: #e2e8f0;
        color: #64748b;
    }}

    .tag-pulse {{
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #2563eb;
        animation: tagPulseGlow 1s infinite ease-in-out;
    }}
    </style>

    <div class="techsight-loader-container">
        <div class="loader-header-row">
            <div class="loader-brand-side">
                <div class="loader-beacon-core"></div>
                <div>
                    <div class="loader-headline-label">Live Intelligence Engine &bull; Real-Time Ingestion</div>
                    <div class="loader-query-title">Synthesizing signals for <span class="loader-query-badge">&ldquo;{query_display}&rdquo;</span></div>
                </div>
            </div>
            <div class="loader-pct-pill">{clamped_pct}% SYNTHESIZED</div>
        </div>

        <div class="loader-progress-track">
            <div class="loader-progress-bar">
                <div class="loader-progress-shimmer"></div>
            </div>
        </div>

        <div class="loader-status-ticker">
            <span class="ticker-dot"></span>
            <span>{current_message}</span>
        </div>

        <div class="loader-stages-grid">
            {stages_grid}
        </div>
    </div>
    """)


def create_analysis_loader(tech_name=None):
    """
    Creates an interactive, animated progress panel for live technology intelligence.
    
    Returns:
        tuple: (update_fn, complete_fn, fail_fn)
    """
    placeholder = st.empty()
    stages_state = {
        "research": "active",
        "patents": "pending",
        "topics": "pending",
        "forecast": "pending",
    }
    initial_msg = "Connecting to global scientific intelligence repositories..."
    placeholder.markdown(
        render_animated_loader_html(tech_name, 10, initial_msg, stages_state),
        unsafe_allow_html=True,
    )

    stage_defaults = {
        "init": "Initializing scientific intelligence pipeline...",
        "research": "Retrieving global publication volume (OpenAlex)...",
        "sample_docs": "Ingesting recent research papers and abstracts...",
        "patents": "Searching worldwide patent registries (European Patent Office DocDB)...",
        "topics": "Discovering key research topics and subfields...",
        "trends": "Evaluating composite trend scores and momentum...",
        "forecast": "Generating predictive growth forecasts...",
    }

    def update(stage, percent=None, custom_msg=None):
        msg = custom_msg or stage_defaults.get(stage, stage)
        pct = percent if percent is not None else 20

        # Update stage states based on progress
        if stage in ("research", "sample_docs"):
            stages_state["research"] = "active"
            stages_state["patents"] = "pending"
            stages_state["topics"] = "pending"
            stages_state["forecast"] = "pending"
        elif stage == "patents":
            stages_state["research"] = "completed"
            stages_state["patents"] = "active"
            stages_state["topics"] = "pending"
            stages_state["forecast"] = "pending"
        elif stage == "topics":
            stages_state["research"] = "completed"
            stages_state["patents"] = "completed"
            stages_state["topics"] = "active"
            stages_state["forecast"] = "pending"
        elif stage in ("trends", "forecast"):
            stages_state["research"] = "completed"
            stages_state["patents"] = "completed"
            stages_state["topics"] = "completed"
            stages_state["forecast"] = "active"
        elif stage == "complete":
            stages_state["research"] = "completed"
            stages_state["patents"] = "completed"
            stages_state["topics"] = "completed"
            stages_state["forecast"] = "completed"

        placeholder.markdown(
            render_animated_loader_html(tech_name, pct, msg, stages_state),
            unsafe_allow_html=True,
        )

    def complete(custom_msg=None):
        stages_state["research"] = "completed"
        stages_state["patents"] = "completed"
        stages_state["topics"] = "completed"
        stages_state["forecast"] = "completed"
        
        # Display subtle completion banner
        completed_html = _safe_html(f"""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.65rem 1rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.625rem;">
                <span style="color: #16a34a; font-size: 1.125rem; font-weight: 800;">&#10003;</span>
                <span style="font-size: 0.8125rem; font-weight: 700; color: #15803d;">
                    Live Intelligence Ready &bull; Verified across OpenAlex &amp; European Patent Office (EPO OPS)
                </span>
            </div>
            <span style="font-size: 0.6875rem; font-weight: 700; color: #16a34a; background: #dcfce7; padding: 0.2rem 0.5rem; border-radius: 4px;">
                100% COMPLETE
            </span>
        </div>
        """)
        placeholder.markdown(completed_html, unsafe_allow_html=True)

    def fail(error_msg=None):
        err = error_msg or "Analysis could not be completed"
        fail_html = _safe_html(f"""
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.75rem;">
            <span style="color: #dc2626; font-size: 1.125rem; font-weight: 800;">&#9888;</span>
            <div style="font-size: 0.8125rem; font-weight: 600; color: #b91c1c;">
                {err}
            </div>
        </div>
        """)
        placeholder.markdown(fail_html, unsafe_allow_html=True)

    return update, complete, fail
