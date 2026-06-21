"""
ui_components.py
------------------
Reusable, premium healthcare-themed UI building blocks for the Streamlit
app: global CSS injection, header banner, KPI cards, stage cards,
confidence badges, and styled alert boxes.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from PIL import Image

PRIMARY = "#0E7C7B"      # teal
PRIMARY_DARK = "#0A5C5B"
ACCENT = "#2F6FED"       # medical blue
SUCCESS = "#1FA97C"
WARNING = "#E2A33D"
DANGER = "#E2543D"
INK = "#1A2B3C"
MUTED = "#6B7C93"
BG = "#F4F8FA"
CARD_BG = "#FFFFFF"


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(180deg, {BG} 0%, #EAF2F3 100%);
        }}

        /* Hide default Streamlit chrome for a cleaner product feel */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}

        /* ---------- Hero header ---------- */
        .app-hero {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 55%, {ACCENT} 130%);
            border-radius: 18px;
            padding: 28px 34px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(14, 124, 123, 0.25);
            color: white;
            animation: fadeIn 0.6s ease-in-out;
        }}
        .app-hero h1 {{
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0 0 4px 0;
            letter-spacing: -0.02em;
        }}
        .app-hero p {{
            margin: 0;
            opacity: 0.92;
            font-size: 0.95rem;
            font-weight: 400;
        }}
        .app-hero .badge-row {{
            margin-top: 14px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .hero-chip {{
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        /* ---------- Generic card ---------- */
        .pr-card {{
            background: {CARD_BG};
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 14px rgba(26, 43, 60, 0.06);
            border: 1px solid #E6EEF0;
            margin-bottom: 16px;
            animation: fadeIn 0.45s ease-in-out;
        }}
        .pr-card h4 {{
            margin: 0 0 10px 0;
            color: {INK};
            font-size: 1rem;
            font-weight: 700;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0 14px 0;
        }}
        .section-header .icon {{
            font-size: 1.3rem;
        }}
        .section-header h2 {{
            font-size: 1.18rem;
            font-weight: 800;
            color: {INK};
            margin: 0;
        }}
        .section-sub {{
            color: {MUTED};
            font-size: 0.86rem;
            margin: -8px 0 16px 32px;
        }}

        /* ---------- KPI cards ---------- */
        .kpi-card {{
            background: {CARD_BG};
            border-radius: 14px;
            padding: 16px 18px;
            border: 1px solid #E6EEF0;
            box-shadow: 0 2px 14px rgba(26, 43, 60, 0.05);
            text-align: left;
            height: 100%;
        }}
        .kpi-card .kpi-icon {{
            font-size: 1.4rem;
            margin-bottom: 6px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: {INK};
            line-height: 1.1;
        }}
        .kpi-card .kpi-label {{
            font-size: 0.8rem;
            color: {MUTED};
            font-weight: 600;
            margin-top: 2px;
        }}

        /* ---------- Stage image cards ---------- */
        .stage-caption {{
            text-align: center;
            font-weight: 700;
            color: {INK};
            font-size: 0.86rem;
            margin-top: 8px;
        }}
        .stage-sub {{
            text-align: center;
            color: {MUTED};
            font-size: 0.72rem;
            margin-top: 2px;
        }}

        /* ---------- Confidence badges ---------- */
        .conf-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .conf-high {{ background: rgba(31,169,124,0.14); color: {SUCCESS}; }}
        .conf-mid  {{ background: rgba(226,163,61,0.16); color: {WARNING}; }}
        .conf-low  {{ background: rgba(226,84,61,0.14); color: {DANGER}; }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.3rem;
            font-weight: 700;
            font-size: 0.92rem;
            transition: all 0.15s ease-in-out;
            box-shadow: 0 4px 14px rgba(14,124,123,0.25);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(14,124,123,0.35);
        }}

        /* ---------- Misc ---------- */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F2F8F8 100%);
            border-right: 1px solid #E6EEF0;
        }}

        /* ---------- Alerts ---------- */
        div[data-testid="stNotification"], .stAlert {{
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(26,43,60,0.08);
        }}

        /* ---------- Dataframe / table polish ---------- */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #E6EEF0;
        }}

        /* ---------- Metric polish (native st.metric) ---------- */
        [data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid #E6EEF0;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 2px 14px rgba(26, 43, 60, 0.05);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, chips: Optional[list] = None) -> None:
    chips = chips or []
    chip_html = "".join(f'<span class="hero-chip">{c}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>🩺 {title}</h1>
            <p>{subtitle}</p>
            <div class="badge-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header"><span class="icon">{icon}</span><h2>{title}</h2></div>
        {f'<div class="section-sub">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(icon: str, value: str, label: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_image_card(image: Image.Image, title: str, subtitle: str) -> None:
    st.image(image, use_container_width=True)
    st.markdown(
        f'<div class="stage-caption">{title}</div><div class="stage-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def confidence_badge_html(confidence: int) -> str:
    if confidence >= 80:
        css_class = "conf-high"
    elif confidence >= 50:
        css_class = "conf-mid"
    else:
        css_class = "conf-low"
    return f'<span class="conf-badge {css_class}">{confidence}%</span>'


def quality_badge_html(quality: str) -> str:
    mapping = {
        "Excellent": "conf-high",
        "Good": "conf-high",
        "Fair": "conf-mid",
        "Poor": "conf-low",
    }
    css_class = mapping.get(quality, "conf-mid")
    return f'<span class="conf-badge {css_class}">{quality}</span>'
