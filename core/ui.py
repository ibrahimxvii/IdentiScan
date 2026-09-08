"""
IdentiScan - Shared UI helpers (glassmorphism dark theme).
"""
import os
import streamlit as st

_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")


def inject_css():
    if os.path.exists(_CSS_PATH):
        with open(_CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def page_header(title, subtitle=None, icon=None):
    # Raw HTML passed via unsafe_allow_html is not re-parsed for Streamlit's
    # ":material/x:" markdown shortcode, so the icon is rendered directly as
    # a Material Symbols ligature span instead (same font Streamlit's own
    # native icons use - see the .material-symbols-outlined rule in style.css).
    icon_html = f'<span class="material-symbols-outlined fg-header-icon">{icon}</span>' if icon else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="fg-header"><h1>{icon_html}{title}</h1>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, hint=None):
    hint_html = f'<div class="fg-metric-hint">{hint}</div>' if hint else ""
    html = (
        f'<div class="fg-metric-card">'
        f'<div class="fg-metric-value">{value}</div>'
        f'<div class="fg-metric-label">{label}</div>'
        f'{hint_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def status_pill(label, ok):
    cls = "fg-pill-ok" if ok else "fg-pill-bad"
    text = "Online" if ok else "Offline"
    return f'<span class="fg-pill {cls}">{label} &middot; {text}</span>'


def empty_state(title, description):
    html = (
        f'<div class="fg-empty-state">'
        f'<h3>{title}</h3>'
        f'<p>{description}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)