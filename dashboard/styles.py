"""Theme, palette and custom Streamlit CSS for the OceanMind AI dashboard.

This module is the single source of truth for the dashboard's visual language:
the dark ocean palette, the injected CSS that restyles Streamlit's chrome, the
Plotly template every chart inherits, and the small HTML builders used for
premium KPI cards and section headers.

No other module should hardcode a colour. Import from :data:`OCEAN` instead.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Dict, Final, Literal, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

logger = logging.getLogger(__name__)

_ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"
_LOGO_PATH: Final[Path] = _ASSETS_DIR / "logo.svg"


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

OCEAN: Final[Dict[str, str]] = {
    # Surfaces, deepest to highest
    "bg_deep": "#050B18",
    "bg_panel": "#0A1424",
    "bg_card": "#0F1E33",
    "bg_card_hover": "#142743",
    "border": "#1C3350",
    "border_strong": "#2A4A70",
    # Typography
    "text_primary": "#E8F1FA",
    "text_secondary": "#A9C0DA",
    "text_muted": "#7591B2",
    # Brand accents
    "accent": "#22D3EE",
    "accent_deep": "#0EA5E9",
    "accent_alt": "#3B82F6",
    "teal": "#2DD4BF",
    "violet": "#8B5CF6",
    # Semantic states
    "good": "#34D399",
    "warn": "#FBBF24",
    "bad": "#F87171",
    "critical": "#EF4444",
}

#: Ordered colours for categorical series (float IDs, variables, regions).
CATEGORICAL: Final[Sequence[str]] = (
    "#22D3EE",
    "#34D399",
    "#FBBF24",
    "#8B5CF6",
    "#F87171",
    "#3B82F6",
    "#2DD4BF",
    "#F472B6",
)

#: Continuous scale for depth, temperature and other magnitude encodings.
SEQUENTIAL: Final[Sequence[str]] = (
    "#0A1424",
    "#12324F",
    "#1C5A7A",
    "#22849B",
    "#2DB0A8",
    "#7DD8A8",
)

#: Diverging scale for anomalies (below normal → above normal).
DIVERGING: Final[Sequence[str]] = (
    "#3B82F6",
    "#7DA9D8",
    "#CBD9E8",
    "#F3C77B",
    "#F87171",
)

FONT_STACK: Final[str] = (
    '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif'
)

PLOTLY_TEMPLATE: Final[str] = "oceanmind"

Tone = Literal["neutral", "good", "warn", "bad", "accent"]

_TONE_COLOURS: Final[Dict[str, str]] = {
    "neutral": OCEAN["text_secondary"],
    "good": OCEAN["good"],
    "warn": OCEAN["warn"],
    "bad": OCEAN["bad"],
    "accent": OCEAN["accent"],
}


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #


def register_plotly_template() -> str:
    """Register and activate the dark ocean Plotly template.

    Registering once at start-up means every figure in :mod:`dashboard.profile_plots`
    inherits the theme without repeating layout code.

    Returns:
        The template name, so callers may pass it explicitly if they wish.
    """
    if PLOTLY_TEMPLATE not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_STACK, color=OCEAN["text_secondary"], size=13),
                title=dict(
                    font=dict(color=OCEAN["text_primary"], size=17),
                    x=0.0,
                    xanchor="left",
                ),
                colorway=list(CATEGORICAL),
                xaxis=dict(
                    gridcolor=OCEAN["border"],
                    zerolinecolor=OCEAN["border_strong"],
                    linecolor=OCEAN["border"],
                    tickfont=dict(color=OCEAN["text_muted"], size=11),
                    title_font=dict(color=OCEAN["text_secondary"], size=12),
                ),
                yaxis=dict(
                    gridcolor=OCEAN["border"],
                    zerolinecolor=OCEAN["border_strong"],
                    linecolor=OCEAN["border"],
                    tickfont=dict(color=OCEAN["text_muted"], size=11),
                    title_font=dict(color=OCEAN["text_secondary"], size=12),
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor=OCEAN["border"],
                    borderwidth=1,
                    font=dict(color=OCEAN["text_secondary"], size=11),
                ),
                hoverlabel=dict(
                    bgcolor=OCEAN["bg_card"],
                    bordercolor=OCEAN["border_strong"],
                    font=dict(family=FONT_STACK, color=OCEAN["text_primary"], size=12),
                ),
                margin=dict(l=56, r=24, t=52, b=48),
                colorscale=dict(
                    sequential=[
                        [index / (len(SEQUENTIAL) - 1), colour]
                        for index, colour in enumerate(SEQUENTIAL)
                    ],
                    diverging=[
                        [index / (len(DIVERGING) - 1), colour]
                        for index, colour in enumerate(DIVERGING)
                    ],
                ),
            )
        )
        logger.debug("Registered Plotly template %r", PLOTLY_TEMPLATE)

    pio.templates.default = PLOTLY_TEMPLATE
    return PLOTLY_TEMPLATE


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #

_GLOBAL_CSS: Final[str] = f"""
<style>
  .stApp {{
    background:
      radial-gradient(1100px 620px at 12% -8%, #10294A 0%, rgba(16,41,74,0) 60%),
      radial-gradient(900px 520px at 92% 4%, #0C3B49 0%, rgba(12,59,73,0) 58%),
      {OCEAN["bg_deep"]};
    color: {OCEAN["text_primary"]};
    font-family: {FONT_STACK};
  }}

  #MainMenu, footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}
  .block-container {{ padding: 1.6rem 2.2rem 3rem; max-width: 1500px; }}

  h1, h2, h3, h4 {{
    color: {OCEAN["text_primary"]};
    font-family: {FONT_STACK};
    letter-spacing: -0.015em;
    font-weight: 650;
  }}
  h1 {{ font-size: 1.95rem; }}
  p, span, label, li {{ color: {OCEAN["text_secondary"]}; }}

  /* ----- Sidebar ----- */
  [data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {OCEAN["bg_panel"]} 0%, {OCEAN["bg_deep"]} 100%);
    border-right: 1px solid {OCEAN["border"]};
  }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  /* ----- Cards ----- */
  .om-card {{
    background: linear-gradient(160deg, {OCEAN["bg_card"]} 0%, {OCEAN["bg_panel"]} 100%);
    border: 1px solid {OCEAN["border"]};
    border-radius: 14px;
    padding: 1.05rem 1.2rem;
    height: 100%;
    transition: border-color .18s ease, transform .18s ease;
  }}
  .om-card:hover {{ border-color: {OCEAN["border_strong"]}; transform: translateY(-1px); }}
  .om-card__label {{
    color: {OCEAN["text_muted"]};
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .085em;
    text-transform: uppercase;
    margin-bottom: .45rem;
  }}
  .om-card__value {{
    color: {OCEAN["text_primary"]};
    font-size: 1.72rem;
    font-weight: 680;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
  }}
  .om-card__delta {{ font-size: .82rem; font-weight: 600; margin-top: .3rem; }}
  .om-card__caption {{ color: {OCEAN["text_muted"]}; font-size: .76rem; margin-top: .3rem; }}

  /* ----- Section header ----- */
  .om-section {{ margin: .35rem 0 1rem; }}
  .om-section__title {{
    color: {OCEAN["text_primary"]};
    font-size: 1.06rem;
    font-weight: 650;
    display: flex;
    align-items: center;
    gap: .5rem;
  }}
  .om-section__title::before {{
    content: "";
    width: 3px; height: 17px; border-radius: 2px;
    background: linear-gradient(180deg, {OCEAN["accent"]}, {OCEAN["accent_alt"]});
  }}
  .om-section__subtitle {{
    color: {OCEAN["text_muted"]};
    font-size: .83rem;
    margin: .3rem 0 0 .85rem;
  }}

  /* ----- Pills / badges ----- */
  .om-pill {{
    display: inline-flex; align-items: center; gap: .38rem;
    padding: .24rem .68rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600;
    border: 1px solid {OCEAN["border_strong"]};
    background: rgba(34,211,238,.08);
    color: {OCEAN["accent"]};
  }}
  .om-pill--good {{ color:{OCEAN["good"]}; background:rgba(52,211,153,.10); border-color:rgba(52,211,153,.35); }}
  .om-pill--warn {{ color:{OCEAN["warn"]}; background:rgba(251,191,36,.10); border-color:rgba(251,191,36,.35); }}
  .om-pill--bad  {{ color:{OCEAN["bad"]};  background:rgba(248,113,113,.10); border-color:rgba(248,113,113,.35); }}

  /* ----- Tabs ----- */
  .stTabs [data-baseweb="tab-list"] {{
    gap: .35rem;
    border-bottom: 1px solid {OCEAN["border"]};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 9px 9px 0 0;
    color: {OCEAN["text_muted"]};
    padding: .55rem 1.05rem;
    font-weight: 570;
  }}
  .stTabs [aria-selected="true"] {{
    background: {OCEAN["bg_card"]};
    color: {OCEAN["accent"]};
  }}

  /* ----- Inputs ----- */
  [data-testid="stWidgetLabel"] label p {{
    color: {OCEAN["text_muted"]};
    font-size: .76rem;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
  }}

  /* ----- Buttons ----- */
  .stButton > button, .stDownloadButton > button {{
    background: {OCEAN["bg_card"]};
    color: {OCEAN["text_primary"]};
    border: 1px solid {OCEAN["border_strong"]};
    border-radius: 10px;
    font-weight: 580;
    transition: all .16s ease;
  }}
  .stButton > button:hover:not(:disabled),
  .stDownloadButton > button:hover:not(:disabled) {{
    border-color: {OCEAN["accent"]};
    color: {OCEAN["accent"]};
    background: {OCEAN["bg_card_hover"]};
  }}
  .stButton > button:disabled, .stDownloadButton > button:disabled {{
    opacity: .45; cursor: not-allowed;
  }}

  /* ----- Dataframes, metrics, chat ----- */
  [data-testid="stDataFrame"] {{
    border: 1px solid {OCEAN["border"]};
    border-radius: 12px;
    overflow: hidden;
  }}
  [data-testid="stMetricValue"] {{ color:{OCEAN["text_primary"]}; font-variant-numeric: tabular-nums; }}
  [data-testid="stChatMessage"] {{
    background: {OCEAN["bg_card"]};
    border: 1px solid {OCEAN["border"]};
    border-radius: 13px;
  }}
  [data-testid="stExpander"] details {{
    background: {OCEAN["bg_panel"]};
    border: 1px solid {OCEAN["border"]};
    border-radius: 12px;
  }}
  hr {{ border-color: {OCEAN["border"]}; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
  ::-webkit-scrollbar-track {{ background: {OCEAN["bg_deep"]}; }}
  ::-webkit-scrollbar-thumb {{ background: {OCEAN["border_strong"]}; border-radius: 6px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {OCEAN["accent_deep"]}; }}

  /* ----- Sidebar brand ----- */
  .om-brand {{ display:flex; align-items:center; gap:.65rem; padding:.2rem 0 1rem; }}
  .om-brand__mark {{
    filter: drop-shadow(0 0 9px rgba(34,211,238,.45));
    animation: om-pulse 5s ease-in-out infinite;
    flex-shrink: 0;
  }}
  .om-brand__name {{
    font-size: 1.18rem; font-weight: 720; color: {OCEAN["text_primary"]};
    letter-spacing: -.02em; line-height: 1.15;
  }}
  .om-brand__tagline {{
    color: {OCEAN["text_muted"]}; font-size: .72rem;
    letter-spacing: .09em; text-transform: uppercase; margin-top: .1rem;
  }}

  /* ----- Page hero ----- */
  .om-hero {{
    display: flex; align-items: center; gap: 1rem;
    padding: .3rem 0 1.3rem;
    border-bottom: 1px solid {OCEAN["border"]};
    margin-bottom: 1.4rem;
    position: relative;
  }}
  .om-hero__mark {{
    filter: drop-shadow(0 0 14px rgba(34,211,238,.4));
    animation: om-pulse 5s ease-in-out infinite;
    flex-shrink: 0;
  }}
  .om-hero__title {{
    font-size: 2rem; font-weight: 720; letter-spacing: -.02em; line-height: 1.1;
    background: linear-gradient(120deg, {OCEAN["text_primary"]} 30%, {OCEAN["accent"]} 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .om-hero__subtitle {{
    color: {OCEAN["text_muted"]}; font-size: .92rem; margin-top: .35rem; max-width: 62ch;
  }}
  .om-hero__pills {{ display:flex; gap:.4rem; margin-top:.55rem; flex-wrap: wrap; }}

  @keyframes om-pulse {{
    0%, 100% {{ filter: drop-shadow(0 0 9px rgba(34,211,238,.35)); }}
    50% {{ filter: drop-shadow(0 0 16px rgba(34,211,238,.65)); }}
  }}

  /* ----- Top navigation bar (the workspace switcher radio) ----- */
  div[data-testid="stRadio"] > div[role="radiogroup"] {{
    gap: .5rem; flex-wrap: wrap; margin-bottom: .2rem;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] label {{
    background: {OCEAN["bg_card"]};
    border: 1px solid {OCEAN["border"]};
    border-radius: 10px;
    padding: .5rem 1.15rem;
    font-weight: 580;
    color: {OCEAN["text_muted"]};
    transition: all .16s ease;
    cursor: pointer;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    border-color: {OCEAN["border_strong"]};
    color: {OCEAN["text_secondary"]};
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
    background: {OCEAN["bg_card_hover"]};
    border-color: {OCEAN["accent"]};
    color: {OCEAN["accent"]};
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {{
    display: none; /* hide the native radio dot; the pill itself shows state */
  }}

  /* ----- Soften Streamlit's rerun dim so filter changes feel snappier ----- */
  [data-stale="true"] {{
    opacity: 0.72 !important;
    transition: opacity .12s ease !important;
  }}
</style>
"""


def inject_global_css() -> None:
    """Inject the dashboard's global stylesheet.

    Call exactly once, immediately after ``st.set_page_config`` in
    :mod:`dashboard.app`.
    """
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def apply_theme() -> None:
    """Apply the full visual theme: CSS plus the Plotly template."""
    inject_global_css()
    register_plotly_template()


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #


def kpi_card_html(
    label: str,
    value: str,
    *,
    delta: Optional[str] = None,
    delta_tone: Tone = "neutral",
    caption: Optional[str] = None,
) -> str:
    """Build the markup for a single premium KPI card.

    Args:
        label: Small uppercase label above the figure.
        value: The headline figure, already formatted for display.
        delta: Optional change indicator, e.g. ``"+2.4% vs last month"``.
        delta_tone: Semantic colour applied to ``delta``.
        caption: Optional muted line beneath the value.

    Returns:
        An HTML fragment for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    parts = [
        '<div class="om-card">',
        f'<div class="om-card__label">{label}</div>',
        f'<div class="om-card__value">{value}</div>',
    ]
    if delta:
        colour = _TONE_COLOURS.get(delta_tone, OCEAN["text_secondary"])
        parts.append(f'<div class="om-card__delta" style="color:{colour}">{delta}</div>')
    if caption:
        parts.append(f'<div class="om-card__caption">{caption}</div>')
    parts.append("</div>")
    return "".join(parts)


def render_kpi_row(cards: Sequence[Dict[str, object]]) -> None:
    """Render a responsive row of KPI cards.

    Args:
        cards: One mapping per card, using the keyword names accepted by
            :func:`kpi_card_html` (``label`` and ``value`` are required).
    """
    if not cards:
        return
    columns = st.columns(len(cards), gap="medium")
    for column, card in zip(columns, cards):
        with column:
            st.markdown(
                kpi_card_html(
                    str(card.get("label", "")),
                    str(card.get("value", "--")),
                    delta=card.get("delta"),  # type: ignore[arg-type]
                    delta_tone=card.get("delta_tone", "neutral"),  # type: ignore[arg-type]
                    caption=card.get("caption"),  # type: ignore[arg-type]
                ),
                unsafe_allow_html=True,
            )


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render a titled section divider with an accent bar.

    Args:
        title: Section title.
        subtitle: Optional supporting description.
    """
    markup = [f'<div class="om-section"><div class="om-section__title">{title}</div>']
    if subtitle:
        markup.append(f'<div class="om-section__subtitle">{subtitle}</div>')
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


@functools.lru_cache(maxsize=1)
def _raw_logo_svg() -> Optional[str]:
    """Read the OceanMind AI wordmark SVG from disk, once per process.

    Returns:
        The raw SVG markup, or ``None`` if the asset is missing (callers
        fall back to an emoji mark so a missing file never breaks the page).
    """
    try:
        return _LOGO_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Logo asset not found at %s; falling back to emoji mark", _LOGO_PATH)
        return None


def logo_svg_markup(size: int = 40, *, css_class: str = "") -> str:
    """Return the brand mark as inline SVG, sized for the given context.

    Inlining (rather than pointing an ``<img>`` at the assets folder) means
    the mark renders correctly without needing Streamlit's static file
    serving enabled, and it can be styled/animated via the classes above.

    Args:
        size: Width and height in pixels.
        css_class: Extra CSS class(es) to attach to the wrapping span.

    Returns:
        An HTML fragment for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    raw = _raw_logo_svg()
    if raw is None:
        return f'<span class="{css_class}" style="font-size:{size * 0.75}px">&#127754;</span>'
    sized = raw.replace('width="64"', f'width="{size}"').replace('height="64"', f'height="{size}"')
    return f'<span class="{css_class}">{sized}</span>'


def render_hero(
    title: str,
    subtitle: str,
    *,
    pills: Optional[Sequence[str]] = None,
) -> None:
    """Render the branded page header shown at the top of every workspace.

    Replaces plain ``st.title``/``st.caption`` calls with the animated logo
    mark, a gradient headline and optional status pills, so every page opens
    with a consistent, designed first impression rather than default
    Streamlit typography.

    Args:
        title: Page headline (e.g. ``"Explore Ocean"``).
        subtitle: One or two sentences of supporting copy.
        pills: Optional pre-built pill HTML fragments (see :func:`pill`) to
            show under the subtitle, e.g. a live data-range badge.
    """
    parts = [
        '<div class="om-hero">',
        logo_svg_markup(52, css_class="om-hero__mark"),
        '<div>',
        f'<div class="om-hero__title">{title}</div>',
        f'<div class="om-hero__subtitle">{subtitle}</div>',
    ]
    if pills:
        parts.append('<div class="om-hero__pills">' + "".join(pills) + "</div>")
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def pill(text: str, tone: Tone = "accent") -> str:
    """Build a small status badge.

    Args:
        text: Badge text.
        tone: Semantic colour variant.

    Returns:
        An HTML fragment for use with ``unsafe_allow_html=True``.
    """
    modifier = "" if tone in {"accent", "neutral"} else f" om-pill--{tone}"
    return f'<span class="om-pill{modifier}">{text}</span>'


# Register on import so any module building figures inherits the theme, even
# when imported outside a Streamlit runtime (tests, notebooks, scripts).
register_plotly_template()
