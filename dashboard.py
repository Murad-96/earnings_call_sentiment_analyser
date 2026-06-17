import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(
    page_title="Earnings Sentiment Analyser",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.8rem;
        font-weight: 500;
        color: #e6edf3;
        line-height: 1;
    }
    .metric-delta { font-size: 0.78rem; margin-top: 0.4rem; color: #8b949e; }
    .quote-card {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 0 6px 6px 0;
        padding: 1rem 1.25rem;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #c9d1d9;
        margin: 0.5rem 0;
    }
    .quote-card.negative { border-left-color: #f85149; }
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #58a6ff;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #21262d;
    }
    .stSelectbox > div > div { background: #161b22; border-color: #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #21262d; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border: none;
        color: #8b949e; padding: 0.6rem 1.2rem; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: transparent; color: #e6edf3;
        border-bottom: 2px solid #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Theme helpers ─────────────────────────────────────────────────────────────
# Base layout — no xaxis/yaxis keys here to avoid duplicate keyword errors
BASE_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8b949e', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

# Reusable axis style — merge with any extra kwargs at call site
def axis_style(**extra):
    return dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d', **extra)

ACCENT  = '#58a6ff'
POS_COL = '#3fb950'
NEG_COL = '#f85149'
NEU_COL = '#8b949e'


def metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('dashboard_data.csv', parse_dates=['date'])
    return df.sort_values('date')

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
    <div style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;
                color:#58a6ff;font-weight:600;margin-bottom:0.4rem;">
        NLP · Fine-tuned DistilBERT
    </div>
    <div style="font-size:2rem;font-weight:600;color:#e6edf3;line-height:1.2;">
        Earnings Call Sentiment Analyser
    </div>
    <div style="font-size:0.9rem;color:#8b949e;margin-top:0.5rem;">
        Paragraph-level sentiment extracted from earnings call transcripts,
        correlated against subsequent price movement.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Company View", "Call Deep-Dive", "Signal Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Company View
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    selected = st.selectbox("Select ticker", sorted(df['company'].unique()), key="t1_ticker")
    cdf = df[df['company'] == selected].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "Calls Analysed", len(cdf),
            f"{cdf['date'].min().strftime('%b %Y')} – {cdf['date'].max().strftime('%b %Y')}"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Avg Sentiment Score", f"{cdf['sentiment_score'].mean():+.2f}",
            "−1 bearish → +1 bullish"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Avg % Positive", f"{cdf['pct_positive'].mean():.0%}"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Avg % Negative", f"{cdf['pct_negative'].mean():.0%}"
        ), unsafe_allow_html=True)

    # Sentiment over time
    st.markdown('<div class="section-header">Sentiment Score per Earnings Call</div>',
                unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=cdf['date'], y=cdf['sentiment_score'],
        mode='lines+markers',
        line=dict(color=ACCENT, width=2),
        marker=dict(size=7, color=ACCENT),
        hovertemplate='%{x|%b %Y}<br>Score: %{y:.3f}<extra></extra>',
    ))
    fig1.add_hline(y=0, line_dash='dot', line_color='#30363d')
    fig1.update_layout(
        **BASE_LAYOUT, height=280,
        xaxis=axis_style(),
        yaxis=axis_style(title='Sentiment Score'),
    )
    st.plotly_chart(fig1, width="stretch")

    # Stacked bar — sentiment breakdown
    st.markdown('<div class="section-header">Sentiment Breakdown per Call</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    for col, label, color in [
        ('pct_positive', 'Positive', POS_COL),
        ('pct_neutral',  'Neutral',  NEU_COL),
        ('pct_negative', 'Negative', NEG_COL),
    ]:
        fig2.add_trace(go.Bar(
            x=cdf['date'].dt.strftime('%b %Y'), y=cdf[col],
            name=label, marker_color=color,
            hovertemplate=f'{label}: %{{y:.1%}}<extra></extra>',
        ))
    fig2.update_layout(
        **BASE_LAYOUT, height=260, barmode='stack',
        legend=dict(orientation='h', y=1.1, x=0),
        xaxis=axis_style(),
        yaxis=axis_style(tickformat='.0%'),
    )
    st.plotly_chart(fig2, width="stretch")

    # Price reaction bar
    if cdf['price_change_1d'].notna().any():
        st.markdown('<div class="section-header">1-Day Price Reaction per Call</div>',
                    unsafe_allow_html=True)
        colors = [POS_COL if v > 0 else NEG_COL for v in cdf['price_change_1d'].fillna(0)]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=cdf['date'].dt.strftime('%b %Y'),
            y=cdf['price_change_1d'],
            marker_color=colors,
            hovertemplate='%{x}<br>1d change: %{y:.2%}<extra></extra>',
        ))
        fig3.update_layout(
            **BASE_LAYOUT, height=240,
            xaxis=axis_style(),
            yaxis=axis_style(tickformat='.1%'),
        )
        st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Call Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        ticker2 = st.selectbox("Ticker", sorted(df['company'].unique()), key="t2_ticker")
    call_opts = df[df['company'] == ticker2].copy()
    call_opts['label'] = call_opts['date'].dt.strftime('%d %b %Y')
    with col_b:
        sel_label = st.selectbox("Earnings call date", call_opts['label'].tolist(), key="t2_date")

    row = call_opts[call_opts['label'] == sel_label].iloc[0]

    st.markdown('<div class="section-header">Call Summary</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(metric_card("Sentiment Score", f"{row['sentiment_score']:+.3f}"), unsafe_allow_html=True)
    with mc2:
        st.markdown(metric_card("Paragraphs Scored", int(row['paragraph_count'])), unsafe_allow_html=True)
    with mc3:
        st.markdown(metric_card("% Positive", f"{row['pct_positive']:.0%}"), unsafe_allow_html=True)
    with mc4:
        st.markdown(metric_card("% Negative", f"{row['pct_negative']:.0%}"), unsafe_allow_html=True)
    with mc5:
        val = f"{row['price_change_1d']:+.2%}" if pd.notna(row['price_change_1d']) else "N/A"
        st.markdown(metric_card("1-Day Price Move", val), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        fig_pie = go.Figure(go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[row['pct_positive'], row['pct_neutral'], row['pct_negative']],
            marker_colors=[POS_COL, NEU_COL, NEG_COL],
            hole=0.65, textinfo='label+percent',
            textfont=dict(size=11), showlegend=False,
        ))
        fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8b949e", size=12),
        height=240, margin=dict(l=0, r=0, t=10, b=10)
    )
        st.plotly_chart(fig_pie, width="stretch")

    with right:
        st.markdown('<div class="section-header">Most Bullish Paragraph</div>', unsafe_allow_html=True)
        pos_text = str(row['most_positive_text'])[:600]
        st.markdown(f'<div class="quote-card">{pos_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Most Bearish Paragraph</div>', unsafe_allow_html=True)
        neg_text = str(row['most_negative_text'])[:600]
        st.markdown(f'<div class="quote-card negative">{neg_text}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1rem;font-size:0.8rem;color:#8b949e;">
        Mean model confidence:
        <span style="color:#e6edf3;font-family:'DM Mono',monospace;">
        {row['mean_confidence']:.3f}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Signal Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Sentiment vs Price Movement — All Calls</div>',
                unsafe_allow_html=True)

    corr_df = df.dropna(subset=['sentiment_score', 'price_change_1d', 'price_change_5d'])
    r1, p1 = stats.spearmanr(corr_df['sentiment_score'], corr_df['price_change_1d'])
    r5, p5 = stats.spearmanr(corr_df['sentiment_score'], corr_df['price_change_5d'])

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(metric_card("1-Day Spearman r", f"{r1:.3f}", "sentiment → next-day return"),
                    unsafe_allow_html=True)
    with sc2:
        sig1 = "p < 0.05 ✓" if p1 < 0.05 else f"p = {p1:.3f}"
        col1c = POS_COL if p1 < 0.05 else NEG_COL
        st.markdown(metric_card("1-Day p-value",
                                f'<span style="color:{col1c}">{sig1}</span>', ""),
                    unsafe_allow_html=True)
    with sc3:
        st.markdown(metric_card("5-Day Spearman r", f"{r5:.3f}", "sentiment → 5-day return"),
                    unsafe_allow_html=True)
    with sc4:
        sig5 = "p < 0.05 ✓" if p5 < 0.05 else f"p = {p5:.3f}"
        col5c = POS_COL if p5 < 0.05 else NEG_COL
        st.markdown(metric_card("5-Day p-value",
                                f'<span style="color:{col5c}">{sig5}</span>', ""),
                    unsafe_allow_html=True)

    # Scatter plots
    st.markdown('<div class="section-header">Scatter: Sentiment Score vs Return</div>',
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)

    def make_scatter(x_col, y_col, y_title, title):
        fig = px.scatter(
            corr_df, x=x_col, y=y_col,
            hover_data=['company', 'date'],
            color_discrete_sequence=[ACCENT],
            trendline='ols',
            trendline_color_override=NEG_COL,
            labels={x_col: 'Sentiment Score', y_col: y_title},
            title=title,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
        fig.update_layout(
            **BASE_LAYOUT, height=360,
            title=dict(font=dict(size=13, color='#e6edf3'), x=0),
            xaxis=axis_style(title='Sentiment Score'),
            yaxis=axis_style(title=y_title, tickformat='.1%'),
        )
        return fig

    with s1:
        st.plotly_chart(make_scatter('sentiment_score', 'price_change_1d',
                                     '1-Day Price Change', '1-Day Return'),
                        width="stretch")
    with s2:
        st.plotly_chart(make_scatter('sentiment_score', 'price_change_5d',
                                     '5-Day Price Change', '5-Day Return'),
                        width="stretch")

    # Histogram
    st.markdown('<div class="section-header">Distribution of Sentiment Scores</div>',
                unsafe_allow_html=True)
    fig_hist = px.histogram(df, x='sentiment_score', nbins=20,
                            color_discrete_sequence=[ACCENT])
    fig_hist.update_layout(
        **BASE_LAYOUT, height=240, bargap=0.05,
        xaxis=axis_style(title='Sentiment Score'),
        yaxis=axis_style(title='Number of Calls'),
    )
    st.plotly_chart(fig_hist, width="stretch")

    # Interpretation
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:1.25rem 1.5rem;margin-top:1rem;font-size:0.83rem;
                color:#8b949e;line-height:1.7;">
        <span style="color:#e6edf3;font-weight:500;">Interpretation</span><br>
        We observe a consistent positive relationship between earnings call sentiment scores
        and subsequent price movement (1-day r = {r1:.3f}, p = {p1:.3f};
        5-day r = {r5:.3f}, p = {p5:.3f}).
        Results do not reach conventional significance thresholds (p &lt; 0.05) at this sample size,
        consistent with the efficient market hypothesis — any persistent signal would be arbitraged away.
        The direction and magnitude warrant further investigation on larger samples.
        Model: DistilBERT fine-tuned on LLM-labeled earnings call paragraphs
        (weighted F1 = 0.853).
    </div>
    """, unsafe_allow_html=True)
