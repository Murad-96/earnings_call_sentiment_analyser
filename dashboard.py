import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Sentiment Analyser",
    page_icon="📈",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    /* Metric cards */
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
    .metric-delta {
        font-size: 0.78rem;
        margin-top: 0.4rem;
        color: #8b949e;
    }

    /* Quote cards */
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

    /* Section headers */
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

    /* Streamlit overrides */
    .stSelectbox > div > div { background: #161b22; border-color: #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #21262d; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #8b949e;
        padding: 0.6rem 1.2rem;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #e6edf3;
        border-bottom: 2px solid #58a6ff;
    }
    div[data-testid="metric-container"] { display: none; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8b949e', size=12),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
    margin=dict(l=10, r=10, t=30, b=10),
)

ACCENT  = '#58a6ff'
POS_COL = '#3fb950'
NEG_COL = '#f85149'
NEU_COL = '#8b949e'


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('dashboard_data.csv', parse_dates=['date'])
    df = df.sort_values('date')
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                color:#58a6ff; font-weight:600; margin-bottom:0.4rem;">
        NLP · Fine-tuned DistilBERT
    </div>
    <div style="font-size:2rem; font-weight:600; color:#e6edf3; line-height:1.2;">
        Earnings Call Sentiment Analyser
    </div>
    <div style="font-size:0.9rem; color:#8b949e; margin-top:0.5rem;">
        Paragraph-level sentiment extracted from earnings call transcripts,
        correlated against subsequent price movement.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Company View", "Call Deep-Dive", "Signal Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Company View
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    companies = sorted(df['company'].unique())
    selected = st.selectbox("Select ticker", companies, key="company_select")
    company_df = df[df['company'] == selected].copy()

    # Summary metrics
    latest = company_df.iloc[-1]
    avg_score = company_df['sentiment_score'].mean()

    col1, col2, col3, col4 = st.columns(4)

    def metric_card(label, value, delta=None):
        delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>"""

    with col1:
        st.markdown(metric_card(
            "Calls Analysed",
            len(company_df),
            f"{company_df['date'].min().strftime('%b %Y')} – {company_df['date'].max().strftime('%b %Y')}"
        ), unsafe_allow_html=True)

    with col2:
        score_fmt = f"{avg_score:+.2f}"
        st.markdown(metric_card("Avg Sentiment Score", score_fmt, "range: −1 (bearish) to +1 (bullish)"),
                    unsafe_allow_html=True)

    with col3:
        pct_pos = company_df['pct_positive'].mean()
        st.markdown(metric_card("Avg % Positive Paragraphs", f"{pct_pos:.0%}"), unsafe_allow_html=True)

    with col4:
        pct_neg = company_df['pct_negative'].mean()
        st.markdown(metric_card("Avg % Negative Paragraphs", f"{pct_neg:.0%}"), unsafe_allow_html=True)

    # Sentiment over time
    st.markdown('<div class="section-header">Sentiment Score per Earnings Call</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=company_df['date'],
        y=company_df['sentiment_score'],
        mode='lines+markers',
        line=dict(color=ACCENT, width=2),
        marker=dict(size=7, color=ACCENT),
        name='Sentiment Score',
        hovertemplate='%{x|%b %Y}<br>Score: %{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='#30363d')
    fig.update_layout(**PLOTLY_THEME, height=280,
                      yaxis=dict(**PLOTLY_THEME['yaxis'], title='Sentiment Score'))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment breakdown stacked bar
    st.markdown('<div class="section-header">Sentiment Breakdown per Call</div>', unsafe_allow_html=True)

    fig2 = go.Figure()
    for col, label, color in [
        ('pct_positive', 'Positive', POS_COL),
        ('pct_neutral',  'Neutral',  NEU_COL),
        ('pct_negative', 'Negative', NEG_COL),
    ]:
        fig2.add_trace(go.Bar(
            x=company_df['date'].dt.strftime('%b %Y'),
            y=company_df[col],
            name=label,
            marker_color=color,
            hovertemplate=f'{label}: %{{y:.1%}}<extra></extra>',
        ))
    fig2.update_layout(**PLOTLY_THEME, height=260, barmode='stack',
                       legend=dict(orientation='h', y=1.1, x=0),
                       yaxis=dict(**PLOTLY_THEME['yaxis'], tickformat='.0%'))
    st.plotly_chart(fig2, use_container_width=True)

    # Price change vs sentiment
    has_price = company_df['price_change_1d'].notna().any()
    if has_price:
        st.markdown('<div class="section-header">Sentiment vs Price Reaction</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        colors = [POS_COL if s > 0 else NEG_COL for s in company_df['price_change_1d'].fillna(0)]
        fig3.add_trace(go.Bar(
            x=company_df['date'].dt.strftime('%b %Y'),
            y=company_df['price_change_1d'],
            marker_color=colors,
            name='1-day price change',
            hovertemplate='%{x}<br>1d change: %{y:.2%}<extra></extra>',
        ))
        fig3.update_layout(**PLOTLY_THEME, height=240,
                           yaxis=dict(**PLOTLY_THEME['yaxis'], tickformat='.1%'))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Call Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        ticker2 = st.selectbox("Ticker", sorted(df['company'].unique()), key="dive_ticker")

    call_options = df[df['company'] == ticker2].copy()
    call_options['label'] = call_options['date'].dt.strftime('%d %b %Y')

    with col_b:
        selected_label = st.selectbox("Earnings call date", call_options['label'].tolist(), key="dive_date")

    row = call_options[call_options['label'] == selected_label].iloc[0]

    # Call metrics
    st.markdown('<div class="section-header">Call Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(metric_card("Sentiment Score", f"{row['sentiment_score']:+.3f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Paragraphs Scored", int(row['paragraph_count'])), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("% Positive", f"{row['pct_positive']:.0%}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("% Negative", f"{row['pct_negative']:.0%}"), unsafe_allow_html=True)
    with c5:
        delta_1d = row['price_change_1d']
        val = f"{delta_1d:+.2%}" if pd.notna(delta_1d) else "N/A"
        st.markdown(metric_card("1-Day Price Move", val), unsafe_allow_html=True)

    # Sentiment donut
    st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        fig_pie = go.Figure(go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[row['pct_positive'], row['pct_neutral'], row['pct_negative']],
            marker_colors=[POS_COL, NEU_COL, NEG_COL],
            hole=0.65,
            textinfo='label+percent',
            textfont=dict(size=11),
            showlegend=False,
        ))
        fig_pie.update_layout(**PLOTLY_THEME, height=240, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Most Bullish Paragraph</div>', unsafe_allow_html=True)
        pos_text = str(row['most_positive_text'])[:600]
        st.markdown(f'<div class="quote-card">{pos_text}{"…" if len(str(row["most_positive_text"])) > 600 else ""}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="section-header">Most Bearish Paragraph</div>', unsafe_allow_html=True)
        neg_text = str(row['most_negative_text'])[:600]
        st.markdown(f'<div class="quote-card negative">{neg_text}{"…" if len(str(row["most_negative_text"])) > 600 else ""}</div>',
                    unsafe_allow_html=True)

    # Model confidence
    st.markdown(f"""
    <div style="margin-top:1rem; font-size:0.8rem; color:#8b949e;">
        Mean model confidence: <span style="color:#e6edf3; font-family:'DM Mono', monospace;">
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

    # Correlation stat cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(metric_card("1-Day Spearman r", f"{r1:.3f}", "sentiment → next-day return"), unsafe_allow_html=True)
    with mc2:
        sig1 = "p < 0.05 ✓" if p1 < 0.05 else f"p = {p1:.3f}"
        col = POS_COL if p1 < 0.05 else NEG_COL
        st.markdown(metric_card("1-Day p-value",
                                f'<span style="color:{col}">{sig1}</span>', ""), unsafe_allow_html=True)
    with mc3:
        st.markdown(metric_card("5-Day Spearman r", f"{r5:.3f}", "sentiment → 5-day return"), unsafe_allow_html=True)
    with mc4:
        sig5 = "p < 0.05 ✓" if p5 < 0.05 else f"p = {p5:.3f}"
        col5 = POS_COL if p5 < 0.05 else NEG_COL
        st.markdown(metric_card("5-Day p-value",
                                f'<span style="color:{col5}">{sig5}</span>', ""), unsafe_allow_html=True)

    # Scatter plots
    st.markdown('<div class="section-header">Scatter: Sentiment Score vs Return</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)

    def scatter_with_trendline(x, y, xlabel, ylabel, title):
        fig = px.scatter(
            corr_df, x=x, y=y,
            hover_data=['company', 'date'],
            color_discrete_sequence=[ACCENT],
            trendline='ols',
            trendline_color_override=NEG_COL,
            labels={x: xlabel, y: ylabel},
            title=title,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
        fig.update_layout(**PLOTLY_THEME, height=360,
                          yaxis=dict(**PLOTLY_THEME['yaxis'], tickformat='.1%'),
                          title=dict(font=dict(size=13, color='#e6edf3'), x=0))
        return fig

    with s1:
        st.plotly_chart(
            scatter_with_trendline('sentiment_score', 'price_change_1d',
                                   'Sentiment Score', '1-Day Price Change', '1-Day Return'),
            use_container_width=True
        )
    with s2:
        st.plotly_chart(
            scatter_with_trendline('sentiment_score', 'price_change_5d',
                                   'Sentiment Score', '5-Day Price Change', '5-Day Return'),
            use_container_width=True
        )

    # Distribution of sentiment scores
    st.markdown('<div class="section-header">Distribution of Call-Level Sentiment Scores</div>',
                unsafe_allow_html=True)

    fig_hist = px.histogram(
        df, x='sentiment_score', nbins=20,
        color_discrete_sequence=[ACCENT],
    )
    fig_hist.update_layout(**PLOTLY_THEME, height=240,
                           xaxis=dict(**PLOTLY_THEME['xaxis'], title='Sentiment Score'),
                           yaxis=dict(**PLOTLY_THEME['yaxis'], title='Number of Calls'),
                           bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Interpretation note
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #21262d; border-radius:8px;
                padding:1.25rem 1.5rem; margin-top:1rem; font-size:0.83rem;
                color:#8b949e; line-height:1.7;">
        <span style="color:#e6edf3; font-weight:500;">Interpretation</span><br>
        We observe a consistent positive relationship between earnings call sentiment scores
        and subsequent price movement (1-day r = {r1:.3f}, p = {p1:.3f};
        5-day r = {r5:.3f}, p = {p5:.3f}).
        Results do not reach conventional significance thresholds (p &lt; 0.05) at this sample size,
        consistent with the efficient market hypothesis — any persistent signal would be
        arbitraged away. The direction and magnitude warrant further investigation on larger samples.
        Model: DistilBERT fine-tuned on LLM-labeled earnings call paragraphs (n = 7,500,
        weighted F1 = 0.853).
    </div>
    """, unsafe_allow_html=True)