import torch
import pandas as pd
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
from scipy import stats
import yfinance as yf
from datetime import timedelta

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = '/content/drive/MyDrive/earnings-sentiment-model'
INPUT_PATH = 'kaggle_paragraphs_test.csv'
PARA_SCORES_PATH = 'kaggle_para_scores.csv'
DASHBOARD_PATH = 'kaggle_dashboard_data.csv'

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DAYS_BEFORE = 1   # trading days before call for baseline price
DAYS_AFTER_1D = 1 # trading days after call for short-term move
DAYS_AFTER_5D = 5 # trading days after call for medium-term move

# ── Model loading ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)
        for pred, prob in zip(predictions, probs):
            results.append({
                'predicted_label': ID2LABEL[pred],
                'prob_negative': prob[0],
                'prob_neutral':  prob[1],
                'prob_positive': prob[2],
                'confidence':    float(prob[pred]),
            })
    return results


def run_inference(para_df: pd.DataFrame) -> pd.DataFrame:
    print(f"Running inference on {len(para_df)} paragraphs...")
    predictions = predict_batch(para_df['text'].tolist())
    pred_df = pd.DataFrame(predictions)
    result = pd.concat([para_df.reset_index(drop=True), pred_df], axis=1)
    result.to_csv(PARA_SCORES_PATH, index=False)
    print(f"Paragraph scores saved to {PARA_SCORES_PATH}")
    return result


# ── Call-level aggregation ────────────────────────────────────────────────────
def aggregate_call_scores(para_df: pd.DataFrame) -> pd.DataFrame:
    agg = para_df.groupby(['company', 'date']).agg(
        pct_positive    =('predicted_label', lambda x: (x == 'positive').mean()),
        pct_negative    =('predicted_label', lambda x: (x == 'negative').mean()),
        pct_neutral     =('predicted_label', lambda x: (x == 'neutral').mean()),
        sentiment_score =('predicted_label',
                          lambda x: x.map({'negative': -1, 'neutral': 0, 'positive': 1}).mean()),
        mean_confidence =('confidence', 'mean'),
        paragraph_count =('text', 'count'),
        most_positive_text=('text', lambda x: x[
            para_df.loc[x.index, 'prob_positive'].idxmax()]),
        most_negative_text=('text', lambda x: x[
            para_df.loc[x.index, 'prob_negative'].idxmax()]),
    ).reset_index()
    return agg


# ── OHLC via yfinance ─────────────────────────────────────────────────────────
def fetch_ohlc(tickers: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Fetch daily OHLC for each ticker over a window that covers all call dates
    plus a buffer for forward price lookups.
    """
    start = (min(dates) - timedelta(days=10)).strftime('%Y-%m-%d')
    end   = (max(dates) + timedelta(days=10)).strftime('%Y-%m-%d')

    print(f"Fetching OHLC for {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when >1 ticker; normalise to long format
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].reset_index().melt(id_vars='Date', var_name='company', value_name='Close')
    else:
        # Single ticker — raw.columns are plain strings
        close = raw[['Close']].reset_index()
        close.columns = ['Date', 'Close']
        close['company'] = tickers[0]

    close = close.rename(columns={'Date': 'date'})
    close['date'] = pd.to_datetime(close['date']).dt.tz_localize(None)
    close = close.dropna(subset=['Close']).sort_values(['company', 'date'])
    return close


def get_price_at_offset(ticker_data: pd.DataFrame, call_date: pd.Timestamp, offset: int, direction: str) -> float:
    """
    Return the closing price offset trading days before (direction='before')
    or after (direction='after') call_date.
    """
    if direction == 'before':
        prices = ticker_data[ticker_data['date'] <= call_date]['Close']
        return float(prices.iloc[-offset]) if len(prices) >= offset else None
    else:
        prices = ticker_data[ticker_data['date'] > call_date]['Close']
        return float(prices.iloc[offset - 1]) if len(prices) >= offset else None


def compute_price_changes(call_df: pd.DataFrame) -> pd.DataFrame:
    call_df = call_df.copy()
    call_df['date'] = pd.to_datetime(call_df['date'])

    tickers = call_df['company'].unique().tolist()
    dates   = call_df['date'].tolist()

    ohlc = fetch_ohlc(tickers, dates)

    # Verify ticker coverage
    matched = set(tickers) & set(ohlc['company'].unique())
    missing = set(tickers) - matched
    if missing:
        print(f"Warning: no OHLC data for {len(missing)} tickers: {missing}")
    print(f"Ticker coverage: {len(matched)}/{len(tickers)}")

    # Pre-group for fast lookup
    grouped = {t: g.sort_values('date') for t, g in ohlc.groupby('company')}

    price_change_1d, price_change_5d = [], []

    for _, row in call_df.iterrows():
        ticker_data = grouped.get(row['company'])
        if ticker_data is None:
            price_change_1d.append(None)
            price_change_5d.append(None)
            continue

        prior  = get_price_at_offset(ticker_data, row['date'], DAYS_BEFORE,  'before')
        fwd_1d = get_price_at_offset(ticker_data, row['date'], DAYS_AFTER_1D, 'after')
        fwd_5d = get_price_at_offset(ticker_data, row['date'], DAYS_AFTER_5D, 'after')

        price_change_1d.append((fwd_1d - prior) / prior if prior and fwd_1d else None)
        price_change_5d.append((fwd_5d - prior) / prior if prior and fwd_5d else None)

    call_df['price_change_1d'] = price_change_1d
    call_df['price_change_5d'] = price_change_5d
    return call_df


# ── Correlation analysis ──────────────────────────────────────────────────────
def run_correlation(dashboard_df: pd.DataFrame):
    corr_df = dashboard_df.dropna(subset=['sentiment_score', 'price_change_1d', 'price_change_5d'])
    print(f"\nCorrelation analysis on {len(corr_df)} calls with price data:")

    r1, p1 = stats.spearmanr(corr_df['sentiment_score'], corr_df['price_change_1d'])
    r5, p5 = stats.spearmanr(corr_df['sentiment_score'], corr_df['price_change_5d'])

    print(f"  Sentiment vs 1-day price change:  r={r1:.3f}, p={p1:.3f}")
    print(f"  Sentiment vs 5-day price change:  r={r5:.3f}, p={p5:.3f}")

    sig_1d = "significant" if p1 < 0.05 else "not significant"
    sig_5d = "significant" if p5 < 0.05 else "not significant"
    print(f"  1-day correlation is {sig_1d} at p<0.05")
    print(f"  5-day correlation is {sig_5d} at p<0.05")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Load paragraphs
    para_df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(para_df)} paragraphs from "
          f"{para_df[['company','date']].drop_duplicates().__len__()} calls")

    # 2. Run inference
    scored_df = run_inference(para_df)

    # 3. Aggregate to call level
    print("\nAggregating call-level scores...")
    call_df = aggregate_call_scores(scored_df)
    print(f"Call-level sentiment summary:\n{call_df['sentiment_score'].describe()}")

    # 4. Fetch OHLC and compute price changes
    dashboard_df = compute_price_changes(call_df)
    dashboard_df.to_csv(DASHBOARD_PATH, index=False)
    print(f"\nDashboard data saved to {DASHBOARD_PATH}")
    print(f"Calls with 1d price data: {dashboard_df['price_change_1d'].notna().sum()}/{len(dashboard_df)}")
    print(f"Calls with 5d price data: {dashboard_df['price_change_5d'].notna().sum()}/{len(dashboard_df)}")

    # 5. Correlation analysis
    run_correlation(dashboard_df)
