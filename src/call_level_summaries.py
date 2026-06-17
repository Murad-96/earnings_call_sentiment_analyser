# Compute call-level summaries
import pandas as pd
from pathlib import Path


def aggregate_call_scores(para_df: pd.DataFrame) -> pd.DataFrame:
    agg = para_df.groupby(['company', 'date']).agg(

        # Sentiment distribution
        pct_positive=('predicted_label', lambda x: (x == 'positive').mean()),
        pct_negative=('predicted_label', lambda x: (x == 'negative').mean()),
        pct_neutral=('predicted_label', lambda x: (x == 'neutral').mean()),

        # Sentiment score: negative=−1, neutral=0, positive=+1
        sentiment_score=(
            'predicted_label',
            lambda x: x.map({'negative': -1, 'neutral': 0, 'positive': 1}).mean()
        ),

        # Average model confidence
        mean_confidence=('confidence', 'mean'),

        # Paragraph count — low counts are less reliable
        paragraph_count=('text', 'count'),

        # Most positive and negative paragraphs for dashboard display
        most_positive_text=('text', lambda x: x[
            para_df.loc[x.index, 'prob_positive'].idxmax()
        ]),
        most_negative_text=('text', lambda x: x[
            para_df.loc[x.index, 'prob_negative'].idxmax()
        ]),

    ).reset_index()

    return agg


def get_prior_price(ticker_data: pd.DataFrame, call_date: pd.Timestamp):
    """Closing price on or immediately before the earnings call date."""
    prior = ticker_data[ticker_data['date'] <= call_date]['Close']
    return prior.iloc[-1] if not prior.empty else None


def get_forward_price(ticker_data: pd.DataFrame, call_date: pd.Timestamp, offset_days: int):
    """Closing price offset_days trading days after the earnings call date."""
    future = ticker_data[ticker_data['date'] > call_date]['Close']
    return future.iloc[offset_days - 1] if len(future) >= offset_days else None


def compute_price_changes(call_df: pd.DataFrame, ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each earnings call, look up the prior close and the 1-day and 5-day
    forward closes, then compute percentage price changes.
    """
    # Pre-group OHLC by ticker for fast lookup
    grouped = {ticker: group.sort_values('date') for ticker, group in ohlc_df.groupby('company')}

    price_change_1d = []
    price_change_5d = []

    for _, row in call_df.iterrows():
        ticker = row['company']
        call_date = pd.Timestamp(row['date'])
        ticker_data = grouped.get(ticker)

        if ticker_data is None:
            price_change_1d.append(None)
            price_change_5d.append(None)
            continue

        prior = get_prior_price(ticker_data, call_date)
        fwd_1d = get_forward_price(ticker_data, call_date, offset_days=1)
        fwd_5d = get_forward_price(ticker_data, call_date, offset_days=5)

        price_change_1d.append((fwd_1d - prior) / prior if prior and fwd_1d else None)
        price_change_5d.append((fwd_5d - prior) / prior if prior and fwd_5d else None)

    result = call_df.copy()
    result['price_change_1d'] = price_change_1d
    result['price_change_5d'] = price_change_5d
    return result


def main():
    print("Aggregating call-level summaries...")
    para_df = pd.read_csv('paragraph_scores.csv')
    call_df = aggregate_call_scores(para_df)
    call_df.to_csv('call_scores.csv', index=False)

    # Load OHLC data
    ohlc_df = pd.read_csv('ohlc_prices.csv')
    ohlc_df.rename(columns={"Date": "date", "Ticker": "company"}, inplace=True)
    ohlc_df['date'] = pd.to_datetime(ohlc_df['date'], dayfirst=True)
    call_df['date'] = pd.to_datetime(call_df['date'])

    # Verify ticker matching before proceeding
    call_companies = set(call_df['company'].unique())
    ohlc_tickers = set(ohlc_df['company'].unique())
    unmatched = call_companies - ohlc_tickers
    if unmatched:
        print(f"Warning: {len(unmatched)} companies in calls have no OHLC data: {unmatched}")
    print(f"Matched tickers: {len(call_companies & ohlc_tickers)}")

    # Compute forward price changes
    dashboard_df = compute_price_changes(call_df, ohlc_df)

    # Report coverage
    missing_1d = dashboard_df['price_change_1d'].isna().sum()
    missing_5d = dashboard_df['price_change_5d'].isna().sum()
    print(f"Price change coverage — 1d: {len(dashboard_df) - missing_1d}/{len(dashboard_df)}, "
          f"5d: {len(dashboard_df) - missing_5d}/{len(dashboard_df)}")

    dashboard_df.to_csv('dashboard_data.csv', index=False)
    print(f"Dashboard data ready: {len(dashboard_df)} earnings calls")


if __name__ == '__main__':
    main()