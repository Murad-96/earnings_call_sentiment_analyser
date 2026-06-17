import torch
import pandas as pd
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer from the specified path
model_path = '/content/drive/MyDrive/earnings-sentiment-model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Paragraph level inference function
# Batch the paragraphs rather than scoring one at a time
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
        
        for j, (pred, prob) in enumerate(zip(predictions, probs)):
            results.append({
                'predicted_label': ID2LABEL[pred],
                'prob_negative': prob[0],
                'prob_neutral': prob[1],
                'prob_positive': prob[2],
                'confidence': prob[pred]
            })
    
    return results

# score full transcripts
from clean_boilerplate import clean_text_column
from transcripts import process_transcript

def score_transcript(
    company: str,
    date: str,
    raw_text: str
) -> pd.DataFrame:
    
    # Split into paragraphs and filter boilerplate
    paragraphs = []
    para_numbers = []
    
    for para_no, paragraph in process_transcript(raw_text, min_words=40):
        if clean_text_column(paragraph):
            paragraphs.append(paragraph)
            para_numbers.append(para_no)
    
    if not paragraphs:
        return pd.DataFrame()
    
    # Run inference
    predictions = predict_batch(paragraphs)
    
    # Assemble result dataframe
    df = pd.DataFrame(predictions)
    df['text'] = paragraphs
    df['para_no'] = para_numbers
    df['company'] = company
    df['date'] = date
    
    return df

# run across all transcripts in a directory
from pathlib import Path
from transcripts import EarningsCall

def _load_test_transcript_paths(
    transcripts_dir: Path,
    index_filename: str = "test.txt"
) -> list[Path]:
    index_path = transcripts_dir / index_filename
    if not index_path.exists():
        raise FileNotFoundError(f"Transcript index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    transcript_paths = []
    for relative_path in lines:
        path = transcripts_dir / relative_path
        if path.exists():
            transcript_paths.append(path)
        else:
            print(f"Warning: listed transcript not found, skipping: {path}")

    return transcript_paths


def score_all_transcripts(
    transcripts_dir: Path,
    test_list_filename: str = "test.txt"
) -> pd.DataFrame:
    all_results = []
    transcript_files = _load_test_transcript_paths(transcripts_dir, test_list_filename)
    
    print(f"Scoring {len(transcript_files)} transcripts from {test_list_filename}...")
    
    for i, path in enumerate(transcript_files):
        try:
            call = EarningsCall.from_file(path)
            call.load_transcript()
            
            result = score_transcript(
                company=call.company,
                date=call.date.strftime('%Y-%m-%d'),
                raw_text=call.transcript
            )
            
            if not result.empty:
                all_results.append(result)
                
        except Exception as e:
            print(f"Skipped {path.name}: {e}")
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(transcript_files)}")
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

if __name__ == "__main__":
    para_df = score_all_transcripts(Path('/content/drive/MyDrive/earnings_call_sentiment_analyser/earnings_data/data/transcripts'))
    para_df.to_csv('paragraph_scores.csv', index=False)
    print(f"Scored {len(para_df)} paragraphs across {para_df['company'].nunique()} companies")

# # Compute call-level summaries
# def aggregate_call_scores(para_df: pd.DataFrame) -> pd.DataFrame:
#     agg = para_df.groupby(['company', 'date']).agg(
        
#         # Sentiment distribution
#         pct_positive=('predicted_label', lambda x: (x == 'positive').mean()),
#         pct_negative=('predicted_label', lambda x: (x == 'negative').mean()),
#         pct_neutral=('predicted_label', lambda x: (x == 'neutral').mean()),
        
#         # Mean confidence-weighted sentiment score
#         # Maps negative=−1, neutral=0, positive=+1
#         sentiment_score=(
#             'predicted_label',
#             lambda x: x.map({'negative': -1, 'neutral': 0, 'positive': 1}).mean()
#         ),
        
#         # Average model confidence
#         mean_confidence=('confidence', 'mean'),
        
#         # Paragraph count — low counts are less reliable
#         paragraph_count=('text', 'count'),
        
#         # Most negative and most positive paragraphs for dashboard display
#         most_positive_text=('text', lambda x: x[
#             para_df.loc[x.index, 'prob_positive'].idxmax()
#         ]),
#         most_negative_text=('text', lambda x: x[
#             para_df.loc[x.index, 'prob_negative'].idxmax()
#         ]),
        
#     ).reset_index()
    
#     return agg

# call_df = aggregate_call_scores(para_df)
# call_df.to_csv('call_scores.csv', index=False)

# # Load your existing OHLC data
# ohlc_df = pd.read_csv('ohlc_prices.csv')
# ohlc_df['date'] = pd.to_datetime(ohlc_df['date'])
# call_df['date'] = pd.to_datetime(call_df['date'])

# # Join on company + date
# dashboard_df = call_df.merge(
#     ohlc_df,
#     on=['company', 'date'],
#     how='left'
# )

# # Compute price change metrics
# dashboard_df['price_change_1d'] = (
#     dashboard_df['close_1d'] - dashboard_df['open']
# ) / dashboard_df['open']

# dashboard_df['price_change_5d'] = (
#     dashboard_df['close_5d'] - dashboard_df['open']
# ) / dashboard_df['open']

# dashboard_df.to_csv('dashboard_data.csv', index=False)
# print(f"Dashboard data ready: {len(dashboard_df)} earnings calls")