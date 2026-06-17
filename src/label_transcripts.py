import openai
import pandas as pd
import time
from pathlib import Path

client = openai.OpenAI(api_key="YOUR_API_KEY_HERE") # Replace with your actual API key

SYSTEM_PROMPT = """You are a financial analyst reading an earnings call transcript.
Label the following paragraph as positive, negative, or neutral based on whether 
it expresses a positive, negative, or uncertain outlook for the company's future 
financial performance.

Guidelines:
- positive: management expresses confidence, growth, strong guidance, beats expectations
- negative: management expresses concern, declining metrics, missed targets, cautious guidance, uncertainty about future performance
- neutral: factual statements with no clear directional implication, or mixed signals

Reply with exactly one word: positive, negative, or neutral."""

def label_paragraph(paragraph: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Paragraph: {paragraph}"}
                ],
                max_tokens=5,
                temperature=0  # deterministic labels
            )
            label = response.choices[0].message.content.strip().lower()
            if label in {"positive", "negative", "neutral"}:
                return label
            return "neutral"  # fallback for unexpected responses
        except Exception as e:
            if attempt == retries - 1:
                return "unknown"
            time.sleep(2 ** attempt)  # exponential backoff

def label_dataset(df: pd.DataFrame, output_path: Path, batch_size: int = 100):
    # Resume from checkpoint if it exists
    if output_path.exists():
        labeled = pd.read_csv(output_path)
        already_done = set(labeled['paragraph_id'])
        df = df[~df['paragraph_id'].isin(already_done)]
        print(f"Resuming from checkpoint: {len(already_done)} already labeled")
    else:
        labeled = pd.DataFrame()

    results = []
    
    for i, row in enumerate(df.itertuples()):
        label = label_paragraph(row.text)
        results.append({
            'paragraph_id': row.paragraph_id,
            'company': row.company,
            'date': row.date,
            'text': row.text,
            'llm_label': label
        })
        
        # Save checkpoint every batch_size rows
        if (i + 1) % batch_size == 0:
            checkpoint = pd.concat([labeled, pd.DataFrame(results)])
            checkpoint.to_csv(output_path, index=False)
            print(f"Checkpoint saved: {i + 1} labeled this session")
            time.sleep(1)  # gentle rate limiting
    
    # Final save
    final = pd.concat([labeled, pd.DataFrame(results)])
    final.to_csv(output_path, index=False)
    return final

if __name__ == "__main__":
    input_path_train = Path("cleaned_processed_test_sentiment.csv")
    output_path_train = Path("labelled_transcripts_train.csv")
    
    df_train = pd.read_csv(input_path_train)
    df_train['paragraph_id'] = df_train.index
    labeled_df = label_dataset(df_train, output_path_train)
    print(f"Labeling for train set complete. Total labeled paragraphs: {len(labeled_df)}")

    input_path_test = Path("cleaned_processed_test_sentiment.csv")
    output_path_test = Path("labelled_transcripts_test.csv")
    df_test = pd.read_csv(input_path_test)
    df_test['paragraph_id'] = df_test.index
    labeled_df_test = label_dataset(df_test, output_path_test)
    print(f"Labeling for test set complete. Total labeled paragraphs: {len(labeled_df_test)}")