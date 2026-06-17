import pandas as pd
import re
from pathlib import Path
import pickle


BOILERPLATE_SUBSTRINGS = [
    "THE INFORMATION CONTAINED IN EVENT TRANSCRIPTS IS A TEXTUAL REPRESENTATION",
    "In the conference calls upon which Event Transcripts are based",
    "PRELIMINARY TRANSCRIPT:",
    "EDITED TRANSCRIPT:",
    "Presentation  Operator",
    "Operator Instructions",
    "earnings call.  (Operator Instructions)",
    "thomson reuters",
    "copyright",
    "safe harbor",
    "forward-looking statements",
]

NUM_ROWS_TO_TAKE = 10621  # for testing, limit the number of rows to process


def is_boilerplate(text: str) -> bool:
    normalized = " ".join(text.split()).lower()
    if any(phrase.lower() in normalized for phrase in BOILERPLATE_SUBSTRINGS):
        return True
    if normalized.startswith("*"):
        return True
    if "welcome to" in normalized and "earnings call" in normalized:
        return True
    return False


def clean_paragraph(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r', '', text)
    #text = text.replace('--', '')
    text = re.sub(r' +', ' ', text)
    return text


def split_transcript(transcript: str, min_words: int = 40):
    """
    Split transcript into speaker turns using '--' as the boundary marker.
    Each speaker block starts with a 'Name -- Title' line followed by their text.
    We discard the speaker label line and keep only the spoken text.
    Yields (para_no, paragraph_text) tuples.
    """
    # Split on lines containing '--' which mark speaker boundaries
    turns = re.split(r'\n(?=[^\n]+--[^\n]+\n)', transcript)
 
    para_no = 0
    for turn in turns:
        lines = turn.strip().split('\n')
 
        # Discard the first line if it's a speaker label (contains '--')
        if lines and '--' in lines[0]:
            lines = lines[1:]
 
        # Join remaining lines as the spoken text
        text = ' '.join(lines).strip()
        cleaned = clean_paragraph(text)
 
        if len(cleaned.split()) < min_words:
            continue
        if is_boilerplate(cleaned):
            continue
 
        yield para_no, cleaned
        para_no += 1


def process_kaggle_transcripts(
    input_path: str,
    output_path: str,
    min_words: int = 40
) -> pd.DataFrame:

    print(f"Loading {input_path}...")
    #df = pd.read_csv(input_path)
    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} transcripts")

    # Normalise date to date-only string, strip timezone
    #df['date'] = pd.to_datetime(df['date'], utc=True).dt.date.astype(str)
    # normalize the date strings and parse them to timezone-aware datetimes
    # the source strings use 'ET' and 'a.m./p.m.' formats, and some rows embed extra header text
    DATE_PATTERN = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?' \
        r'\s+\d{1,2},\s*\d{4},?\s*\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)(?:\s*ET)?)',
        re.I,
    )

    def normalize_date_string(value):
        if isinstance(value, list):
            value = value[-1]
        if not isinstance(value, str):
            return value
        match = DATE_PATTERN.search(value)
        if match:
            value = match.group(1)
        value = value.replace(' ET', '')
        value = value.replace(' a.m.', ' AM').replace(' p.m.', ' PM')
        value = value.replace(' a.m', ' AM').replace(' p.m', ' PM')
        return value.strip()
    
    # Normalize and parse
    if 'date' in df.columns:
        df['date'] = df['date'].apply(normalize_date_string)
    else:
        df['date'] = df.iloc[:, 0].apply(normalize_date_string)

    parsed_dates = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = parsed_dates.dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date.astype(str)

    records = []
    skipped = 0

    for _, row in df.iterrows():
        ticker = row['ticker']
        date = row['date']
        transcript = row['transcript']

        if not isinstance(transcript, str) or not transcript.strip():
            skipped += 1
            continue

        for para_no, text in split_transcript(transcript, min_words=min_words):
            records.append({
                'company': ticker,
                'date': date,
                'para_no': para_no,
                'label': None,  # to be filled by LLM labeling
                'text': text,
            })

    result_df = pd.DataFrame(records)

    print(f"Skipped {skipped} empty transcripts")
    print(f"Generated {len(result_df)} paragraphs from "
          f"{result_df[['company', 'date']].drop_duplicates().__len__()} calls")
    print(f"Paragraphs per call — mean: {result_df.groupby(['company','date']).size().mean():.1f}, "
          f"median: {result_df.groupby(['company','date']).size().median():.1f}")

    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return result_df

def train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and test sets based on unique (company, date) pairs.
    Ensures that all paragraphs from the same call are in the same split.
    """
    unique_calls = df[['company', 'date']].drop_duplicates()
    train_calls = unique_calls.sample(frac=1 - test_size, random_state=random_state)
    test_calls = unique_calls.drop(train_calls.index)

    train_df = df.merge(train_calls, on=['company', 'date'], how='inner')
    test_df = df.merge(test_calls, on=['company', 'date'], how='inner')

    print(f"Train set: {len(train_df)} paragraphs from {len(train_calls)} calls")
    print(f"Test set: {len(test_df)} paragraphs from {len(test_calls)} calls")

    train_df.to_csv('kaggle_paragraphs_train.csv', index=False)
    test_df.to_csv('kaggle_paragraphs_test.csv', index=False)

    print("Saved train and test splits to 'kaggle_paragraphs_train.csv' and 'kaggle_paragraphs_test.csv'")

    return train_df, test_df


if __name__ == '__main__':
    process_kaggle_transcripts(
        input_path='motley-fool-data.pkl',
        output_path='kaggle_paragraphs.csv',
    )
    df = pd.read_csv('kaggle_paragraphs.csv')
    df = df[0:NUM_ROWS_TO_TAKE]  # Limit rows for testing
    train_test_split(df, test_size=0.2, random_state=42)