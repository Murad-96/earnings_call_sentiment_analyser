import re
import pandas as pd
from pathlib import Path

# Explicit boilerplate phrases to drop (Exact or heavy substring matches)
BOILERPLATE_SUBSTRINGS = [
    "THE INFORMATION CONTAINED IN EVENT TRANSCRIPTS IS A TEXTUAL REPRESENTATION",
    "In the conference calls upon which Event Transcripts are based",
    "PRELIMINARY TRANSCRIPT:",
    "EDITED TRANSCRIPT:",
    "Presentation  Operator",
    "Operator Instructions",
    "earnings call.  (Operator Instructions)",
]

def clean_text_column(text: str) -> bool:
    """
    Returns False if the text matches any boilerplate pattern, meaning it should be dropped.
    Returns True if the text is clean and should be kept.
    """
    if not isinstance(text, str):
        return False
        
    # Standardize whitespace and remove hidden newlines for easier matching
    normalized_text = " ".join(text.split()).strip()
    
    if not normalized_text:
        return False

    # Rule 1: Check against our known boilerplate substrings
    for boilerplate in BOILERPLATE_SUBSTRINGS:
        if boilerplate.lower() in normalized_text.lower():
            return False
            
    # Rule 2: Remove rows where the text starts with a star (*)
    # This catches executive introductions like "* Marilyn Mora..."
    if normalized_text.startswith("*"):
        return False
        
    # Rule 3: Catch any lingering variations of operator introductions 
    # (e.g., "Greetings and welcome to [Company Name]'s earnings call")
    if "welcome to" in normalized_text.lower() and "earnings call" in normalized_text.lower():
        return False

    return True

def clean_csv_files():
    # Target the files we generated previously
    files_to_clean = ["processed_train_sentiment.csv", "processed_test_sentiment.csv"]
    
    for filename in files_to_clean:
        file_path = Path(filename)
        
        if not file_path.exists():
            print(f"⚠️ Could not find {filename}, skipping...")
            continue
            
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(file_path)
        initial_rows = len(df)
        
        # The 4th column is 'text' based on your previous schema
        # We create a boolean mask using our cleaning function
        keep_mask = df['text'].apply(clean_text_column)
        
        # Filter the dataframe
        df_cleaned = df[keep_mask].reset_index(drop=True)
        dropped_rows = initial_rows - len(df_cleaned)
        
        # Save back to a new file so you don't overwrite your raw progress
        output_filename = f"cleaned_{filename}"
        df_cleaned.to_csv(output_filename, index=False)
        
        print(f" -> Removed {dropped_rows} boilerplate paragraphs.")
        print(f" -> Kept {len(df_cleaned)} clean paragraphs.")
        print(f" -> Saved to {output_filename}")

if __name__ == "__main__":
    clean_csv_files()