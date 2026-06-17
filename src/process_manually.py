import pandas as pd
from pathlib import Path
import datetime

# Import the logic written by the dataset author
from stocks import StockMarketAnalyzer
from transcripts import load_stock_prices, EarningsCall

def load_manifest_paths(manifest_path: Path, base_dir: Path) -> set[Path]:
    """Reads train.txt or test.txt and converts relative paths to absolute local Paths."""
    if not manifest_path.exists():
        return set()
    
    # Read lines and resolve them against the base transcripts directory
    with open(manifest_path, "r") as f:
        relative_paths = f.read().splitlines()
        
    return {base_dir / p for p in relative_paths if p.strip()}

def generate_csv_splits():
    # 1. Define your data directories
    data_dir = Path("./data") 
    transcripts_dir = data_dir / "transcripts"
    stock_dir = data_dir / "stock_prices"
    
    train_txt = transcripts_dir / "train.txt"
    test_txt = transcripts_dir / "test.txt"
    
    # Find all stock files
    stock_files = list(stock_dir.glob("**/*.csv"))
    if not stock_files:
        # Fallback to general search if folder structure varies slightly
        stock_files = list(data_dir.glob("**/*.csv"))

    if not train_txt.exists() or not test_txt.exists():
        raise FileNotFoundError(
            f"Could not find train.txt or test.txt inside {transcripts_dir}. "
            "Please check your directory structure."
        )

    # 2. Parse the manifests to separate train and test file paths
    train_paths = load_manifest_paths(train_txt, transcripts_dir)
    test_paths = load_manifest_paths(test_txt, transcripts_dir)
    
    print(f"Manifests loaded: {len(train_paths)} training files, {len(test_paths)} testing files.")

    # 3. Initialize the Market Analyzer
    print("Loading stock prices...")
    stock_prices = load_stock_prices(stock_files)
    market_analyzer = StockMarketAnalyzer(stock_prices)

    # 4. Process splits independently
    splits = {"train": train_paths, "test": test_paths}
    
    for split_name, file_paths in splits.items():
        print(f"\nProcessing {split_name} split...")
        processed_records = []
        skipped_files = 0
        
        for f in file_paths:
            if not f.exists():
                # Handles minor case mismatch or missing files safely
                skipped_files += 1
                continue
                
            try:
                # Load the earnings call and calculate sentiment labels
                call = EarningsCall.from_file(f)
                call.set_sentiment(market_analyzer) 
                
                # Extract paragraph-level blocks
                for prompt in call.generate_prompts(): 
                    processed_records.append(prompt.to_dict())
            except (ValueError, IndexError, KeyError) as e:
                skipped_files += 1

        print(f" -> Loaded {len(file_paths) - skipped_files} calls (Skipped {skipped_files} invalid/missing files).")
        print(f" -> Generated {len(processed_records)} paragraph records.")

        if processed_records:
            # Turn into DataFrame and save to disk
            df = pd.DataFrame(processed_records)
            
            # Shuffling paragraphs within the split so sequential text blocks are randomized 
            # (highly recommended for model training pipelines)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            output_filename = f"processed_{split_name}_sentiment.csv"
            df.to_csv(output_filename, index=False)
            print(f" -> Saved to {output_filename}")
        else:
            print(f" -> ⚠️ No records generated for {split_name} split.")

    print("\n🎉 Success! Splitting completed using official manifests.")

if __name__ == "__main__":
    generate_csv_splits()