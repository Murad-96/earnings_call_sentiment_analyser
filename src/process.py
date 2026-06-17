from datasets import load_dataset

# Point load_dataset directly to your local python script file!
# This tells Hugging Face to use your local files to build the dataset.
dataset = load_dataset("./earnings_call.py", "transcript-sentiment")

# Print the generated splits
print(dataset)

# Optional: View an example of the processed text and labels
print("\nFirst training sample:")
print(dataset["train"][0])

# Optional: Save the cleanly processed data to a CSV so you don't need the scripts anymore
dataset["train"].to_csv("processed_train_sentiment.csv", index=False)
dataset["test"].to_csv("processed_test_sentiment.csv", index=False)
print("\nSaved processed splits to CSV files!")