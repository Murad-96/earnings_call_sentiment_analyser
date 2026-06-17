import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)

# 1. Custom Dataset Wrapper for PyTorch/Transformers
class EarningsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# 2. Evaluation Metrics Setup
def compute_metrics(eval_pred):
    """Computes accuracy, weighted F1, and prints a confusion matrix."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Standard metrics requested
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    print("\n📊 Evaluation Confusion Matrix:")
    print(f"TN: {cm[0][0]:<5} FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]:<5} TP: {cm[1][1]}\n")
    
    return {
        'accuracy': acc,
        'f1_weighted': f1
    }

def main():
    # 3. Load the data
    print("Loading cleaned dataset CSVs...")
    train_df = pd.read_csv("cleaned_processed_train_sentiment.csv")
    test_df = pd.read_csv("cleaned_processed_test_sentiment.csv")

    # 4. Map string labels to 0/1 integers
    # positive -> 1, negative -> 0
    label_mapping = {"positive": 1, "negative": 0}
    train_labels = train_df['label'].map(label_mapping).tolist()
    test_labels = test_df['label'].map(label_mapping).tolist()

    # Extract text columns
    train_texts = train_df['text'].astype(str).tolist()
    test_texts = test_df['text'].astype(str).tolist()

    # 5. Tokenize Paragraphs
    print("Initializing DistilBERT Tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Padding and truncation ensure uniform lengths for batch processing
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    # Wrap in PyTorch Dataset objects
    train_dataset = EarningsDataset(train_encodings, train_labels)
    test_dataset = EarningsDataset(test_encodings, test_labels)

    # 6. Initialize Model for 2 classes (Positive/Negative)
    print("Downloading DistilBERT base weights...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=2
    )

    # 7. Define Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',          # where to save models/checkpoints
        num_train_epochs=3,              # 3-4 epochs
        per_device_train_batch_size=16,  # Batch size 16 as requested
        per_device_eval_batch_size=16,   
        warmup_steps=100,                # standard learning rate scheduling warmup
        weight_decay=0.01,               # regularization to prevent overfitting
        logging_dir='./logs',            
        logging_steps=50,
        eval_strategy="epoch",           # evaluate at the end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,     # keeps your optimal checkpoint
        metric_for_best_model="f1_weighted"
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 9. Train the Model!
    print("🚀 Starting training pipeline...")
    trainer.train()

    # 10. Final evaluation run
    print("\n🎉 Training complete! Running final evaluation on test split...")
    final_metrics = trainer.evaluate()
    print("\nFinal Metrics Summary:")
    print(f"Accuracy: {final_metrics['eval_accuracy']:.4f}")
    print(f"Weighted F1: {final_metrics['eval_f1_weighted']:.4f}")

if __name__ == "__main__":
    main()