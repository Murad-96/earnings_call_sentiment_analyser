import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Label mapping
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Load labeled data from CSV files
hf_train_df = pd.read_csv("labelled_transcripts_train.csv")
kaggle_train_df = pd.read_csv("kaggle_paragraphs_train_labeled.csv")
train_df = pd.concat([hf_train_df, kaggle_train_df], ignore_index=True)
hf_val_df = pd.read_csv("labelled_transcripts_test.csv")
kaggle_val_df = pd.read_csv("kaggle_paragraphs_test_labeled.csv")
val_df = pd.concat([hf_val_df, kaggle_val_df], ignore_index=True)

train_df = train_df[train_df['llm_label'] != 'unknown'].reset_index(drop=True)
val_df = val_df[val_df['llm_label'] != 'unknown'].reset_index(drop=True)

train_df['label'] = train_df['llm_label'].map(LABEL2ID)
val_df['label'] = val_df['llm_label'].map(LABEL2ID)

# Tokenisation — upfront, no HuggingFace datasets library involved
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(
    train_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=256
)
val_encodings = tokenizer(
    val_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

# Custom PyTorch Dataset — no dependency on HuggingFace datasets formatting
class EarningsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EarningsDataset(train_encodings, train_df['label'].tolist())
val_dataset = EarningsDataset(val_encodings, val_df['label'].tolist())

# Class weights
def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df['label'].value_counts().sort_index()
    total = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float)

class_weights = compute_class_weights(train_df)
print("Class weights:", {ID2LABEL[i]: w.item() for i, w in enumerate(class_weights)})

# Weighted Trainer
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics — per-class F1 is what matters here
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, predictions,
        target_names=["negative", "neutral", "positive"],
        output_dict=True
    )
    return {
        "accuracy": report["accuracy"],
        "f1_negative": report["negative"]["f1-score"],
        "f1_neutral": report["neutral"]["f1-score"],
        "f1_positive": report["positive"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./earnings-sentiment-model",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=50,
    fp16=True,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    save_path = '/content/drive/MyDrive/earnings-sentiment-model'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")