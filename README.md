# Earnings Call Sentiment Analyser

Paragraph-level sentiment extraction from earnings call transcripts, correlated against subsequent stock price movement.

A DistilBERT model is fine-tuned on ~16,000 labeled paragraphs from 300 earnings calls. Labels are generated using GPT-4o-mini (LLM-assisted weak supervision) rather than hand-annotation. The model classifies each paragraph as **positive**, **neutral**, or **negative** with respect to the company's future financial outlook. Call-level sentiment scores are then correlated against 1-day and 5-day post-call price movement.

**[Live Dashboard →](https://earningscallsentimentanalyser-y4cuukgsofrznzhvz4amcb.streamlit.app)**

---

## Results

| Metric | Value |
|---|---|
| Weighted F1 | 0.853 |
| Negative F1 | 0.563 |
| Sentiment vs 1-day return (Spearman r) | 0.22 (p = 0.08) |
| Sentiment vs 5-day return (Spearman r) | 0.15 (p = 0.22) |

The positive but non-significant correlation is consistent with efficient market theory — any strong persistent signal would be priced in quickly. Results warrant further investigation on larger samples.

---

## Architecture

```
Raw earnings call transcripts
          ↓
Paragraph splitting + boilerplate removal
          ↓
GPT-4o-mini paragraph labeling (positive / neutral / negative)
          ↓
DistilBERT fine-tuning with class-weighted cross-entropy loss
          ↓
Inference on held-out transcripts
          ↓
Call-level sentiment aggregation
          ↓
OHLC price correlation + Streamlit dashboard
```

---

## Data

**Primary dataset:** [`jlh-ibm/earnings_call`](https://huggingface.co/datasets/jlh-ibm/earnings_call) on Hugging Face. Contains raw HTML earnings call transcripts and OHLC price data for a range of NASDAQ/NYSE tickers (2016–2020).

**Supplementary transcripts:** Additional earnings call transcripts sourced from [`Kaggle`](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts/data) to expand the inference dataset.

---

## Reproducing the Results

### Prerequisites

```bash
pip install transformers torch pandas scikit-learn datasets \
            openai beautifulsoup4 yfinance streamlit plotly scipy
```

You will also need an OpenAI API key for the labeling step (step 3).

---

### Step 1 — Download the raw data

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jlh-ibm/earnings_call",
    repo_type="dataset",
    local_dir="./earnings_data"
)
```

This downloads raw transcript files and OHLC price CSVs into `./earnings_data`.

---

### Step 2 — Process transcripts into paragraphs

Split raw transcripts into paragraphs and remove boilerplate (operator instructions, legal disclaimers, safe harbour statements):

```bash
python process_manually.py
python clean_boilerplate.py
```

`process_manually.py` reads the raw transcript files and produces `processed_train_sentiment.csv` and `processed_test_sentiment.csv`. `clean_boilerplate.py` filters those files and outputs `cleaned_train_sentiment.csv` and `cleaned_test_sentiment.csv`.

---

### Step 3 — Label paragraphs using GPT-4o-mini

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Then run:

```bash
python label_transcripts.py
```

This calls the OpenAI API to classify each paragraph as `positive`, `neutral`, or `negative` based on forward-looking financial outlook. Labels are saved with checkpointing every 100 rows — if interrupted, the script resumes from where it left off.

Approximate cost: **$0.30–0.50** for ~7,500 paragraphs using `gpt-4o-mini`.

---

### Step 4 — Fine-tune DistilBERT

Training is designed to run on a GPU. [Google Colab](https://colab.research.google.com) with a free T4 GPU works well (~15 minutes per run).

```bash
python finetune_distilbert.py
```

Key training configuration:

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Loss | Class-weighted cross-entropy |
| Max sequence length | 256 tokens |

The model is saved to `./earnings-sentiment-model` on completion.

---

### Step 5 — Run inference and aggregate call-level scores

```bash
python inference.py
python call_level_summaries.py
```

`inference.py` runs the fine-tuned model over the test transcripts and outputs `paragraph_scores.csv` with per-paragraph sentiment predictions and confidence scores.

`call_level_summaries.py` aggregates paragraph scores to call level, computes sentiment score per call, joins against OHLC data to compute 1-day and 5-day price changes, and outputs `dashboard_data.csv`.

---

### Step 6 — Run the dashboard

```bash
streamlit run dashboard.py
```

Requires `dashboard_data.csv` in the same directory.

---

## Project Structure

```
├── process_manually.py        # Raw transcript → paragraph splitting
├── clean_boilerplate.py       # Boilerplate filtering
├── label_transcripts.py       # GPT-4o-mini paragraph labeling
├── finetune_distilbert.py     # DistilBERT fine-tuning
├── inference.py               # Paragraph-level inference
├── call_level_summaries.py    # Call-level aggregation + OHLC join
├── dashboard.py               # Streamlit dashboard
├── transcripts.py             # Transcript parsing utilities
├── stocks.py                  # OHLC utilities
├── dashboard_data.csv         # Pre-computed results (test set)
└── README.md
```

---

## Key Design Decisions

**LLM-assisted labeling over manual annotation** — GPT-4o-mini labels 7,500 paragraphs in under an hour at minimal cost. Manual review of 100 sampled labels confirmed ~87% agreement with human judgment.

**Forward-looking outlook over linguistic sentiment** — Labels reflect whether a paragraph implies positive or negative expectations for future performance, not surface-level tone. This aligns labels with the downstream task of predicting price movement.

**Class-weighted loss for imbalanced data** — Negative paragraphs represent only ~5% of the dataset, reflecting a genuine property of earnings call language rather than a sampling artifact. Executives rarely express negative outlook directly. Class-weighted cross-entropy upweights the minority class during training.

**Paragraph-level scoring over document-level** — Scoring full transcripts as single documents loses the variation between sections. Paragraph-level scoring with call-level aggregation preserves this signal and enables the call deep-dive view in the dashboard.

---

## Limitations

- Negative class F1 of 0.56 reflects fundamental data scarcity — only ~375 unique negative training examples
- Correlation analysis is based on 38 held-out calls; results should be interpreted as preliminary
- Model trained on 2016–2020 data; domain drift may affect performance on more recent transcripts
- Labels are generated by an LLM and are not ground truth; inter-annotator agreement was not formally measured
