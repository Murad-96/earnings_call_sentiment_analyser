"""DatasetBuilder for earnings_call dataset."""
from pathlib import Path
from typing import Optional

import datasets
import pandas as pd

from stocks import StockMarketAnalyzer
from transcripts import load_earnings_calls, load_stock_prices

_CITATION = """\
@data{TJE0D0_2021,
author = {Roozen, Dexter and Lelli, Francesco},
publisher = {DataverseNL},
title = {{Stock Values and Earnings Call Transcripts: a Sentiment Analysis Dataset}},
year = {2021},
version = {V1},
doi = {10.34894/TJE0D0},
url = {https://doi.org/10.34894/TJE0D0}
}
"""


_DESCRIPTION = """\
The dataset reports a collection of earnings call transcripts, the related stock prices, and the sector index In terms of volume, there is a total of 188 transcripts, 11970 stock prices, and 1196 sector index values. Furthermore, all of these data originated in the period 2016-2020 and are related to the NASDAQ stock market. Furthermore, the data collection was made possible by Yahoo Finance and Thomson Reuters Eikon. Specifically, Yahoo Finance enabled the search for stock values and Thomson Reuters Eikon provided the earnings call transcripts. Lastly, the dataset can be used as a benchmark for the evaluation of several NLP techniques to understand their potential for financial applications. Moreover, it is also possible to expand the dataset by extending the period in which the data originated following a similar procedure.
"""  # noqa: E501


_HOMEPAGE = "https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/TJE0D0"


_LICENSE = " CC0 1.0"


_URLS = {
    "transcripts": "./transcripts.zip",
    "stock_prices": "./stock_prices.zip",
    "transcript-sentiment": {
        "stock_prices": "./stock_prices.zip",
        "transcripts": "./transcripts.zip",
    },
}


class EarningsCallDataset(datasets.GeneratorBasedBuilder):
    """Stock Values and Earnings Call Transcripts - a Sentiment Analysis Dataset"""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="transcripts",
            version=VERSION,
            description="Raw Earnings Call Transcripts",
        ),
        datasets.BuilderConfig(
            name="stock_prices", version=VERSION, description="Raw Company Stock Prices"
        ),
        datasets.BuilderConfig(
            name="transcript-sentiment",
            version=VERSION,
            description="Paragraphs from Earnings Call Transcripts with Sentiment Labels",  # noqa: E501
        ),
    ]

    DEFAULT_CONFIG_NAME = "transcript-sentiment"

    def _info(self):
        supervised_keys = None
        if self.config.name == "transcripts":
            features = datasets.Features(
                {
                    "company": datasets.Value("string"),
                    "date": datasets.Value("date64"),
                    "transcript": datasets.Value("string"),
                }
            )
        elif self.config.name == "stock_prices":
            features = datasets.Features(
                {
                    "date": datasets.Value("date64"),
                    "open": datasets.Value("float32"),
                    "high": datasets.Value("float32"),
                    "low": datasets.Value("float32"),
                    "close": datasets.Value("float32"),
                    "adj_close": datasets.Value("float32"),
                    "volume": datasets.Value("int64"),
                    "company": datasets.Value("string"),
                }
            )
        elif self.config.name == "transcript-sentiment":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "company": datasets.Value("string"),
                    "date": datasets.Value("date64"),
                    "para_no": datasets.Value("int32"),
                }
            )
            supervised_keys = ("text", "label")
        else:
            raise ValueError(f"Unknown config name: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
            supervised_keys=supervised_keys,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS[self.config.name])

        if self.config.name == "transcript-sentiment":
            transcript_dir = Path(data_dir["transcripts"]) / "transcripts"
            stocks_dir = Path(data_dir["stock_prices"]) / "stock_prices"

            return [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "filepath": transcript_dir / "train.txt",
                        "split": "train",
                        "transcript_dir": transcript_dir,
                        "stock_prices_dir": stocks_dir,
                    },
                ),
                datasets.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "filepath": transcript_dir / "test.txt",
                        "split": "test",
                        "transcript_dir": transcript_dir,
                        "stock_prices_dir": stocks_dir,
                    },
                ),
            ]
        elif self.config.name == "transcripts":
            data_dir = Path(data_dir) / "transcripts"
            return [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "filepath": data_dir / "train.txt",
                        "split": "train",
                        "transcript_dir": data_dir,
                    },
                ),
                datasets.SplitGenerator(
                    name="test",
                    gen_kwargs={
                        "filepath": data_dir / "test.txt",
                        "split": "test",
                        "transcript_dir": data_dir,
                    },
                ),
            ]
        elif self.config.name == "stock_prices":
            data_dir = Path(data_dir) / "stock_prices"
            return [
                datasets.SplitGenerator(
                    name="train",
                    gen_kwargs={
                        "filepath": data_dir,
                        "split": "train",
                    },
                ),
            ]
        else:
            raise ValueError(f"Unknown config name: {self.config.name}")

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self,
        filepath: Path,
        split: str,
        transcript_dir: Optional[Path] = None,
        stock_prices_dir: Optional[Path] = None,
    ):
        if self.config.name == "transcript-sentiment":
            assert (
                transcript_dir is not None
            ), "transcript_dir must passed in as a parameter"
            assert (
                stock_prices_dir is not None
            ), "stock_prices_dir must passed in as a parameter"

            transcript_filepaths = [
                transcript_dir / p for p in filepath.read_text().splitlines()
            ]
            calls = list(load_earnings_calls(transcript_filepaths))

            companies = set(call.company for call in calls) | {"NASDAQ"}

            company_stocks_paths = [stock_prices_dir / f"{c}.csv" for c in companies]
            stock_prices = load_stock_prices(company_stocks_paths)
            market_analyzer = StockMarketAnalyzer(stock_prices)

            idx = 0

            for call in calls:
                call.set_sentiment(market_analyzer)
                for prompt in call.generate_prompts():
                    yield idx, prompt.to_dict()
                    idx += 1

        elif self.config.name == "transcripts":
            transcript_filepaths = [
                transcript_dir / p for p in filepath.read_text().splitlines()
            ]
            calls = load_earnings_calls(transcript_filepaths)

            for i, call in enumerate(calls):
                call.load_transcript()
                yield i, {
                    "company": call.company,
                    "date": call.date,
                    "transcript": call.transcript,
                }

        elif self.config.name == "stock_prices":
            i = 0
            for f in filepath.iterdir():
                company = f.stem
                df = pd.read_csv(f, parse_dates=["Date"])

                for _, dt, open, high, low, close, adj_close, vol in df.itertuples():
                    yield i, {
                        "date": dt,
                        "open": open,
                        "high": high,
                        "low": low,
                        "close": close,
                        "adj_close": adj_close,
                        "volume": vol,
                        "company": company,
                    }
                    i += 1
        else:
            raise ValueError(f"Unknown config name: {self.config.name}")
