import datetime
from enum import Enum
from pathlib import Path
import bs4
from typing import Generator, Iterable, Optional
from dataclasses import dataclass


from stocks import StockMarketAnalyzer, load_stock_prices


def clean_paragraph(paragraph: str):
    return paragraph.strip().replace("\n", " ").replace("\r", "").replace("--", "")


def process_transcript(transcript: str, min_words: int = 30):
    paragraphs = transcript.split("\n\n")
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.split()) > min_words:
            yield i, clean_paragraph(paragraph)


class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


@dataclass
class EarningsPrompt:
    company: str
    date: datetime.datetime
    para_no: int
    label: Sentiment
    text: str

    def to_dict(self):
        return {
            "company": self.company,
            "date": self.date,
            "para_no": self.para_no,
            "label": self.label,
            "text": self.text,
        }


@dataclass
class EarningsCall:
    company: str
    date: datetime.datetime
    file_path: Path
    sentiment: Optional[Sentiment] = None
    transcript: Optional[str] = None

    def load_transcript(self):
        if self.transcript is None:
            self.transcript = bs4.BeautifulSoup(
                self.file_path.read_text(), "html.parser"
            ).text

    def generate_prompts(self, min_words: int = 30):
        if self.sentiment is None:
            raise ValueError("EarningsCall sentiment must be set")
        if self.transcript is None:
            self.load_transcript()

        for para_no, paragraph in process_transcript(
            self.transcript, min_words=min_words
        ):
            yield EarningsPrompt(
                company=self.company,
                date=self.date,
                para_no=para_no,
                label=self.sentiment,
                text=paragraph,
            )

    def set_sentiment(
        self,
        market_analyzer: StockMarketAnalyzer,
        days_before: int = 1,
        days_after: int = 1,
    ):
        beats_market = market_analyzer.beats_market(
            self.company, self.date, days_before=days_before, days_after=days_after
        )
        self.sentiment = Sentiment.positive if beats_market else Sentiment.negative

    @classmethod
    def from_file(cls, path: Path):
        """
        Given a path to an earnings call transcript file, extracts the company name, date,
        and file path and returns an EarningsCall object containing this information.

        Args:
            path (Path): The path to the earnings call transcript file.

        Returns:
            EarningsCall: An object containing the company name, date, and file path.
        """

        company = path.parent.stem
        date = datetime.datetime.strptime(path.stem[: -(1 + len(company))], "%Y-%b-%d")

        return cls(company=company, date=date, file_path=path)


def load_earnings_calls(files: Iterable[Path]) -> Generator[EarningsCall, None, None]:
    return (EarningsCall.from_file(f) for f in files)


def process_earnings_calls(
    market_data_directory: Path, days_before: int = 1, days_after: int = 1
) -> Generator[EarningsCall, None, None]:
    stock_prices = load_stock_prices(market_data_directory.glob("**/*.csv"))
    earnings_calls = load_earnings_calls(market_data_directory.glob("**/*.txt"))
    market_analyzer = StockMarketAnalyzer(stock_prices)

    for call in earnings_calls:
        call.set_sentiment(market_analyzer, days_before, days_after)
        yield call


def generate_prompts(calls: Iterable[EarningsCall], min_words: int = 30):
    for call in calls:
        yield from call.generate_prompts(min_words=min_words)
