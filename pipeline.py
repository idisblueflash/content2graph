from dataclasses import dataclass
import dspy
from signatures import ExtractKeywords


@dataclass
class ExtractionResult:
    keywords: list[str]
    reasoning: str | None = None


class KeywordExtractor:
    """Thin wrapper around DSPy module to keep your app code LLM-agnostic."""

    def __init__(self, use_cot: bool = True):
        Module = dspy.ChainOfThought if use_cot else dspy.Predict
        self._module = Module(ExtractKeywords)

    def run(self, text: str) -> ExtractionResult:
        pred = self._module(text=text)
        # DSPy predictions often expose .keywords and optional .reasoning
        keywords = getattr(pred, "keywords", []) or []
        reasoning = getattr(pred, "reasoning", None)
        return ExtractionResult(keywords=keywords, reasoning=reasoning)
