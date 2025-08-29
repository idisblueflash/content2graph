import dspy


class ExtractKeywords(dspy.Signature):
    """Extract keywords from text."""

    text: str = dspy.InputField()
    keywords: list[str] = dspy.OutputField(
        desc="a list of keywords from the original text, the keyword should be noun")