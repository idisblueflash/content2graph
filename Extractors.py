import dspy


class ExtractNodes(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    keywords: list[str] = dspy.OutputField(
        desc="a list of keywords from the original text")


lm = dspy.LM("openai/meta-llama-3.1-8b-instruct",
             api_base="http://10.62.1.190:1234/v1",  # fixe me with a ENV KEY
             api_key="any non-empty string",
             stream=False,
             temperature=0.3,
             max_tokens=512, )
dspy.configure(lm=lm)

module = dspy.ChainOfThought(ExtractNodes)

if __name__ == "__main__":
    text = """A Story Event creates meaningful change in the life
    situation of a character that is expressed and experi¬
    enced in terms of a value and ACHIEVED THROUGH
    CONFLICT."""
    response = module(text=text)

    print(response)
    # Prediction(
    #     reasoning="A story event is a pivotal moment that brings about significant change in the character's life, driven by their values and achieved through conflict.",
    #     keywords=['Story Event', 'change', 'character', 'life situation', 'value', 'conflict']
    # )
