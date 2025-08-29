from lm_config import configure_global_lm
from pipeline import KeywordExtractor

configure_global_lm()

if __name__ == "__main__":
    text = """A Story Event creates meaningful change in the life
    situation of a character that is expressed and experi¬
    enced in terms of a value and ACHIEVED THROUGH
    CONFLICT."""
    extractor = KeywordExtractor()
    response = extractor.run(text)

    print(response)
    # Prediction(
    #     reasoning="A story event is a pivotal moment that brings about significant change in the character's life, driven by their values and achieved through conflict.",
    #     keywords=['Story Event', 'change', 'character', 'life situation', 'value', 'conflict']
    # )
