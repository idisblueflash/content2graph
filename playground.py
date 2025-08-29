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
    # ExtractionResult(
    #   keywords=['Story Event', 'character', 'life situation', 'value', 'conflict'],
    #   reasoning="The text describes the concept of a Story Event, which is an event that
    #              creates meaningful change in a character's life situation. This change is achieved
    #              through conflict, and it is expressed and experienced in terms of a value.")

