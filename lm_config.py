import os
import dspy


def build_lm():
    """
    Create a DSPy LM from env vars with sensible defaults.
    Env:
      DSPY_MODEL        (e.g., openai/meta-llama-3.1-8b-instruct)
      DSPY_API_BASE     (e.g., http://10.62.1.190:1234/v1)
      DSPY_API_KEY      (any non-empty string for LM Studio)
      DSPY_TEMPERATURE  (float)
      DSPY_MAX_TOKENS   (int)
      DSPY_STREAM       (0/1)
    """
    model = os.getenv("DSPY_MODEL", "openai/meta-llama-3.1-8b-instruct")
    api_base = os.getenv("DSPY_API_BASE", "http://10.62.1.190:1234/v1")
    api_key = os.getenv("DSPY_API_KEY", "local-key")
    temperature = float(os.getenv("DSPY_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("DSPY_MAX_TOKENS", "512"))
    stream = bool(int(os.getenv("DSPY_STREAM", "0")))

    return dspy.LM(
        model,
        api_base=api_base,
        api_key=api_key,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def configure_global_lm(lm=None):
    if lm is None:
        lm = build_lm()
    dspy.configure(lm=lm)
    return lm
