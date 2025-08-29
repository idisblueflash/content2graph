import dspy

lm = dspy.LM("openai/gpt-oss-20b",
             api_base="http://10.62.1.190:1234/v1",  # fixe me with a ENV KEY
             api_key="any non-empty string",
             temperature=0.2,
             max_tokens=512, )
dspy.configure(lm=lm)

math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

if __name__ == "__main__":
    print(result)
