# Prompt Engineering

A prompt is the input given to a language model. Zero-shot prompting asks the model to perform a task with instructions only, while few-shot prompting adds a handful of worked examples to show the desired format.

A system prompt sets the role and rules for the model, for example telling it to answer only from provided context. Clear constraints reduce unwanted behaviour.

Chain-of-thought prompting asks the model to reason step by step before answering, which can improve results on multi-step problems.

The temperature parameter controls randomness. A temperature of zero makes output nearly deterministic, which is preferred for factual question answering, while higher values give more varied text.

When a program needs to parse the output, ask for a structured format such as JSON and validate the result. Telling the model what to do when information is missing helps prevent made-up answers.
