# Transformers and Large Language Models

The Transformer is a neural network architecture built around self-attention. Self-attention lets every token in a sequence weigh the relevance of every other token when computing its representation.

Text is first split into tokens by a tokenizer, often using byte-pair encoding. Each token is mapped to an embedding, and positional information is added so the model knows word order.

Encoder models such as BERT read the whole input at once and are well suited to classification and embeddings. Decoder models such as GPT generate text one token at a time, each conditioned on the previous tokens.

A model's context window is the maximum number of tokens it can read in one request. Anything beyond the window must be truncated or summarised, which is one reason retrieval is useful for long documents.

Fine-tuning adapts a pretrained model to a task. Parameter-efficient methods such as LoRA train small adapter matrices instead of all weights, which greatly reduces memory use.
