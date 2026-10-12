# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retriever with a language model. Instead of relying only on what the model memorised during training, the system first searches an external knowledge base and passes the most relevant passages to the model as context.

A typical RAG pipeline has four steps. First, documents are split into chunks. Second, each chunk is converted into an embedding and stored in a vector index. Third, at question time the query is embedded and the nearest chunks are retrieved. Fourth, the language model writes an answer grounded in those chunks.

RAG reduces hallucination because the model can quote retrieved evidence, and answers can cite their sources. It also lets teams update knowledge by re-indexing documents, without retraining the model.

Chunk size is a key design choice. Small chunks give precise matches but may lose surrounding context, while large chunks keep context but dilute the embedding. Overlapping chunks help when an important fact straddles a boundary.

Hybrid retrieval combines dense vector search with keyword search such as BM25. The two rankings can be merged with Reciprocal Rank Fusion, which rewards documents that rank well in either list.
