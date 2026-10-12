# Embeddings and Vector Search

An embedding is a dense vector that represents the meaning of a piece of text. Texts with similar meaning are mapped to vectors that are close together in the embedding space.

Closeness is usually measured with cosine similarity. If all vectors are normalised to unit length, cosine similarity equals the inner product, which is cheap to compute.

FAISS is a library from Meta for fast similarity search over dense vectors. The simplest index, IndexFlatIP, performs exact brute-force search and is a good choice for up to a few hundred thousand vectors.

For much larger collections, approximate indexes such as IVF or HNSW trade a small loss in recall for large speed gains. IVF clusters the vectors and only searches the closest clusters.

The sentence-transformers library provides pretrained models such as all-MiniLM-L6-v2, which produces 384-dimensional embeddings and runs comfortably on a CPU. A vector database stores embeddings together with metadata so results can be filtered.
