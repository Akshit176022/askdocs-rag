FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 ASKDOCS_STORAGE_DIR=/data

# Install dependencies first so Docker can cache this layer between code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY askdocs ./askdocs
COPY data ./data

VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "askdocs.api:app", "--host", "0.0.0.0", "--port", "8000"]
