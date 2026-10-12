from __future__ import annotations

from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def read_bytes(name: str, data: bytes) -> str:
    """Decode an uploaded file's bytes into text based on its extension."""
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")


def load_directory(path: str | Path) -> list[tuple[str, str]]:
    """Load every supported file under ``path`` as (relative_name, text)."""
    root = Path(path)
    docs = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            docs.append((f.relative_to(root).as_posix(), read_bytes(f.name, f.read_bytes())))
    return docs
