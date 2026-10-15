"""Lê .txt e .md de um diretório. A ordem é o caminho relativo, para o índice ser estável.

Extensões fora dessa lista são ignoradas: o corpus de exemplo não mistura
binário nem .env.
"""

from pathlib import Path

from rag_study.types import Document

_SUFFIXES = (".txt", ".md")


def load_corpus(directory: Path) -> list[Document]:
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus não encontrado: {directory}")
    paths = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _SUFFIXES
    ]
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    documents: list[Document] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        documents.append(
            Document(
                doc_id=relative,
                source=relative,
                text=path.read_text(encoding="utf-8"),
            )
        )
    return documents
