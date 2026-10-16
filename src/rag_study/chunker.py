"""Dois cortes de documento: janela fixa e recursivo por separadores.

Com overlap 0, a concatenação dos pedaços reconstrói o texto. O recursivo
prefere parágrafo, linha, frase e palavra antes de cortar no caractere, e
nenhum pedaço passa de `size`.
"""

from rag_study.errors import ConfigError
from rag_study.types import Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def chunk_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    _validate(size, overlap)
    if text == "":
        return []
    if strategy == "fixed":
        return _chunk_fixed(text, size, overlap)
    if strategy == "recursive":
        units = _split_units(text, size, _SEPARATORS)
        return _merge(units, size, overlap)
    raise ConfigError(f"CHUNK_STRATEGY inválida: {strategy}. Use: fixed, recursive.")


def chunk_documents(
    documents: list[Document],
    strategy: str,
    size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        parts = chunk_text(document.text, strategy, size, overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}",
                    doc_id=document.doc_id,
                    source=document.source,
                    text=part,
                    index=index,
                )
            )
    return chunks


def _validate(size: int, overlap: int) -> None:
    if size < 1:
        raise ConfigError("CHUNK_SIZE precisa ser >= 1.")
    if overlap < 0 or overlap >= size:
        raise ConfigError("CHUNK_OVERLAP precisa ser >= 0 e menor que CHUNK_SIZE.")


def _chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _split_units(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if text == "":
        return []
    if len(text) <= size:
        return [text]
    if not separators:
        return _chunk_fixed(text, size, 0)
    separator = separators[0]
    if separator not in text:
        return _split_units(text, size, separators[1:])
    parts = text.split(separator)
    units: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        unit = part if index == last else part + separator
        if unit == "":
            continue
        if len(unit) <= size:
            units.append(unit)
        else:
            units.extend(_split_units(unit, size, separators[1:]))
    return units


def _merge(units: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current == "":
            current = unit
            continue
        if len(current) + len(unit) <= size:
            current += unit
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = tail + unit if len(tail) + len(unit) <= size else unit
    if current:
        chunks.append(current)
    return chunks
