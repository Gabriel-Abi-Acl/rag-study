"""Contrato dos chunkers: tamanho, overlap e reconstrução com overlap 0."""

import pytest

from rag_study.chunker import chunk_documents, chunk_text
from rag_study.errors import ConfigError
from rag_study.types import Document


def test_fixed_windows_cover_text_and_respect_overlap():
    text = "abcdefghij"
    assert chunk_text(text, "fixed", size=4, overlap=0) == ["abcd", "efgh", "ij"]
    assert chunk_text(text, "fixed", size=6, overlap=2) == ["abcdef", "efghij"]
    assert chunk_text("abc", "fixed", size=5, overlap=2) == ["abc"]


def test_fixed_overlap_zero_reconstructs_text():
    text = "Vila Aurora\n\nfevereiro"
    parts = chunk_text(text, "fixed", size=7, overlap=0)
    assert "".join(parts) == text
    assert all(len(part) <= 7 for part in parts)


def test_recursive_prefers_paragraph_boundary():
    text = "AAAA\n\nBBBB\n\nCCCC"
    assert chunk_text(text, "recursive", size=6, overlap=0) == [
        "AAAA\n\n",
        "BBBB\n\n",
        "CCCC",
    ]
    assert chunk_text(text, "recursive", size=12, overlap=0) == [
        "AAAA\n\nBBBB\n\n",
        "CCCC",
    ]


def test_recursive_overlap_reuses_tail_when_it_fits():
    text = "AAAA\n\nBBBB\n\nCCCC"
    assert chunk_text(text, "recursive", size=12, overlap=4) == [
        "AAAA\n\nBBBB\n\n",
        "BB\n\nCCCC",
    ]


def test_recursive_falls_back_to_characters_and_stays_within_size():
    parts = chunk_text("abcdefghij", "recursive", size=4, overlap=0)
    assert parts == ["abcd", "efgh", "ij"]
    assert "".join(parts) == "abcdefghij"


def test_empty_text_yields_no_chunks():
    assert chunk_text("", "fixed", size=10, overlap=0) == []
    assert chunk_text("", "recursive", size=10, overlap=2) == []


def test_invalid_size_or_strategy_raises():
    with pytest.raises(ConfigError):
        chunk_text("a", "fixed", size=0, overlap=0)
    with pytest.raises(ConfigError):
        chunk_text("abcd", "fixed", size=4, overlap=4)
    with pytest.raises(ConfigError):
        chunk_text("abcd", "semantic", size=4, overlap=0)


def test_documents_get_stable_chunk_ids():
    documents = [
        Document(doc_id="a.txt", source="a.txt", text="AAA\n\nBBB"),
        Document(doc_id="b.md", source="b.md", text="tres"),
    ]
    chunks = chunk_documents(documents, "recursive", size=5, overlap=0)
    assert [(chunk.chunk_id, chunk.text) for chunk in chunks] == [
        ("a.txt:0", "AAA\n\n"),
        ("a.txt:1", "BBB"),
        ("b.md:0", "tres"),
    ]
