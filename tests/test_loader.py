"""O loader só lê texto e markdown, em ordem de caminho."""

from rag_study.loader import load_corpus


def test_loader_sorts_text_files_and_skips_other_suffixes(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (nested / "a.md").write_text("alfa", encoding="utf-8")
    (tmp_path / "segredo.env").write_text("GEMINI_API_KEY=nao-ler", encoding="utf-8")
    documents = load_corpus(tmp_path)
    assert [document.doc_id for document in documents] == ["b.txt", "sub/a.md"]
    assert documents[1].text == "alfa"
