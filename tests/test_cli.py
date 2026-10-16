"""O demo offline imprime as três pipelines e não chama provedor."""

from rag_study.cli import main


def test_cli_prints_three_pipelines_without_network(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "nota.txt").write_text(
        "A população de Vila Aurora é de 12450 habitantes.",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("CORPUS_PATH", str(corpus))
    assert main(["Quantos habitantes?"]) == 0
    output = capsys.readouterr().out
    assert "[no_context]" in output
    assert "[naive_rag]" in output
    assert "[rag_rerank]" in output
    assert output.count("resposta-fake") == 3
    assert "contextos: 0" in output
