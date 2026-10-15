"""Erros de configuração e de provedor, separados de falhas de domínio."""


class ConfigError(ValueError):
    """Variável ausente, tipo inválido ou combinação proibida."""


class ProviderError(RuntimeError):
    """SDK ausente ou resposta vazia do provedor. Não inclui a chave."""
