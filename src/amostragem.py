import torch


# IDs confirmados no tokenizer_remi.json
# Estes valores servem como referência; no gerador real devem ser obtidos de tokenizer.vocab.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
EMOTION_IDS = (4, 5, 6, 7)
IDS_PROIBIDOS_PADRAO = (PAD_ID, BOS_ID, MASK_ID, *EMOTION_IDS)


def aplicar_tokens_proibidos(
    logits: torch.Tensor,
    ids_proibidos: list[int] | tuple[int, ...],
) -> torch.Tensor:
    #Retorna uma cópia dos logits com os IDs proibidos definidos como -inf
    logits_filtrados = logits.clone()

    ids_convertidos = []
    for token_id in ids_proibidos:
        try:
            token_id_convertido = int(token_id)
        except (TypeError, ValueError, OverflowError):
            raise TypeError("IDs proibidos devem ser inteiros") from None

        if token_id_convertido < 0 or token_id_convertido >= logits.shape[-1]:
            raise ValueError(f"ID proibido fora do vocabulário: {token_id_convertido}")

        ids_convertidos.append(token_id_convertido)

    if ids_convertidos:
        indices = torch.tensor(ids_convertidos, dtype=torch.long, device=logits.device)
        logits_filtrados[:, indices] = float("-inf")

    return logits_filtrados


def amostrar_proximo_token(
    logits: torch.Tensor,
    temperatura: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    ids_proibidos: list[int] | tuple[int, ...] = (),
) -> torch.Tensor:
    #Amostra o próximo token usando temperatura, top-k e top-p
    if logits.ndim != 2:
        raise ValueError("logits deve ter formato (lote, tamanho_do_vocabulario)")
    if logits.shape[0] == 0:
        raise ValueError("o lote não pode estar vazio")
    if temperatura < 0:
        raise ValueError("temperatura deve ser maior ou igual a zero")

    # Temperatura zero significa escolha gulosa entre os tokens permitidos.
    if temperatura == 0:
        logits = aplicar_tokens_proibidos(logits, ids_proibidos)
        return logits.argmax(dim=-1)

    if top_k is not None:
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k deve ser um inteiro maior ou igual a zero")

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p deve estar entre 0 e 1")

    logits = aplicar_tokens_proibidos(logits, ids_proibidos)

    # 1. Temperatura
    logits = logits / temperatura

    # 2. Top-k: 0 significa desligado; mantém os k maiores logits.
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        indices = torch.topk(logits, top_k, dim=-1).indices
        mascara = torch.zeros_like(logits, dtype=torch.bool)
        mascara.scatter_(1, indices, True)
        logits = logits.masked_fill(~mascara, float("-inf"))

    # 3. Top-p (núcleo): mantém o menor conjunto cuja probabilidade acumulada alcança p.
    if top_p is not None:
        logits_ordenados, indices_ordenados = torch.sort(logits, descending=True, dim=-1)

        probs_ordenadas = torch.softmax(logits_ordenados, dim=-1)
        cumulativa = torch.cumsum(probs_ordenadas, dim=-1)

        remover = cumulativa > top_p
        remover[..., 1:] = remover[..., :-1].clone()
        remover[..., 0] = False

        mascara_remover = torch.zeros_like(logits, dtype=torch.bool)
        mascara_remover.scatter_(1, indices_ordenados, remover)
        logits = logits.masked_fill(mascara_remover, float("-inf"))

    # 4. Softmax
    probabilidades = torch.softmax(logits, dim=-1)

    # 5. Multinomial
    return torch.multinomial(probabilidades, num_samples=1).squeeze(-1)


if __name__ == "__main__":
    logits_teste = torch.tensor([[4.0, 2.0, 1.0, 0.5]])

    # Temperatura zero deve ser igual ao argmax.
    guloso = amostrar_proximo_token(logits_teste, temperatura=0)
    assert torch.equal(guloso, logits_teste.argmax(dim=-1))

    # Top-k=1 e top-k=0 sempre aceitam o token mais provável sem gerar NaN.
    amostras_top_k = torch.stack([
        amostrar_proximo_token(logits_teste, top_k=1)
        for _ in range(100)
    ])
    assert torch.all(amostras_top_k == 0)

    amostras_top_k_zero = torch.stack([
        amostrar_proximo_token(logits_teste, top_k=0)
        for _ in range(100)
    ])
    assert amostras_top_k_zero.shape == (100, 1)

    # Temperatura baixa concentra a distribuição no argmax.
    amostras_baixa_temperatura = torch.stack([
        amostrar_proximo_token(logits_teste, temperatura=0.01)
        for _ in range(100)
    ])
    assert torch.all(amostras_baixa_temperatura == 0)

    # Top-p pequeno mantém apenas o núcleo mais provável.
    amostras_top_p = torch.stack([
        amostrar_proximo_token(logits_teste, top_p=0.1)
        for _ in range(100)
    ])
    assert torch.all(amostras_top_p == 0)

    # Top-p=1 sem top-k reproduz aproximadamente o softmax após 10.000 amostras.
    torch.manual_seed(42)
    amostras_frequencia = amostrar_proximo_token(
        logits_teste.expand(10_000, -1),
        top_p=1.0,
    )
    frequencias = torch.bincount(amostras_frequencia, minlength=4) / 10_000
    probabilidades_esperadas = torch.softmax(logits_teste, dim=-1).squeeze(0)
    assert torch.allclose(frequencias, probabilidades_esperadas, atol=0.02)

    # Cada linha de um lote deve usar seus próprios logits.
    logits_lote = torch.tensor([
        [4.0, 1.0, 0.0],
        [0.0, 1.0, 4.0],
    ])
    amostras_lote = amostrar_proximo_token(logits_lote, temperatura=0)
    assert torch.equal(amostras_lote, torch.tensor([0, 2]))

    # Teste com tokens proibidos.
    logits_proibido = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]])
    amostras_proibidas = torch.stack([
        amostrar_proximo_token(
            logits_proibido,
            ids_proibidos=IDS_PROIBIDOS_PADRAO,
        )
        for _ in range(100)
    ])
    proibidos = torch.tensor(IDS_PROIBIDOS_PADRAO)
    assert not torch.isin(amostras_proibidas, proibidos).any()

    # EOS não pertence à máscara padrão e pode ser sorteado quando permitido.
    logits_eos = torch.tensor([[0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    amostras_eos = torch.stack([
        amostrar_proximo_token(
            logits_eos,
            temperatura=0,
            ids_proibidos=IDS_PROIBIDOS_PADRAO,
        )
        for _ in range(100)
    ])
    assert torch.all(amostras_eos == EOS_ID)

    print("Testes de amostragem concluídos:")
    print("- Modo guloso: OK")
    print("- Top-k=0 e top-k=1: OK")
    print("- Temperatura baixa: OK")
    print("- Top-p pequeno e top-p=1.0: OK")
    print("- Amostragem em lote: OK")
    print("- Tokens proibidos: OK")
