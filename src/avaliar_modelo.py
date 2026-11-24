from pathlib import Path
import argparse
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset_treino import (
    PASTA_PROJETO,
    PASTA_TOKENS_TREINO,
    MusicaDataset,
    criar_collate_fn,
)
from src.modelo import carregar_modelo
from src.treinar import PASTA_RESULTADOS_TREINO, obter_device


# Avalia um checkpoint: loss e perplexidade no total e por quadrante (Q1–Q4).
# O conjunto de teste só é usado com --incluir_teste, na avaliação final.

CHECKPOINT_PADRAO = PASTA_PROJETO / "checkpoints" / "lstm_final" / "melhor.pt"

EMOCOES = ["Q1", "Q2", "Q3", "Q4"]


@torch.no_grad()
def avaliar_por_emocao(modelo, loader, device):

    modelo.eval()

    criterio = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    acumulado = {
        chave: {"soma_loss": 0.0, "tokens": 0, "janelas": 0}
        for chave in ["total"] + EMOCOES
    }

    for lote in loader:

        entrada = lote["input_ids"].to(device)
        alvo = lote["labels"].to(device)

        logits, _ = modelo(entrada)

        loss_tokens = criterio(
            logits.float().reshape(-1, logits.shape[-1]),
            alvo.reshape(-1)
        ).reshape(alvo.shape)

        # Soma da loss e número de tokens válidos de cada janela.
        soma_por_janela = loss_tokens.sum(dim=1).tolist()
        tokens_por_janela = (alvo != -100).sum(dim=1).tolist()

        for emocao, soma, tokens in zip(
            lote["emotions"], soma_por_janela, tokens_por_janela
        ):
            for chave in ("total", emocao):
                acumulado[chave]["soma_loss"] += soma
                acumulado[chave]["tokens"] += tokens
                acumulado[chave]["janelas"] += 1

    resultados = {}

    for chave, valores in acumulado.items():

        if valores["tokens"] == 0:
            continue

        loss = valores["soma_loss"] / valores["tokens"]

        resultados[chave] = {
            "janelas": valores["janelas"],
            "tokens": valores["tokens"],
            "loss": loss,
            "perplexidade": math.exp(loss),
        }

    return resultados


def nome_da_avaliacao(caminho_checkpoint):

    caminho = Path(caminho_checkpoint)

    return f"{caminho.parent.name}_{caminho.stem}"


def avaliar_checkpoint(caminho_checkpoint, incluir_teste=False, tamanho_lote=32):

    device = obter_device()

    print("Checkpoint:", caminho_checkpoint)
    print("Device:", device)

    modelo, config = carregar_modelo(caminho_checkpoint, device)
    pad_id = config["modelo"]["pad_id"]

    print("Parâmetros:", f"{modelo.contar_parametros():,}")

    conjuntos = ["val"]
    if incluir_teste:
        conjuntos.append("test")

    collate_fn = criar_collate_fn(pad_id)
    linhas = []

    for conjunto in conjuntos:

        dataset = MusicaDataset(conjunto, PASTA_TOKENS_TREINO)

        loader = DataLoader(
            dataset,
            batch_size=tamanho_lote,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        resultados = avaliar_por_emocao(modelo, loader, device)

        print(f"\n{conjunto.upper()}")

        for chave, valores in resultados.items():

            print(
                f"  {chave:>5}: loss {valores['loss']:.4f} | "
                f"ppl {valores['perplexidade']:.2f} | "
                f"{valores['janelas']} janelas, {valores['tokens']} tokens"
            )

            linhas.append({
                "conjunto": conjunto,
                "quadrante": chave,
                "janelas": valores["janelas"],
                "tokens": valores["tokens"],
                "loss": round(valores["loss"], 6),
                "perplexidade": round(valores["perplexidade"], 4),
            })

    PASTA_RESULTADOS_TREINO.mkdir(parents=True, exist_ok=True)

    caminho_saida = (
        PASTA_RESULTADOS_TREINO
        / f"avaliacao_{nome_da_avaliacao(caminho_checkpoint)}.csv"
    )

    with open(caminho_saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    print("\nResultados salvos em:", caminho_saida)

    return linhas


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Avalia loss e perplexidade de um checkpoint por quadrante."
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PADRAO))
    parser.add_argument(
        "--incluir_teste",
        action="store_true",
        help="Inclui o conjunto de teste (apenas na avaliação final)."
    )
    parser.add_argument("--tamanho_lote", type=int, default=32)

    argumentos = parser.parse_args()

    avaliar_checkpoint(
        argumentos.checkpoint,
        argumentos.incluir_teste,
        argumentos.tamanho_lote
    )
